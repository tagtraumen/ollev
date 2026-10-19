//! Lowerings kept from one call to the next: a trace's partial scenes and its result share
//! most objects, so each object is lowered once. An object's lowering depends on itself, the
//! clips it lies in and the lowering's setting; the clips it adds are kept relative to where
//! they were added and placed after the clips there are when it is replayed.

use std::sync::Arc;

use super::{Capabilities, Lowering};
use crate::HashMap;
use crate::model::{Clip, Curve, FillRule, Geometry, Object, Paint};

/// The lowerings of objects and clips under one setting, each with the last lowering that
/// used it; what two lowerings in a row did not use is dropped.
#[derive(Debug, Default)]
pub(crate) struct Memo {
    setting: Option<Setting>,
    objects: HashMap<Vec<u8>, Lowered>,
    clips: HashMap<Vec<u8>, LoweredClip>,
    generation: u64,
}

/// What an object's lowering depends on besides the object and its clips.
#[derive(Debug)]
struct Setting {
    host: Capabilities,
    width: usize,
    height: usize,
    price: Option<u64>,
    tolerance: Option<Arc<Vec<f32>>>,
}

impl Setting {
    fn of(l: &Lowering) -> Self {
        Self {
            host: l.host,
            width: l.width,
            height: l.height,
            price: l.price.map(f64::to_bits),
            tolerance: l.tolerance.clone(),
        }
    }

    fn same(&self, other: &Self) -> bool {
        let tolerance = match (&self.tolerance, &other.tolerance) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                Arc::ptr_eq(a, b)
                    || (a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()))
            }
            _ => false,
        };
        tolerance
            && (self.host, self.width, self.height, self.price)
                == (other.host, other.width, other.height, other.price)
    }
}

/// A clip's loops and rule as lowered, and the last lowering that used them.
#[derive(Debug)]
struct LoweredClip {
    loops: Vec<Vec<Curve>>,
    rule: FillRule,
    used: u64,
}

/// An object as lowered: the objects written and the clips added after the first `base`.
#[derive(Debug)]
struct Lowered {
    base: usize,
    objects: Vec<Object>,
    added: Vec<Clip>,
    used: u64,
}

impl Memo {
    /// Starts a lowering under `l`'s setting, forgetting what another setting lowered.
    pub(super) fn begin(&mut self, l: &Lowering) {
        let setting = Setting::of(l);
        if !self.setting.as_ref().is_some_and(|s| s.same(&setting)) {
            self.objects.clear();
            self.clips.clear();
            self.setting = Some(setting);
        }
        self.generation += 1;
    }

    /// Drops what neither this lowering nor the one before used.
    pub(super) fn sweep(&mut self) {
        let g = self.generation;
        self.objects.retain(|_, o| o.used + 1 >= g);
        self.clips.retain(|_, c| c.used + 1 >= g);
    }

    /// The object lowered by `lower`, or as it was before with its clips placed after `l`'s;
    /// a quick lowering takes a full one where there is one.
    pub(super) fn object(
        &mut self,
        l: &mut Lowering,
        o: Object,
        lower: impl FnOnce(&mut Lowering, Object) -> Vec<Object>,
    ) -> Vec<Object> {
        let mut key = key(&o, &l.clips);
        if l.quick && !self.objects.contains_key(&key) {
            key[0] = QUICK;
        }
        let generation = self.generation;
        let base = l.clips.len();
        let kept = match self.objects.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let objects = lower(l, o);
                let added = l.clips.split_off(base);
                e.insert(Lowered {
                    base,
                    objects,
                    added,
                    used: generation,
                })
            }
        };
        kept.used = generation;
        let at = |c: usize| if c >= kept.base { c - kept.base + base } else { c };
        l.clips.extend(kept.added.iter().map(|c| Clip {
            parent: c.parent.map(at),
            ..c.clone()
        }));
        kept.objects
            .iter()
            .map(|o| Object {
                clip: o.clip.map(at),
                ..o.clone()
            })
            .collect()
    }

    /// Whether the full lowering of the object of `key` is kept, marking it used.
    pub(super) fn touch(&mut self, key: &[u8]) -> bool {
        let generation = self.generation;
        self.objects
            .get_mut(key)
            .map(|kept| kept.used = generation)
            .is_some()
    }

    /// Keeps the full lowering of the object of `key`: the objects written and the clips it
    /// added after the first `base`.
    pub(super) fn put(&mut self, key: Vec<u8>, base: usize, objects: Vec<Object>, added: Vec<Clip>) {
        let used = self.generation;
        self.objects.insert(
            key,
            Lowered {
                base,
                objects,
                added,
                used,
            },
        );
    }

    /// The clip's loops and rule as `lower` gives them, or as it gave them before; a quick
    /// lowering takes a full one where there is one.
    pub(super) fn clip(&mut self, c: &mut Clip, quick: bool, lower: impl FnOnce(&mut Clip)) {
        let mut key = vec![FULL];
        loops_key(&mut key, &c.loops);
        if quick && !self.clips.contains_key(&key) {
            key[0] = QUICK;
        }
        key.push(u8::from(c.rule == FillRule::EvenOdd));
        let generation = self.generation;
        let kept = self.clips.entry(key).or_insert_with(|| {
            let mut lowered = c.clone();
            lower(&mut lowered);
            LoweredClip {
                loops: lowered.loops,
                rule: lowered.rule,
                used: generation,
            }
        });
        kept.used = generation;
        (c.loops, c.rule) = (kept.loops.clone(), kept.rule);
    }
}

/// The first byte of the key of a full lowering, and of a quick one.
const FULL: u8 = 0;
const QUICK: u8 = 1;

/// The object and the clips it lies in, as bytes: equal bytes, equal lowerings. Every run of
/// values is led by its length, so no two objects read alike.
pub(super) fn key(o: &Object, clips: &[Clip]) -> Vec<u8> {
    let mut key = vec![FULL];
    let put = |key: &mut Vec<u8>, v: u64| key.extend_from_slice(&v.to_le_bytes());
    let text = |key: &mut Vec<u8>, t: String| {
        put(key, t.len() as u64);
        key.extend_from_slice(t.as_bytes());
    };
    match &o.geometry {
        Geometry::Fill { loops, rule } => {
            key.push(u8::from(*rule == FillRule::EvenOdd));
            loops_key(&mut key, loops);
        }
        Geometry::Stroke(s) => {
            key.push(2);
            loops_key(&mut key, std::slice::from_ref(&s.centerline));
            put(&mut key, s.width.to_bits());
            put(&mut key, s.corners.len() as u64);
            key.extend(s.corners.iter().map(|&c| u8::from(c)));
            let (pattern, offset) = s.dashes.clone().unwrap_or((Vec::new(), f64::NAN));
            put(&mut key, u64::from(s.dashes.is_some()));
            put(&mut key, pattern.len() as u64);
            pattern.iter().for_each(|v| put(&mut key, v.to_bits()));
            put(&mut key, offset.to_bits());
            text(&mut key, format!("{} {:?} {:?}", s.closed, s.cap, s.join));
        }
    }
    match &o.paint {
        Paint::Solid(c) => {
            key.push(0);
            c.to_array().iter().for_each(|v| put(&mut key, v.to_bits()));
        }
        Paint::Gradient(g) => {
            key.push(1);
            text(&mut key, format!("{g:?}"));
        }
    }
    put(&mut key, o.opacity.to_bits());
    put(&mut key, o.blur.to_bits());
    let mut chain = o.clip;
    while let Some(c) = chain {
        key.push(1);
        put(&mut key, c as u64);
        key.push(u8::from(clips[c].rule == FillRule::EvenOdd));
        loops_key(&mut key, &clips[c].loops);
        chain = clips[c].parent;
    }
    key.push(0);
    key
}

/// The loops as bytes: their count, then each loop's length and points.
fn loops_key(key: &mut Vec<u8>, loops: &[Vec<Curve>]) {
    key.extend_from_slice(&(loops.len() as u64).to_le_bytes());
    for l in loops {
        key.extend_from_slice(&(l.len() as u64).to_le_bytes());
        for p in l.iter().flatten() {
            key.extend_from_slice(&p.x.to_bits().to_le_bytes());
            key.extend_from_slice(&p.y.to_bits().to_le_bytes());
        }
    }
}
