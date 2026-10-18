//! The model fitted to what a host can draw: each feature the host lacks is rewritten in the
//! ones it has, every object at the coarsest form whose render misses the model's by less than
//! a segment's price (the verify loop's rule, judged on the object alone over its box).
//!
//! The rewrites, in the order they run (each may use what the later ones remove):
//! a path gradient as bands along its curve ([`bands`]); a radial focus dropped, or the
//! gradient as nested discs ([`focus`]); a blur, a ramp's alpha or a clip baked into nested
//! shapes traced from the object's coverage ([`layers`]); dashes cut into strokes ([`dashes`]);
//! even-odd loops wound for the nonzero rule ([`nonzero`]).

use crate::model::{Clip, Item, Model, Object};

pub mod bands;
mod dashes;
mod focus;
pub(crate) mod layers;
mod nonzero;
mod render;
#[cfg(test)]
mod tests;

/// What a host draws beyond filled and stroked cubic paths painted with a solid colour or a
/// linear or centred radial gradient, at an opacity. Every feature off is lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Fills by the even-odd rule; without, loops are wound for the nonzero rule.
    pub even_odd: bool,
    /// A gradient along a curve ([`crate::model::GradientShape::Path`]).
    pub path_gradients: bool,
    /// A Gaussian blur of an object's coverage ([`Object::blur`]).
    pub blur: bool,
    /// A radial gradient's focus away from its centre.
    pub radial_focus: bool,
    /// Clips ([`Object::clip`], [`Model::clip`]).
    pub clips: bool,
    /// A stroke's dash pattern.
    pub dashes: bool,
    /// Gradient stops with alpha, interpolated in straight colour as SVG does.
    pub transparent_stops: bool,
}

impl Capabilities {
    /// Everything the model uses: SVG, which needs no lowering.
    pub const ALL: Self = Self {
        even_odd: true,
        path_gradients: true,
        blur: true,
        radial_focus: true,
        clips: true,
        dashes: true,
        transparent_stops: true,
    };
    /// Only plain filled and stroked paths.
    pub const NONE: Self = Self {
        even_odd: false,
        path_gradients: false,
        blur: false,
        radial_focus: false,
        clips: false,
        dashes: false,
        transparent_stops: false,
    };
}

impl Default for Capabilities {
    fn default() -> Self {
        Self::ALL
    }
}

/// What the lowering did: objects rewritten per feature, the objects written for them, and
/// those whose best form still misses the model by more than a segment's price.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub path_gradients: usize,
    pub foci: usize,
    pub layered: usize,
    pub dashed: usize,
    pub wound: usize,
    /// Objects written in place of the rewritten ones.
    pub written: usize,
    pub missed: usize,
    /// The largest miss of a rewritten object, in segment prices.
    pub worst: f64,
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{} path gradients, {} foci, {} layered, {} dashed, {} wound; {} objects written, {} missed, worst {:.2}",
            self.path_gradients,
            self.foci,
            self.layered,
            self.dashed,
            self.wound,
            self.written,
            self.missed,
            self.worst
        )
    }
}

/// The model in what `host` draws, and what it took.
pub fn lower(model: &Model, host: &Capabilities) -> (Model, Report) {
    let mut out = model.clone();
    let mut report = Report::default();
    if *host == Capabilities::ALL {
        return (out, report);
    }
    // Without clips the drawing's own clip goes onto every object, so instances are drawn out
    if !host.clips
        && let Some(top) = out.clip.take()
    {
        out.items = expanded(&out, &out.items);
        out.definitions.clear();
        for c in 0..out.clips.len() {
            if c != top && out.clips[c].parent.is_none() {
                out.clips[c].parent = Some(top);
            }
        }
        clip_all(&mut out.items, top);
    }
    let mut lowering = Lowering {
        host: *host,
        width: model.width,
        height: model.height,
        price: model.fidelity.as_ref().map(|f| f.segment_price),
        tolerance: model.fidelity.as_ref().and_then(|f| f.tolerance.clone()),
        clips: std::mem::take(&mut out.clips),
        report: &mut report,
    };
    out.items = lowering.items(std::mem::take(&mut out.items));
    out.definitions = std::mem::take(&mut out.definitions)
        .into_iter()
        .map(|d| lowering.items(d))
        .collect();
    let mut clips = std::mem::take(&mut lowering.clips);
    if !host.even_odd {
        clips.iter_mut().for_each(|c| nonzero::clip(&lowering, c));
    }
    out.clips = clips;
    (out, report)
}

/// The items with every instance drawn out as the objects of its definition.
fn expanded(model: &Model, items: &[Item]) -> Vec<Item> {
    let mut out = Vec::new();
    for item in items {
        match item {
            Item::Group { layer, items } => out.push(Item::Group {
                layer: *layer,
                items: expanded(model, items),
            }),
            Item::Object(_) => out.push(item.clone()),
            Item::Instance(_) | Item::Repeat { .. } => {
                let one = Model {
                    items: vec![item.clone()],
                    ..model.clone()
                };
                out.extend(
                    one.drawn()
                        .into_iter()
                        .map(|(_, o)| Item::Object(Box::new(o))),
                );
            }
        }
    }
    out
}

/// Every object drawn under clip `top`: objects without a clip take it, the rest have it as
/// their clips' root.
fn clip_all(items: &mut [Item], top: usize) {
    for item in items {
        match item {
            Item::Object(o) if o.clip.is_none() => o.clip = Some(top),
            Item::Group { items, .. } => clip_all(items, top),
            _ => {}
        }
    }
}

/// One lowering of a model: the host, the price an object's miss is judged by, and the clips,
/// to which the rewrites add.
struct Lowering<'a> {
    host: Capabilities,
    width: usize,
    height: usize,
    /// The price of a segment; `None` takes the finest form of each rewrite.
    price: Option<f64>,
    /// The picture's uncertainty per pixel, below which a miss is no error.
    tolerance: Option<std::sync::Arc<Vec<f32>>>,
    clips: Vec<Clip>,
    report: &'a mut Report,
}

impl Lowering<'_> {
    fn items(&mut self, items: Vec<Item>) -> Vec<Item> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Item::Object(o) => {
                    let lowered = self.object(*o);
                    out.extend(lowered.into_iter().map(|o| Item::Object(Box::new(o))));
                }
                Item::Group { layer, items } => out.push(Item::Group {
                    layer,
                    items: self.items(items),
                }),
                other => out.push(other),
            }
        }
        out
    }

    /// The object in the host's features: each rewrite in turn over what the last one wrote.
    fn object(&mut self, o: Object) -> Vec<Object> {
        let original = o.clone();
        let mut objects = vec![o];
        let mut changed = false;
        if !self.host.path_gradients {
            objects = self.each(objects, &mut changed, |l, o| bands::lower(l, o));
        }
        if !self.host.radial_focus {
            objects = self.each(objects, &mut changed, |l, o| focus::lower(l, o));
        }
        objects = self.each(objects, &mut changed, |l, o| layers::lower(l, o));
        if !self.host.dashes {
            objects = self.each(objects, &mut changed, |l, o| {
                let cut = dashes::lower(o);
                l.report.dashed += usize::from(cut.is_some());
                cut
            });
        }
        if !self.host.even_odd {
            objects = self.each(objects, &mut changed, |l, o| nonzero::lower(l, o));
        }
        if changed {
            self.report.written += objects.len();
            let miss = self.miss(std::slice::from_ref(&original), &objects);
            if let Some(price) = self.price {
                self.report.missed += usize::from(miss > price);
                self.report.worst = self.report.worst.max(miss / price.max(f64::MIN_POSITIVE));
            }
        }
        objects
    }

    /// Each object rewritten by `f`, where it rewrites it (`Some`), else kept.
    fn each(
        &mut self,
        objects: Vec<Object>,
        changed: &mut bool,
        mut f: impl FnMut(&mut Self, &Object) -> Option<Vec<Object>>,
    ) -> Vec<Object> {
        let mut out = Vec::with_capacity(objects.len());
        for o in objects {
            match f(self, &o) {
                Some(rewritten) => {
                    *changed = true;
                    out.extend(rewritten);
                }
                None => out.push(o),
            }
        }
        out
    }

    /// Adds a clip, returning its place.
    fn clip(&mut self, clip: Clip) -> usize {
        self.clips.push(clip);
        self.clips.len() - 1
    }

    /// Of candidate forms of `original`, coarsest first, the first whose miss is within a
    /// segment's price, else the one that missed least (the finest without a price).
    fn coarsest(
        &mut self,
        original: &Object,
        forms: impl IntoIterator<Item = Vec<Object>>,
    ) -> Vec<Object> {
        let Some(price) = self.price else {
            return forms.into_iter().last().unwrap_or_default();
        };
        let mut best: Option<(f64, Vec<Object>)> = None;
        for form in forms {
            let miss = self.miss(std::slice::from_ref(original), &form);
            if miss <= price {
                return form;
            }
            if best.as_ref().is_none_or(|b| miss < b.0) {
                best = Some((miss, form));
            }
        }
        best.map(|b| b.1).unwrap_or_default()
    }

    /// How much `form` misses `original` by over the original's box, each drawn alone over
    /// white: the squared colour distance beyond the anti-aliasing pixel and the noise.
    fn miss(&self, original: &[Object], form: &[Object]) -> f64 {
        render::miss(self, original, form)
    }
}
