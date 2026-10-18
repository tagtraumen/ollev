//! A dashed stroke, for a host without dashes: each dash cut from the centerline as a stroke
//! of its own, with SVG's rules (an odd pattern repeats twice, the offset is how far into the
//! pattern the centerline starts, every dash takes the caps); a dash of no length is the cap
//! alone, a dot for a round cap and a square for a square one.

use glam::DVec2;

use crate::model::{Cap, Curve, Geometry, Object, StrokeGeometry};

/// Samples per cubic its length is measured by.
const SAMPLES: usize = 32;

pub(super) fn lower(o: &Object) -> Option<Vec<Object>> {
    let Geometry::Stroke(s) = &o.geometry else {
        return None;
    };
    let (array, offset) = s.dashes.as_ref()?;
    let mut pattern = array.clone();
    if pattern.len() % 2 == 1 {
        pattern.extend_from_within(..);
    }
    let period: f64 = pattern.iter().sum();
    let solid = Object {
        geometry: Geometry::Stroke(StrokeGeometry {
            dashes: None,
            ..s.clone()
        }),
        ..o.clone()
    };
    if period <= 0. || pattern.iter().any(|&v| v < 0.) {
        return Some(vec![solid]);
    }
    let path = Measured::new(&s.centerline);
    // Walk the pattern from the offset into it
    let mut at = -offset.rem_euclid(period);
    let mut out = Vec::new();
    'walk: loop {
        for (k, &length) in pattern.iter().enumerate() {
            let (a, b) = (at, at + length);
            at = b;
            if a >= path.length {
                break 'walk;
            }
            if k % 2 == 1 || b < 0. {
                continue;
            }
            let (a, b) = (a.max(0.), b.min(path.length));
            out.extend(dash(o, s, &path, a, b));
        }
    }
    Some(out)
}

/// The dash from `a` to `b` along the centerline: its stroke, or its cap where it has no length.
fn dash(o: &Object, s: &StrokeGeometry, path: &Measured, a: f64, b: f64) -> Option<Object> {
    let geometry = if b > a {
        let (centerline, corners) = path.cut(s, a, b);
        Geometry::Stroke(StrokeGeometry {
            centerline,
            closed: false,
            corners,
            dashes: None,
            ..s.clone()
        })
    } else {
        let (p, tangent) = path.at(a);
        let r = 0.5 * s.width;
        let (u, v) = (tangent * r, tangent.perp() * r);
        let loops = match s.cap {
            Cap::Butt => return None,
            Cap::Round => {
                let k = super::focus::KAPPA;
                let axes = [u, v, -u, -v];
                (0..4)
                    .map(|j| {
                        let (x, y) = (axes[j], axes[(j + 1) % 4]);
                        [p + x, p + x + y * k, p + y + x * k, p + y]
                    })
                    .collect()
            }
            Cap::Square => {
                let corners = [p + u + v, p - u + v, p - u - v, p + u - v];
                (0..4)
                    .map(|j| super::bands::line(corners[j], corners[(j + 1) % 4]))
                    .collect()
            }
        };
        Geometry::Fill {
            loops: vec![loops],
            rule: crate::model::FillRule::NonZero,
        }
    };
    Some(Object {
        geometry,
        ..o.clone()
    })
}

/// A centerline measured along its length: per cubic, the length at each sample.
struct Measured<'a> {
    cubics: &'a [Curve],
    lengths: Vec<[f64; SAMPLES + 1]>,
    starts: Vec<f64>,
    length: f64,
}

pub(super) fn eval(c: &Curve, t: f64) -> DVec2 {
    let u = 1. - t;
    c[0] * (u * u * u) + c[1] * (3. * u * u * t) + c[2] * (3. * u * t * t) + c[3] * (t * t * t)
}

/// The part of `c` from `t0` to `t1`.
fn part(c: &Curve, t0: f64, t1: f64) -> Curve {
    // The tail from t0, then its head up to t1 rescaled
    let split = |c: &Curve, t: f64| -> (Curve, Curve) {
        let l = |a: DVec2, b: DVec2| a + (b - a) * t;
        let (p01, p12, p23) = (l(c[0], c[1]), l(c[1], c[2]), l(c[2], c[3]));
        let (p012, p123) = (l(p01, p12), l(p12, p23));
        let p = l(p012, p123);
        ([c[0], p01, p012, p], [p, p123, p23, c[3]])
    };
    let tail = split(c, t0).1;
    let u = if t0 < 1. { (t1 - t0) / (1. - t0) } else { 0. };
    split(&tail, u.clamp(0., 1.)).0
}

impl<'a> Measured<'a> {
    fn new(cubics: &'a [Curve]) -> Self {
        let mut starts = Vec::with_capacity(cubics.len());
        let mut length = 0.;
        let lengths = cubics
            .iter()
            .map(|c| {
                starts.push(length);
                let mut l = [0.; SAMPLES + 1];
                let mut last = c[0];
                for (s, v) in l.iter_mut().enumerate().skip(1) {
                    let p = eval(c, s as f64 / SAMPLES as f64);
                    length += p.distance(last);
                    *v = length - starts[starts.len() - 1];
                    last = p;
                }
                l
            })
            .collect();
        Measured {
            cubics,
            lengths,
            starts,
            length,
        }
    }

    /// The cubic and its parameter at length `s` along the centerline.
    fn locate(&self, s: f64) -> (usize, f64) {
        let k = self
            .starts
            .partition_point(|&start| start <= s)
            .saturating_sub(1);
        let l = &self.lengths[k];
        let local = s - self.starts[k];
        let j = l.partition_point(|&v| v <= local).clamp(1, SAMPLES);
        let span = (l[j] - l[j - 1]).max(f64::MIN_POSITIVE);
        let t = ((j - 1) as f64 + ((local - l[j - 1]) / span).clamp(0., 1.)) / SAMPLES as f64;
        (k, t)
    }

    /// The point at length `s` and the unit tangent there.
    fn at(&self, s: f64) -> (DVec2, DVec2) {
        let (k, t) = self.locate(s);
        let c = &self.cubics[k];
        let (a, b) = (eval(c, (t - 1e-3).max(0.)), eval(c, (t + 1e-3).min(1.)));
        (eval(c, t), (b - a).normalize_or(DVec2::X))
    }

    /// The centerline from length `a` to `b`, and which of its joints are corners.
    fn cut(&self, s: &StrokeGeometry, a: f64, b: f64) -> (Vec<Curve>, Vec<bool>) {
        let ((ka, ta), (kb, tb)) = (self.locate(a), self.locate(b));
        let mut cubics = Vec::new();
        let mut corners = Vec::new();
        for k in ka..=kb {
            let (t0, t1) = (if k == ka { ta } else { 0. }, if k == kb { tb } else { 1. });
            if t1 <= t0 && ka != kb {
                continue;
            }
            corners.push(k != ka && s.corners.get(k).copied().unwrap_or(false));
            cubics.push(part(&self.cubics[k], t0, t1));
        }
        (cubics, corners)
    }
}
