//! A blur, a ramp's alpha or a clip the host cannot draw, baked into nested shapes: the
//! object's alpha over its box (its coverage, blurred, clipped and times the ramp's alpha, as
//! the host lacks each) cut at evenly spaced levels, each level's region traced into loops and
//! painted with the paint made opaque, at the opacity that brings the stack up to that level.
//! Layers of one paint stacked so composite to the level exactly, whatever lies below.

use crate::render::{FillRule, Gradient as Ramp, Stops};
use glam::{DVec2, IVec2, Vec3};

use super::Lowering;
use crate::model::{Clip, Geometry, Gradient, Object, Paint};

/// Layer counts tried, coarsest first: one for a hard edge, up to the count that steps the
/// alpha by half the noise (see [`crate::translucency::raster::NOISE`]) for a soft one.
const LAYERS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

pub(super) async fn lower(l: &mut Lowering<'_>, o: &Object) -> Option<Vec<Object>> {
    let blur = o.blur > 0. && !l.host.blur;
    let alpha =
        !l.host.transparent_stops && matches!(&o.paint, Paint::Gradient(g) if !g.is_opaque());
    let clip = o.clip.is_some() && !l.host.clips;
    if !(blur || alpha || clip) {
        return None;
    }
    l.report.layered += 1;
    let base = Object {
        paint: if alpha {
            opaque(&o.paint)
        } else {
            o.paint.clone()
        },
        blur: if blur { 0. } else { o.blur },
        clip: if clip { None } else { o.clip },
        ..o.clone()
    };
    if l.quick {
        return Some(vec![base]);
    }
    let field = Field::of(l, o, blur, alpha, clip);
    let peak = field.values.iter().fold(0f32, |a, &v| a.max(v));
    if peak <= 0. {
        return Some(Vec::new());
    }
    // Only a field with a soft part needs more than one level; the forms as `coarsest` takes
    // them, a checkpoint between levels
    let soft = blur || alpha;
    let counts = &LAYERS[..if soft { LAYERS.len() } else { 1 }];
    let Some(price) = l.price else {
        let finest = counts[counts.len() - 1];
        return Some(field.layers(&base, finest, peak, o.opacity).await);
    };
    let mut best: Option<(f64, Vec<Object>)> = None;
    for &k in counts {
        let form = field.layers(&base, k, peak, o.opacity).await;
        crate::progress::poll().await;
        let miss = l.miss(std::slice::from_ref(o), &form);
        if miss <= price {
            return Some(form);
        }
        if best.as_ref().is_none_or(|b| miss < b.0) {
            best = Some((miss, form));
        }
    }
    Some(best.map(|b| b.1).unwrap_or_default())
}

/// The object's alpha over the box it paints, `scale` samples per pixel along each axis.
pub(super) struct Field {
    /// The box's first sample, in samples from the image's corner.
    pub(super) x0: usize,
    pub(super) y0: usize,
    pub(super) w: usize,
    pub(super) h: usize,
    pub(super) scale: usize,
    values: Vec<f32>,
}

impl Field {
    /// The object's coverage, blurred where `blur`, times its clips' coverage where `clip` and
    /// its ramp's alpha where `alpha`. Samples are an eighth of the field's finest scale apart:
    /// the blur's deviation where it is blurred, else the anti-aliasing pixel.
    pub(super) fn of(l: &Lowering, o: &Object, blur: bool, alpha: bool, clip: bool) -> Self {
        let sigma = if blur { o.blur } else { 0. };
        let scale = (8. / sigma.max(1.)).ceil() as usize;
        let s = scale as f64;
        let sampled = Object {
            geometry: o.geometry.scaled(s),
            blur: sigma * s,
            clip: None,
            ..o.clone()
        };
        let r = sampled.painted();
        let (fw, fh) = (l.width * scale, l.height * scale);
        let (x0, y0) = (r.x0.min(fw), r.y0.min(fh));
        let (x1, y1) = ((r.x0 + r.w).min(fw), (r.y0 + r.h).min(fh));
        let (w, h) = (x1 - x0, y1 - y0);
        let mut values: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| r.values[(y0 + y - r.y0) * r.w + x0 + x - r.x0])
            .collect();
        let mut chain = if clip { o.clip } else { None };
        while let Some(c) = chain {
            let Clip {
                loops,
                rule,
                parent,
            } = &l.clips[c];
            let loops = Geometry::Fill {
                loops: loops.clone(),
                rule: *rule,
            }
            .scaled(s);
            let (loops, rule) = loops.outline();
            let cover = crate::scene::coverage_raster(&crate::scene::loops_path(&loops), rule);
            for y in 0..h {
                for x in 0..w {
                    values[y * w + x] *= cover.at((y0 + y) * fw + x0 + x, fw);
                }
            }
            chain = *parent;
        }
        if alpha {
            let fill = Object {
                opacity: 1.,
                ..o.clone()
            }
            .paint
            .fill(1.);
            for y in 0..h {
                for x in 0..w {
                    let p = glam::Vec2::new(x0 as f32 + x as f32 + 0.5, y0 as f32 + y as f32 + 0.5);
                    values[y * w + x] *= fill.paint(p / scale as f32).1;
                }
            }
        }
        Field {
            x0,
            y0,
            w,
            h,
            scale,
            values,
        }
    }

    /// The loops round the samples at or above `level`, in pixels.
    pub(super) fn loops(&self, level: f32) -> Vec<Vec<crate::model::Curve>> {
        let s = self.scale as f32;
        // A tenth of a pixel of fit, which keeps a sliver's sides apart, and the corner window of
        // the outline's own fit, in samples
        let fit = crate::tracer::fit::FitParams {
            tolerance: 0.1 * s,
            corner_window: crate::outline::FIT.corner_window * self.scale,
            ..crate::outline::FIT
        };
        let size = IVec2::new(self.w as i32, self.h as i32);
        let loops = crate::outline::outline_of(IVec2::ZERO, size, self.w, self.h, &fit, |i| {
            self.values[i] >= level
        });
        let origin = DVec2::new(self.x0 as f64, self.y0 as f64);
        loops
            .iter()
            .map(|l| {
                l.iter()
                    .map(|c| c.map(|p| (p.as_dvec2() + origin) / f64::from(s)))
                    .collect()
            })
            .collect()
    }

    /// `count` nested layers of `base` up to `peak`, the stack brought to `opacity` times the
    /// field: layer `k` the region at the middle of its step, at the opacity that takes the
    /// stack's alpha from step `k - 1` to step `k`.
    async fn layers(&self, base: &Object, count: usize, peak: f32, opacity: f64) -> Vec<Object> {
        let n = count as f64;
        let reached = |k: usize| opacity * f64::from(peak) * k as f64 / n;
        let mut layers = Vec::with_capacity(count);
        for k in 1..=count {
            crate::progress::poll().await;
            let loops = self.loops(peak * (k as f32 - 0.5) / count as f32);
            if loops.is_empty() {
                continue;
            }
            let (below, now) = (reached(k - 1), reached(k));
            let alpha = if below >= 1. {
                0.
            } else {
                1. - (1. - now) / (1. - below)
            };
            layers.push(Object {
                geometry: Geometry::Fill {
                    loops,
                    rule: FillRule::EvenOdd,
                },
                opacity: alpha,
                ..base.clone()
            });
        }
        layers
    }
}

/// The paint with every stop opaque: a transparent stop takes its nearest neighbour's colour,
/// so the straight colour it is mixed in keeps on through it.
fn opaque(paint: &Paint) -> Paint {
    let Paint::Gradient(g) = paint else {
        return paint.clone();
    };
    let (from, ramp) = g.ramp();
    let mut knots: Vec<(f32, [f32; 4])> = std::iter::once((0., [from.x, from.y, from.z, 1.]))
        .chain(ramp.stops().iter())
        .chain(std::iter::once((1., ramp.to())))
        .collect();
    let colours: Vec<Option<Vec3>> = knots
        .iter()
        .map(|(_, c)| (c[3] > 0.).then(|| Vec3::new(c[0], c[1], c[2])))
        .collect();
    for (k, knot) in knots.iter_mut().enumerate() {
        let near = (0..colours.len())
            .filter_map(|j| colours[j].map(|c| (k.abs_diff(j), c)))
            .min_by_key(|(d, _)| *d)
            .map_or(Vec3::ZERO, |(_, c)| c);
        let c = colours[k].unwrap_or(near);
        knot.1 = [c.x, c.y, c.z, 1.];
    }
    let n = knots.len();
    let stops = Stops::new(&knots[1..n - 1]);
    let to = knots[n - 1].1;
    let mut ramp = *ramp;
    match &mut ramp {
        Ramp::Linear {
            to: t, stops: s, ..
        }
        | Ramp::Radial {
            to: t, stops: s, ..
        }
        | Ramp::Path {
            to: t, stops: s, ..
        } => (*t, *s) = (to, stops),
    }
    Paint::Gradient(Gradient::new(from, ramp))
}

/// The loops round where `values` (one per pixel of a `width` by `height` frame) reach
/// `level`, read between the pixels' centres at four samples per pixel (a picture's alpha
/// for the drawing's clip).
pub(crate) fn loops_at(
    values: &[f32],
    width: usize,
    height: usize,
    level: f32,
) -> Vec<Vec<crate::model::Curve>> {
    const SCALE: usize = 4;
    let (w, h) = (width * SCALE, height * SCALE);
    let at = |x: usize, y: usize| values[y.min(height - 1) * width + x.min(width - 1)];
    let mut sampled = Vec::with_capacity(w * h);
    for y in 0..h {
        // the sample's place in pixel centres
        let fy = ((y as f32 + 0.5) / SCALE as f32 - 0.5).max(0.);
        let (y0, ty) = (fy.floor() as usize, fy.fract());
        for x in 0..w {
            let fx = ((x as f32 + 0.5) / SCALE as f32 - 0.5).max(0.);
            let (x0, tx) = (fx.floor() as usize, fx.fract());
            let top = at(x0, y0) * (1. - tx) + at(x0 + 1, y0) * tx;
            let bottom = at(x0, y0 + 1) * (1. - tx) + at(x0 + 1, y0 + 1) * tx;
            sampled.push(top * (1. - ty) + bottom * ty);
        }
    }
    Field {
        x0: 0,
        y0: 0,
        w,
        h,
        scale: SCALE,
        values: sampled,
    }
    .loops(level)
}
