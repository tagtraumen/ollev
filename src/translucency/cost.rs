//! What the plan's objective says of a translucent layer beyond error, segments and paths, when
//! it compares the scene with the layer (beneath traced, layer on top) against the opaque
//! trace: [`shared_outline`] and the error under the layer weighed by `1 / k^2`.

use super::SHOW_THROUGH;
use crate::{Context, Scene, Window};

/// The terms of a comparison of a scene with a layer (its last shape) against the scene
/// without it: the squared error under the layer's mask in each, the
/// shared outline and the materials meeting.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerTerms {
    pub k: f32,
    pub under_with: f64,
    pub under_without: f64,
    /// The same over the pixels the layer is seen through, see [`seen_through`].
    pub seen_with: f64,
    pub seen_without: f64,
    /// The layer's costed segments that run along the beneath's boundaries, see
    /// [`shared_outline`].
    pub shared: f64,
    /// The materials whose images meet in the layer's region under its relation, see
    /// [`meeting`].
    pub meeting: usize,
}

impl LayerTerms {
    /// Measures the terms: `with` holds the layer as its last shape over the beneath's shapes,
    /// `mask` the layer's pixels in `ctx`'s frame.
    pub fn measure(
        ctx: &Context,
        with: &Scene,
        without: &Scene,
        candidate: &super::Candidate,
    ) -> Self {
        let mask = &candidate.mask[..];
        let seen = seen_through(ctx, candidate);
        Self {
            k: candidate.k,
            meeting: meeting(ctx, candidate),
            under_with: error_under(ctx, with, mask),
            under_without: error_under(ctx, without, mask),
            seen_with: error_under(ctx, with, &seen),
            seen_without: error_under(ctx, without, &seen),
            shared: shared_outline(with, ctx),
        }
    }
}

/// The materials whose images meet inside a candidate's region under its relation: the inner
/// fit at the candidate's `k` and `m` ([`super::inner::fit_fixed`]), then [`super::rank::contact`]
/// within [`super::rank::reach`]. The ranking's cue for a real layer, here a gate on the terms.
pub fn meeting(ctx: &Context, candidate: &super::Candidate) -> usize {
    let img = super::raster::Image::from_target(&ctx.target, ctx.width);
    let rim = super::materials::rim_of(&img);
    let inp = super::inner::InnerInputs::new(&img, &candidate.mask, &rim, None);
    let fit = super::inner::fit_fixed(&inp, candidate.k as f64, candidate.m.as_dvec3());
    super::rank::contact(&inp, &fit, super::rank::reach(ctx.width)).meeting
}

/// The pixels of a layer's region whose colour the layer `o = k b + m` can produce from some
/// `b` in the unit cube, to the show-through ([`super::beneath::outside_image`]): where the
/// `1 / k` weights hold, as elsewhere the pixel is not under the layer and its error is raw.
pub fn seen_through(ctx: &Context, candidate: &super::Candidate) -> Vec<bool> {
    let (k, m) = (candidate.k as f64, candidate.m.as_dvec3());
    candidate
        .mask
        .iter()
        .zip(ctx.target.iter())
        .map(|(&inside, c)| {
            inside && super::beneath::outside_image(c.as_dvec3(), k, m) <= SHOW_THROUGH as f64
        })
        .collect()
}

/// The squared error of `scene` against `ctx`'s image over the pixels of `mask`, weighted as
/// [`Context::error`] weighs them.
pub fn error_under(ctx: &Context, scene: &Scene, mask: &[bool]) -> f64 {
    let full = Window::full(ctx.width, ctx.height);
    let render = scene.render(full, 0..scene.shapes.len(), [1.; 3]);
    render
        .iter()
        .enumerate()
        .filter(|(i, _)| mask[*i])
        .map(|(i, r)| ctx.pixel_error(i, *r) as f64)
        .sum()
}

/// The costed segments of `scene`'s last shape (the layer) that run along the boundaries of
/// the shapes beneath it (samples at a quarter, half and three quarters within a pixel of a
/// change of topmost shape, or the frame): an outline the beneath draws anyway.
pub fn shared_outline(scene: &Scene, ctx: &Context) -> f64 {
    let Some(layer) = scene.shapes.last() else {
        return 0.;
    };
    let (width, height) = (ctx.width, ctx.height);
    let top = top_shapes(scene, scene.shapes.len() - 1, width, height);
    let mut boundary = vec![false; width * height];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let edge = x == 0
                || y == 0
                || x + 1 == width
                || y + 1 == height
                || top[i - 1] != top[i]
                || top[i + 1] != top[i]
                || top[i - width] != top[i]
                || top[i + width] != top[i];
            if edge {
                for yy in y.saturating_sub(1)..(y + 2).min(height) {
                    for xx in x.saturating_sub(1)..(x + 2).min(width) {
                        boundary[yy * width + xx] = true;
                    }
                }
            }
        }
    }
    let on = |p: glam::Vec2| {
        let (x, y) = (p.x.floor() as i64, p.y.floor() as i64);
        x < 0
            || y < 0
            || x as usize >= width
            || y as usize >= height
            || boundary[y as usize * width + x as usize]
    };
    let count = layer
        .loops
        .iter()
        .flatten()
        .filter(|cubic| {
            [0.25f32, 0.5, 0.75].iter().all(|&t| {
                let u = 1. - t;
                on(cubic[0] * (u * u * u)
                    + cubic[1] * (3. * u * u * t)
                    + cubic[2] * (3. * u * t * t)
                    + cubic[3] * (t * t * t))
            })
        })
        .count();
    count as f64 * layer.cost_segments() / layer.segments().max(1) as f64
}

/// Per pixel the topmost of `scene`'s first `count` shapes covering it by half or more.
pub fn top_shapes(scene: &Scene, count: usize, width: usize, height: usize) -> Vec<Option<u32>> {
    let mut top = vec![None; width * height];
    for (s, shape) in scene.shapes.iter().enumerate().take(count) {
        let window = Window::around(shape.min, shape.max, 1., width, height);
        for (i, a) in window.pixels(width).zip(shape.coverage(window)) {
            if a >= 0.5 {
                top[i] = Some(s as u32);
            }
        }
    }
    top
}
