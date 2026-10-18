//! The objective in its parts, the decision in a window, and the layer put on top.

use super::*;

/// The plan's objective over a scene in its parts: the squared color error, the segments
/// (as their costed count), the paths, and what segments and paths cost together.
#[derive(Clone, Copy, Debug, Default)]
pub struct Parts {
    pub error: f64,
    pub segments: f64,
    pub paths: usize,
    pub cost: f64,
}

impl Parts {
    pub fn total(&self) -> f64 {
        self.error + self.cost
    }
}

/// The plan's objective over a whole scene in its parts: its error against `ctx`'s image and
/// what its segments and paths cost.
pub fn parts(ctx: &Context, scene: &Scene) -> Parts {
    let full = Window::full(ctx.width, ctx.height);
    let error = ctx.error(full, &scene.render(full, 0..scene.shapes.len(), [1.; 3]));
    let cost: f64 = scene
        .shapes
        .iter()
        .map(|s| s.cost_segments() * ctx.segment_cost + ctx.path_cost)
        .sum();
    Parts {
        error: error as f64,
        segments: scene.shapes.iter().map(|s| s.cost_segments()).sum(),
        paths: scene.shapes.len(),
        cost,
    }
}

/// What a shading layer costs less than a colored translucent shape: its color is black or
/// white, so it pays one number (its opacity) against four: a segment's worth, at a segment
/// per two numbers.
pub(super) fn shading_discount(ctx: &Context, candidate: &crate::translucency::Candidate) -> f64 {
    if candidate.shade.is_some() {
        ctx.segment_cost
    } else {
        0.
    }
}

/// Pixels of context around a candidate's box in its decision window: the overlay fit's reach
/// (10 px) plus the leaves' sampling past the contact (5 px).
pub(super) const WINDOW_CONTEXT: f32 = 16.;

/// Whether the shape pays in its window, traced as it is and peeled with the shape on top.
/// Where the window exceeds a quarter of the image (its two traces would cost more than half
/// the whole one) None at the top level, and at a nested level the densest quarter's verdict.
pub(super) fn decide_in_window(
    ctx: &Context,
    candidate: &crate::translucency::Candidate,
    peeled: &[u8],
    detail: f64,
    options: Options,
    above: Option<&[bool]>,
) -> Option<WindowVerdict> {
    let (width, height) = (ctx.width, ctx.height);
    let mut window = Window::around(
        candidate.min.as_vec2(),
        candidate.max.as_vec2(),
        WINDOW_CONTEXT,
        width,
        height,
    );
    if 4 * window.width * window.height > width * height {
        // A nested level decides even a large candidate in a window, since its whole traces
        // only recurse (kemonomimi 660: nine levels of frame-wide near-white layers)
        if super::layers::LEVEL.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return None;
        }
        window = densest_quarter(&candidate.mask, width, height);
    }
    let crop = |rgb: &[u8]| -> Vec<u8> {
        window
            .pixels(width)
            .flat_map(|i| [rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]])
            .collect()
    };
    let original: Vec<u8> = ctx
        .target
        .iter()
        .flat_map(|c| (*c * 255.).round().to_array().map(|v| v as u8))
        .collect();
    // The same picture as the whole image, so the same costs and tolerances
    let trace = |rgb: Vec<u8>, edge: Option<&[bool]>, peel: bool| -> (Context, Scene) {
        let mut timings = Timings::default();
        let (mut c, h) = crate::hierarchy::analyze(&rgb, window.width, window.height, detail, &mut timings);
        c.segment_cost = ctx.segment_cost;
        c.path_cost = ctx.path_cost;
        c.frame = ctx.frame;
        let peel = peel.then_some((candidate.mask.as_slice(), candidate.k));
        crate::uncertainty::inherit(&mut c, ctx, window, peel, edge);
        let scene = trace_level(
            c.clone(),
            h,
            detail,
            options,
            edge,
            false,
            &mut timings,
            &mut |_| {},
        );
        (c, scene)
    };
    let cropped = candidate.crop(window, width);
    let above: Option<Vec<bool>> = above.map(|e| window.pixels(width).map(|i| e[i]).collect());
    let edge = peeled_edge(&cropped.mask, window.width, window.height, above.as_deref());
    let (local, plain) = trace(crop(&original), None, false);
    let (_, mut with) = trace(crop(peeled), Some(&edge), true);
    let overlay = cropped.overlay(window.width, window.height)?;
    append_overlay(
        &local,
        &mut with,
        overlay,
        &cropped,
        None,
        options.stages.primitives,
    );
    let verdict = WindowVerdict {
        with: parts(&local, &with),
        plain: parts(&local, &plain),
        discount: shading_discount(&local, candidate),
        terms: layer_terms(&local, &with, &plain, &cropped),
        segment_cost: local.segment_cost,
    };
    Some(verdict)
}

/// The window of half the image's width and height holding most of `mask`'s pixels, the
/// first in raster order among equals.
fn densest_quarter(mask: &[bool], width: usize, height: usize) -> Window {
    let (w, h) = (width.div_ceil(2), height.div_ceil(2));
    let mut sum = vec![0u32; (width + 1) * (height + 1)];
    for y in 0..height {
        let mut row = 0u32;
        for x in 0..width {
            row += u32::from(mask[y * width + x]);
            sum[(y + 1) * (width + 1) + x + 1] = sum[y * (width + 1) + x + 1] + row;
        }
    }
    let at = |x: usize, y: usize| sum[y * (width + 1) + x];
    let mut best = (0u32, 0usize, 0usize);
    for y in 0..=height - h {
        for x in 0..=width - w {
            let n = at(x + w, y + h) + at(x, y) - at(x + w, y) - at(x, y + h);
            if n > best.0 {
                best = (n, x, y);
            }
        }
    }
    Window {
        x: best.1,
        y: best.2,
        width: w,
        height: h,
    }
}

/// A candidate's window traced with and without the shape: the objective's parts of each,
/// and the shading discount the shape gets.
#[derive(Clone, Copy, Debug)]
pub struct WindowVerdict {
    pub with: Parts,
    pub plain: Parts,
    pub discount: f64,
    /// The layer's terms in the window, see [`crate::translucency::cost`].
    pub terms: LayerTerms,
    /// The window's segment cost, for [`Self::margin`].
    pub segment_cost: f64,
}

impl WindowVerdict {
    /// Whether the window keeps the shape: the scene with it explains the window for less.
    pub fn keeps(&self) -> bool {
        self.margin() < 0.
    }

    /// The objective with the shape minus without it in the window.
    pub fn margin(&self) -> f64 {
        margin(
            &self.with,
            &self.plain,
            self.discount,
            &self.terms,
            self.segment_cost,
        )
    }
}

use crate::translucency::cost::LayerTerms;

/// The objective of a scene with a layer minus the one without, with the segment cost of the
/// context the layer's `terms` were measured in: the error under the layer weighed by `1 / k^2`
/// where it is seen through, and its outline credited where it runs along the beneath's.
pub(super) fn margin(
    with: &Parts,
    without: &Parts,
    discount: f64,
    terms: &LayerTerms,
    segment_cost: f64,
) -> f64 {
    // Gated: the terms count only where two materials' images meet in the region
    if terms.meeting < 2 {
        return (with.total() - discount) - without.total();
    }
    // an error of the beneath reaches the raster `k` times smaller
    let k = (terms.k as f64).max(1e-3);
    let w = 1. / (k * k) - 1.;
    let credit = terms.shared * segment_cost;
    let (under_with, under_without) = (terms.seen_with, terms.seen_without);
    (with.total() - discount + w * under_with - credit) - (without.total() + w * under_without)
}

/// The layer's terms of a comparison.
pub(super) fn layer_terms(
    ctx: &Context,
    with: &Scene,
    without: &Scene,
    candidate: &crate::translucency::Candidate,
) -> LayerTerms {
    LayerTerms::measure(ctx, with, without, candidate)
}

/// Puts a translucent shape peeled off `ctx`'s image on top of `scene` (the trace beneath it):
/// completed (with `primitives` proposed where the stage runs), simplified and fitted against
/// `ctx`'s image, its fill refitted to what shows.
pub(super) fn append_overlay(
    ctx: &Context,
    scene: &mut Scene,
    overlay: crate::translucency::Overlay,
    candidate: &crate::translucency::Candidate,
    free: Option<&[bool]>,
    primitives: bool,
) {
    let mask = &candidate.mask;
    let shade = candidate.shade;
    let width = ctx.width;
    let tolerances = tolerances(ctx);
    let reach = MARGIN + tolerances[tolerances.len() - 1];
    // `free`: the pixels inside the layer's region that the parts above cover (drawn over the
    // shape afterwards), which the shape is fitted without; off the region they stay, so the
    // outline does not run out over them
    let free: Option<Vec<bool>> = free.map(|f| f.iter().zip(mask).map(|(&h, &m)| h && m).collect());
    let free = free.as_deref();
    let fit = Context {
        weights: free.map(|f| Weights {
            window: Window::full(ctx.width, ctx.height),
            values: f.iter().map(|&h| if h { 0. } else { 1. }).collect(),
            masked_color: None,
        }),
        ..ctx.clone()
    };
    let mut shape = Shape::new(overlay.loops, overlay.fill);
    shape.smooth = true;
    let position = scene.shapes.len();
    // Fitted near its traced outline first, as the plan's shapes are: a mask's trace is only a
    // pixel's guess at the edge
    if let Some(near) = Place::new(&fit, scene, &shape, position, MARGIN) {
        near.refine(&fit, &mut shape);
    }
    if let Some(place) = Place::new(&fit, scene, &shape, position, reach) {
        // The shape's coverage after each step as `overlay-<step>.pgm` in the mask directory
        // (the window's size, its origin in `overlay-window.txt`)
        let dump = |shape: &Shape, step: &str| {
            #[cfg(not(feature = "diagnostics"))]
            let _ = (shape, step);
            #[cfg(feature = "diagnostics")]
            if let Some(dir) = crate::diagnostics::mask_dir() {
                let w = place.window;
                let cov = shape.coverage(w);
                let mut pgm = format!("P5\n{} {}\n255\n", w.width, w.height).into_bytes();
                pgm.extend(cov.iter().map(|c| (c * 255.).round().clamp(0., 255.) as u8));
                let _ = std::fs::write(
                    dir.join(format!("overlay-{}x{}-{step}.pgm", w.width, w.height)),
                    pgm,
                );
                let _ = std::fs::write(
                    dir.join(format!("overlay-{}x{}-window.txt", w.width, w.height)),
                    format!("{} {} {} {}\n", w.x, w.y, w.width, w.height),
                );
            }
        };
        dump(&shape, "0-refined");
        let fit_cost = place.cost(&fit, shape.fill);
        let what = completion(Some(1), 0., primitives);
        complete(&fit, &place, &fit_cost, &mut shape, what);
        dump(&shape, "1-completed");
        // The outline is simplified against the region (black on white), not the composite,
        // where a layer over a beneath near its colour is weak and a lobe never pays its
        // segments (the blouse's lobe over niabot's skin)
        {
            let mask_ctx = Context {
                target: std::sync::Arc::new(
                    mask.iter()
                        .map(|m| if *m { Vec3::ZERO } else { Vec3::ONE })
                        .collect(),
                ),
                unmixed: None,
                weights: None,
                tolerance: None,
                accept: None,
                ..ctx.clone()
            };
            let mask_scene = Scene::with_background(ctx.width, ctx.height, Fill::solid(Vec3::ONE));
            let fill = shape.fill;
            shape.fill = Fill::solid(Vec3::ZERO);
            if let Some(mask_place) = Place::new(
                &mask_ctx,
                &mask_scene,
                &shape,
                mask_scene.shapes.len(),
                reach,
            ) {
                let mask_cost = mask_place.cost(&mask_ctx, shape.fill);
                simplify_with(
                    &mask_ctx,
                    &mask_place,
                    &mask_cost,
                    &mut shape,
                    &tolerances,
                    crate::outline::CORNER_ANGLE,
                );
            }
            shape.fill = fill;
        }
        dump(&shape, "2-simplified");
        complete(&fit, &place, &fit_cost, &mut shape, what);
        dump(&shape, "3-completed");
        let window = place.window;
        let target: Vec<Vec3> = window.pixels(width).map(|i| ctx.target[i]).collect();
        let mut coverage = shape.coverage(window);
        if let Some(f) = free {
            for (c, i) in coverage.iter_mut().zip(window.pixels(width)) {
                if f[i] {
                    *c = 0.;
                }
            }
        }
        let fill =
            crate::translucency::refit_fill(&target, &place.below, &coverage, shape.fill, shade);
        shape.fill = fill;
    }
    let change = scene.change();
    shape.changed = change;
    shape.fitted = change;
    // The layers are numbered bottom to top: a layer kept on a level beneath this one is
    // already in the scene
    shape.layer = Some(
        scene
            .shapes
            .iter()
            .filter_map(|s| s.layer)
            .max()
            .map_or(0, |n| n + 1),
    );
    // Pushed with an id of its own, as every shape the plan adds
    scene.push(shape);
}

#[cfg(test)]
mod tests {
    /// The densest quarter holds a block that fits it whole, wherever the block lies.
    #[test]
    fn the_densest_quarter_holds_the_block() {
        let (w, h) = (40, 30);
        for (bx, by) in [(0, 0), (22, 3), (25, 14), (7, 16)] {
            let mask: Vec<bool> = (0..w * h)
                .map(|i| (bx..bx + 12).contains(&(i % w)) && (by..by + 10).contains(&(i / w)))
                .collect();
            let q = super::densest_quarter(&mask, w, h);
            assert_eq!((q.width, q.height), (20, 15));
            assert!(q.x <= bx && bx + 12 <= q.x + q.width, "{bx} {by} {q:?}");
            assert!(q.y <= by && by + 10 <= q.y + q.height, "{bx} {by} {q:?}");
        }
    }
}
