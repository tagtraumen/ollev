//! Fitting one shape's outline to the image, with everything below it held fixed.

use crate::render::coverage_cost::CoverageCost;
use crate::render::{Adam, ParamKind};
use crate::tracer::fit::untangle;
use glam::{Vec2, Vec3};

use crate::scene::{Fill, Layer};
use crate::{Context, Scene, Shape, Window};

/// Pixels around a shape's bounding box that its fit looks at, for edges that move outwards.
pub const MARGIN: f32 = 2.;
/// Optimizer steps at most.
pub(crate) const MAX_STEPS: usize = 60;
/// Step size of the control points in pixels, roughly.
pub(crate) const LEARNING_RATE: f32 = 0.1;
/// Steps without the error improving by `PLATEAU_GAIN` (relative) before the step size halves.
pub(crate) const PLATEAU_STEPS: usize = 5;
pub(crate) const PLATEAU_GAIN: f32 = 1e-3;
/// Pixels around a hidden pixel within which an outline counts as free, see [`Place::free`].
const FREE_REACH: usize = 1;
/// Step size, relative to the initial one, below which the optimization stops.
pub(crate) const MIN_STEP_SCALE: f32 = 1. / 16.;
/// The pull on a line's handles toward a third of its chord, resolving the fit's null space
/// along a straight edge; tiny against any real image gradient, so it only acts where nothing
/// else pulls.
pub const HANDLE_PULL: f32 = 1e-3;
/// The least σ a blurred shape's fit may reach, in pixels: below it the kernel is one tap.
pub(crate) const LEAST_SIGMA: f32 = 0.1;
/// How far a cubic's handles may lie from its chord for the pull to treat it as a line.
const LINE_TOLERANCE: f32 = 0.5;

/// Moves the outline of `shape` (not its color) to match the image within `window`, drawn over
/// `below`, through `clip` (how much of each pixel shows) and under `above`. Returns the error
/// within the window afterwards.
pub fn refine(
    ctx: &Context,
    shape: &mut Shape,
    window: Window,
    below: &[Vec3],
    clip: Option<&[f32]>,
    above: Option<&Layer>,
) -> f32 {
    let cost = coverage_cost(ctx, window, shape.fill, below, clip, above);
    fit(&cost, shape, window.origin(), ctx.pull)
}

/// Moves the outline of `shape` to lower `cost`, the error of drawing it in its window, whose
/// corner is at `origin`, with `pull` on line handles, see [`Context::pull`]. Returns the error
/// afterwards.
pub fn fit(cost: &CoverageCost, shape: &mut Shape, origin: Vec2, pull: f32) -> f32 {
    let mut params: Vec<f32> = shape
        .loops
        .iter()
        .flatten()
        .flat_map(|c| [c[0], c[1], c[2]])
        .flat_map(|p| (p - origin).to_array())
        .collect();
    // A blurred shape's σ is fitted with its outline, as the last parameter
    let blurred = shape.blur > 0.;
    if blurred {
        params.push(shape.blur);
    }
    let kinds = vec![ParamKind::Point; params.len()];
    let lengths: Vec<usize> = shape.loops.iter().map(Vec::len).collect();
    let mut adam = Adam::new(LEARNING_RATE, 0.);

    let path = |params: &[f32]| {
        let mut path = shape.path(origin);
        let mut k = 0;
        for subpath in &mut path.subpaths {
            for p in &mut subpath.points {
                *p = Vec2::new(params[k], params[k + 1]);
                k += 2;
            }
        }
        if blurred {
            path.blur = params[k];
        }
        path
    };

    let mut best = (f32::INFINITY, params.clone());
    let mut since_best = 0;
    for _ in 0..MAX_STEPS {
        let (value, grad, d_sigma) = if blurred {
            cost.blurred(&path(&params), true)
        } else {
            let (value, grad) = cost.evaluate(&path(&params));
            (value, grad, 0.)
        };
        if value < best.0 * (1. - PLATEAU_GAIN) {
            since_best = 0;
        } else {
            since_best += 1;
        }
        if value < best.0 {
            best = (value, params.clone());
        }
        if since_best >= PLATEAU_STEPS {
            adam.lr_scale *= 0.5;
            since_best = 0;
            if adam.lr_scale < MIN_STEP_SCALE {
                break;
            }
        }
        let mut flat: Vec<f32> = grad.iter().flatten().flat_map(Vec2::to_array).collect();
        if blurred {
            flat.push(d_sigma);
        }
        if pull > 0. {
            pull_handles(&params, &lengths, pull, &mut flat);
        }
        adam.step_flat(&mut params, &kinds, &flat);
        untangle_all(&mut params, &lengths);
        if blurred {
            let sigma = params.last_mut().expect("the blur's σ");
            *sigma = sigma.max(LEAST_SIGMA);
        }
    }

    let mut k = 0;
    let point = |k: usize| Vec2::new(best.1[2 * k], best.1[2 * k + 1]) + origin;
    for cubics in &mut shape.loops {
        let start = k;
        let n = cubics.len();
        for (i, c) in cubics.iter_mut().enumerate() {
            let next = start + (3 * i + 3) % (3 * n);
            *c = [point(k), point(k + 1), point(k + 2), point(next)];
            k += 3;
        }
    }
    if blurred {
        shape.blur = best.1[best.1.len() - 1];
    }
    shape.update_bounds();
    shape.snap_corners();
    best.0
}

/// The error in `window` of a shape with `fill` as a quadratic in its coverage of each pixel:
/// drawn over `below`, through `clip` and under `above`, a pixel shows `u + coverage * v`
/// against its target, with `u` what shows without the shape and `v` what the shape changes.
fn coverage_cost(
    ctx: &Context,
    window: Window,
    fill: Fill,
    below: &[Vec3],
    clip: Option<&[f32]>,
    above: Option<&Layer>,
) -> CoverageCost {
    let n = window.width * window.height;
    let (mut a, mut b, mut c) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (k, i) in window.pixels(ctx.width).enumerate() {
        let (weight, target) = ctx.weighted_target(i);
        let (color, alpha) = fill.paint_pixel(i % ctx.width, i / ctx.width);
        let mut u = below[k] - target;
        // A translucent fill moves the pixel only part of the way to its color
        let mut v = (color - below[k]) * alpha * clip.map_or(1., |m| m[k]);
        if let Some(layer) = above {
            u = layer.color[k] + layer.transmittance[k] * below[k] - target;
            v *= layer.transmittance[k];
        }
        a.push(weight * v.length_squared());
        b.push(2. * weight * u.dot(v));
        c.push(weight * u.length_squared());
    }
    CoverageCost::new(window.width, window.height, a, b, &c)
}

/// The error of a shape's render `drawn` over `below` once shown through `clip` and under
/// `above`, as the context measures it.
pub(crate) fn composite_error(
    ctx: &Context,
    window: Window,
    drawn: &[Vec3],
    below: &[Vec3],
    clip: Option<&[f32]>,
    above: Option<&Layer>,
) -> f32 {
    let mut shown = drawn.to_vec();
    if let Some(clip) = clip {
        for ((s, b), c) in shown.iter_mut().zip(below).zip(clip) {
            *s = *b + *c * (*s - *b);
        }
    }
    if let Some(layer) = above {
        for ((s, c), t) in shown.iter_mut().zip(&layer.color).zip(&layer.transmittance) {
            *s = *c + *t * *s;
        }
    }
    ctx.error(window, &shown)
}

/// Adds to `grad`, on the handles only, the gradient of `pull` times the squared difference
/// between each handle's length and a third of its chord, for cubics within [`LINE_TOLERANCE`]
/// of a line. Curves are left alone, since a long arc needs its long handles.
fn pull_handles(params: &[f32], lengths: &[usize], pull: f32, grad: &mut [f32]) {
    let point = |k: usize| Vec2::new(params[2 * k], params[2 * k + 1]);
    let mut start = 0;
    for &n in lengths {
        for i in 0..n {
            let k = start + 3 * i;
            let next = start + (3 * i + 3) % (3 * n);
            let chord = point(next) - point(k);
            let length = chord.length();
            if length < 1e-6 {
                continue;
            }
            let normal = chord.perp() / length;
            let is_line = [k + 1, k + 2]
                .iter()
                .all(|&h| (point(h) - point(k)).dot(normal).abs() <= LINE_TOLERANCE);
            if !is_line {
                continue;
            }
            let wanted = length / 3.;
            for (handle, anchor) in [(k + 1, k), (k + 2, next)] {
                let arm = point(handle) - point(anchor);
                let length = arm.length();
                if length < 1e-6 {
                    continue;
                }
                let g = 2. * pull * (length - wanted) * arm / length;
                grad[2 * handle] += g.x;
                grad[2 * handle + 1] += g.y;
            }
        }
        start += 3 * n;
    }
}

/// Keeps every cubic free of loops, see [`untangle`].
fn untangle_all(params: &mut [f32], lengths: &[usize]) {
    let point = |params: &[f32], k: usize| Vec2::new(params[2 * k], params[2 * k + 1]);
    let mut start = 0;
    for &n in lengths {
        for i in 0..n {
            let k = start + 3 * i;
            let next = start + (3 * i + 3) % (3 * n);
            let cubic = untangle([k, k + 1, k + 2, next].map(|k| point(params, k)));
            for (j, p) in [(k + 1, cubic[1]), (k + 2, cubic[2])] {
                params[2 * j] = p.x;
                params[2 * j + 1] = p.y;
            }
        }
        start += 3 * n;
    }
}

/// Tolerances in pixels at which a fitted outline is simplified, in turn while it pays.
pub const SIMPLIFY: [f32; 3] = [2., 4., 8.];

/// Where a shape sits in a scene while it is fitted: a window around it, the render of the shapes
/// below it, those above it, and its clip.
pub struct Place {
    pub window: Window,
    pub below: Vec<Vec3>,
    pub above: Option<Layer>,
    pub clip: Option<Vec<f32>>,
    /// The error in the window without the shape.
    pub before: f32,
}

impl Place {
    /// `shape` at `position` in the stack of `scene`, so the shapes from there up lie above it,
    /// over a window `margin` pixels around it, or `None` for an empty window.
    pub fn new(
        ctx: &Context,
        scene: &Scene,
        shape: &Shape,
        position: usize,
        margin: f32,
    ) -> Option<Self> {
        Self::over(ctx, scene, shape, position, margin, |window| {
            scene.render(window, 0..position, [1.; 3])
        })
    }

    /// [`Place::new`] with the render of the shapes below the place over a window given by
    /// `below`, as a pass walking up the stack keeps it.
    pub fn over(
        ctx: &Context,
        scene: &Scene,
        shape: &Shape,
        position: usize,
        margin: f32,
        below: impl FnOnce(Window) -> Vec<Vec3>,
    ) -> Option<Self> {
        let window = Window::around(
            shape.paint_min(),
            shape.paint_max(),
            margin,
            ctx.width,
            ctx.height,
        );
        if window.is_empty() {
            return None;
        }
        let below = below(window);
        let above = scene.layer(window, position);
        let clip = scene.clip_of(window, shape.clip);
        let before = composite_error(ctx, window, &below, &below, None, above.as_ref());
        Some(Self {
            window,
            below,
            above,
            clip,
            before,
        })
    }

    /// This place with the shape clipped to the shape with id `clip` instead.
    #[must_use]
    pub fn clipped(&self, scene: &Scene, clip: Option<usize>) -> Self {
        Self {
            window: self.window,
            below: self.below.clone(),
            above: self.above.clone(),
            clip: scene.clip_of(self.window, clip),
            before: self.before,
        }
    }

    /// The error of drawing a shape with `fill` here, for evaluating and fitting outlines.
    pub fn cost(&self, ctx: &Context, fill: Fill) -> CoverageCost {
        coverage_cost(
            ctx,
            self.window,
            fill,
            &self.below,
            self.clip.as_deref(),
            self.above.as_ref(),
        )
    }

    /// The error in the window with `shape` drawn, as the context measures it.
    pub fn error(&self, ctx: &Context, shape: &Shape) -> f32 {
        self.error_of(ctx, std::slice::from_ref(shape))
    }

    /// The error in the window with `shapes` drawn here in their order, as the context
    /// measures it: the lines of a network of strokes, each in its own colour.
    pub fn error_of(&self, ctx: &Context, shapes: &[Shape]) -> f32 {
        let scene = crate::render::Scene {
            width: self.window.width as u32,
            height: self.window.height as u32,
            paths: shapes
                .iter()
                .map(|s| s.path(self.window.origin()))
                .collect(),
        };
        let drawn = crate::render::render_over(&scene, self.below.clone());
        composite_error(
            ctx,
            self.window,
            &drawn,
            &self.below,
            self.clip.as_deref(),
            self.above.as_ref(),
        )
    }

    /// The error in the window with `shape` drawn: from `cost` where that measures what the
    /// context does (RGB, no [`Context::tolerance`]), else rendered and measured by the context.
    pub fn measure(&self, ctx: &Context, cost: &CoverageCost, shape: &Shape) -> f32 {
        if ctx.tolerance.is_none() {
            cost.value(&shape.path(self.window.origin()))
        } else {
            self.error(ctx, shape)
        }
    }

    /// Fits `shape` here, see [`refine`], returning its error.
    pub fn refine(&self, ctx: &Context, shape: &mut Shape) -> f32 {
        refine(
            ctx,
            shape,
            self.window,
            &self.below,
            self.clip.as_deref(),
            self.above.as_ref(),
        )
    }

    /// Per pixel of the window, whether a shape's outline is free to go anywhere there: pixels
    /// left out of the error, hidden under the shapes above, or outside the clip, and their four
    /// neighbors, since an outline along such pixels runs on their edge.
    pub fn free(&self, ctx: &Context) -> Vec<bool> {
        self.grown(ctx, |_| false)
    }

    /// [`Place::free`] at a point: whether the window's pixel holding it is free.
    pub fn free_at(&self, ctx: &Context) -> impl Fn(Vec2) -> bool + use<> {
        let (free, window) = (self.free(ctx), self.window);
        move |p| window.cell(p).is_some_and(|k| free[k])
    }

    /// [`Place::free`] with the pixels the weights count at less than full too: the regions of
    /// shapes drawn later, which cover them. What an outline there shows is no evidence.
    pub fn hidden(&self, ctx: &Context) -> Vec<bool> {
        let weights = ctx.weights.as_ref();
        self.grown(ctx, |i| {
            weights.is_some_and(|w| {
                w.window
                    .index(i, ctx.width)
                    .is_some_and(|k| w.values[k] < 1.)
            })
        })
    }

    fn grown(&self, ctx: &Context, also: impl Fn(usize) -> bool) -> Vec<bool> {
        let (w, h) = (self.window.width, self.window.height);
        let mut free: Vec<bool> = self
            .window
            .pixels(ctx.width)
            .enumerate()
            .map(|(k, i)| {
                ctx.is_free(i)
                    || also(i)
                    || self
                        .above
                        .as_ref()
                        .is_some_and(|a| a.transmittance[k].max_element() < 0.5)
                    || self.clip.as_ref().is_some_and(|c| c[k] < 0.5)
            })
            .collect();
        for _ in 0..FREE_REACH {
            free = (0..w * h)
                .map(|k| free[k] || crate::context::neighbours(k, w, h).any(|j| free[j]))
                .collect();
        }
        free
    }
}

/// Simplifies `shape` at each of `tolerances` in turn while the segments saved pay for the
/// error added, pruning spikes first (see [`prune_spikes`]), since beautify never removes a
/// corner. Returns the error of the shape kept.
pub fn simplify_with(
    ctx: &Context,
    place: &Place,
    cost: &CoverageCost,
    shape: &mut Shape,
    tolerances: &[f32],
    corner_angle: f32,
) -> f32 {
    let origin = place.window.origin();
    let mut error = cost.value(&shape.path(origin));
    // Hidden stretches are bridged, not pruned, so a notch around a neighbor's end is not
    // filled in where the neighbor may fail to cover it
    let is_free = place.free_at(ctx);
    for &tolerance in tolerances {
        let loops = shape
            .loops
            .iter()
            .enumerate()
            .map(|(l, cubics)| {
                let flags = shape.corner_flags(l, corner_angle);
                let (pruned, flags) = prune_spikes(cubics, &flags, tolerance, &is_free);
                crate::tracer::beautify::beautify_chain_at(&pruned, &flags, tolerance)
            })
            .collect();
        let mut simpler = shape.successor(loops).with_corners(shape.corners.clone());
        let saved = shape.cost_segments() - simpler.cost_segments();
        if saved <= 0. {
            continue;
        }
        let simpler_error = fit(cost, &mut simpler, origin, ctx.pull);
        if ((simpler_error - error) as f64) >= saved * ctx.segment_cost {
            break;
        }
        *shape = simpler;
        error = simpler_error;
    }
    error
}

/// How much deeper than wide a protrusion may be and still be a spike: deeper, it is a limb,
/// or the whole of a thin line, whose two long sides lie within any tolerance wider than it.
const SPIKE_DEPTH: f32 = 4.;

/// The loop and its corner flags with every spike or notch narrower than `tolerance` removed,
/// repeatedly: at a corner whose sides run within the tolerance of each other for at most
/// [`SPIKE_DEPTH`] tolerances, both go for a line. Free sides stay; a loop keeps three sides.
pub fn prune_spikes(
    cubics: &[crate::tracer::Cubic],
    corners: &[bool],
    tolerance: f32,
    is_free: &impl Fn(Vec2) -> bool,
) -> (Vec<crate::tracer::Cubic>, Vec<bool>) {
    use crate::tracer::fit::eval;
    const SAMPLES: usize = 8;
    let mut c = cubics.to_vec();
    let mut flags = corners.to_vec();
    'again: while c.len() > 3 {
        let n = c.len();
        for j in 0..n {
            let i = (j + n - 1) % n;
            let (a, b) = (c[i], c[j]);
            if !flags[j] {
                continue;
            }
            // Both sides within the tolerance of each other: every sample of the shorter side
            // near some sample of the longer one
            let samples = |q: crate::tracer::Cubic| -> Vec<Vec2> {
                (0..=SAMPLES)
                    .map(|k| eval(&q, k as f32 / SAMPLES as f32))
                    .collect()
            };
            let (short, long) = if chord_length(&a) <= chord_length(&b) {
                (a, b)
            } else {
                (b, a)
            };
            if chord_length(&short) > SPIKE_DEPTH * tolerance {
                continue;
            }
            let longer = samples(long);
            let shorter = samples(short);
            if longer.iter().chain(&shorter).any(|&p| is_free(p)) {
                continue;
            }
            let thin = shorter.iter().all(|p| {
                longer
                    .iter()
                    .map(|q| p.distance(*q))
                    .fold(f32::INFINITY, f32::min)
                    <= tolerance
            });
            if !thin {
                continue;
            }
            let (start, end) = (a[0], b[3]);
            let joined = [
                start,
                start.lerp(end, 1. / 3.),
                start.lerp(end, 2. / 3.),
                end,
            ];
            if i < j {
                c[i] = joined;
                c.remove(j);
                flags.remove(j);
            } else {
                // The pair wraps around the end of the loop
                c[j] = joined;
                c.remove(i);
                flags.remove(i);
            }
            continue 'again;
        }
        break;
    }
    (c, flags)
}

/// The length of a cubic's control polygon, an upper bound on its length.
fn chord_length(c: &crate::tracer::Cubic) -> f32 {
    (c[1] - c[0]).length() + (c[2] - c[1]).length() + (c[3] - c[2]).length()
}

/// Fits every shape once more over and under its final neighbors, whose colors may have changed
/// since its fit; shapes nothing around has changed for are left as they are.
pub async fn settle(ctx: &Context, scene: &mut Scene) {
    for k in 1..scene.shapes.len() {
        crate::progress::poll().await;
        let mut shape = scene.shapes[k].clone();
        let window = Window::around(
            shape.paint_min(),
            shape.paint_max(),
            MARGIN,
            ctx.width,
            ctx.height,
        );
        if window.is_empty() || !scene.is_stale(k, window) {
            continue;
        }
        let below = scene.render(window, 0..k, [1.; 3]);
        let above = scene.layer(window, k + 1);
        let clip = scene.clip_mask(window, k);
        if let Some((outer, primitive)) = shape.primitive {
            // A primitive stays exact: only its parameters move
            let cost = coverage_cost(
                ctx,
                window,
                shape.fill,
                &below,
                clip.as_deref(),
                above.as_ref(),
            );
            shape =
                crate::complete::refine_primitive(&cost, &shape, outer, primitive, window.origin())
                    .1;
        } else {
            refine(
                ctx,
                &mut shape,
                window,
                &below,
                clip.as_deref(),
                above.as_ref(),
            );
        }
        let change = scene.change();
        scene.shapes[k] = Shape {
            changed: change,
            fitted: change,
            ..shape
        };
    }
}
