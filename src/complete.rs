//! Completing outlines where nothing constrains them (hidden, left to later detail, outside the
//! clip): one smooth cubic bridging each free stretch, and the circle, ellipse or rounded
//! rectangle matching the stretches that show, compete with the outline as fitted.

use crate::tracer::Cubic;
use crate::tracer::fit::eval;
use crate::tracer::primitives::Primitive;
use glam::Vec2;

use crate::render::coverage_cost::CoverageCost;

use crate::refine::{Place, fit};
use crate::{Context, Shape};

/// Points sampled per cubic to tell whether it runs through free pixels and to fit primitives.
const SAMPLES: usize = 8;
/// Fewest samples of an outline that must show for a primitive to be fitted to them, in total and
/// as a share of all.
const MIN_VISIBLE: usize = 12;
const MIN_VISIBLE_SHARE: f32 = 0.3;
/// Mean distance in pixels between an outline and a primitive proposed for it. Loose, since the
/// objective decides; tight enough to skip the hopeless fits.
pub const PRIMITIVE_TOLERANCE: f32 = 1.;

/// Replaces `shape` by the best of its completions at `place` when that costs less in error plus
/// segments. `cost` is that of drawing the shape's color there under `ctx`, see [`Place::cost`].
/// See [`Completion`] for what is proposed. Returns the error of the shape kept.
pub fn complete(
    ctx: &Context,
    place: &Place,
    cost: &CoverageCost,
    shape: &mut Shape,
    what: Completion,
) -> f32 {
    let window = place.window;
    let origin = window.origin();
    let is_free = place.free_at(ctx);
    let total = |error: f32, shape: &Shape| error as f64 + shape.cost_segments() * ctx.segment_cost;

    let mut error = cost.value(&shape.path(origin));
    let mut proposals = Vec::new();
    let loops: Vec<Vec<Cubic>> = shape.loops.iter().map(|l| bridge(l, &is_free)).collect();
    // A bridge across every hidden stretch, whether or not it saves segments: a hidden stretch
    // left as traced follows an edge of whatever hid it and drifts once free
    if loops != shape.loops {
        let mut bridged = shape.successor(loops).with_corners(shape.corners.clone());
        let bridged_error = fit(cost, &mut bridged, origin, ctx.pull);
        proposals.push((bridged_error, bridged));
    }
    // A primitive for the outer loop, keeping any holes, fitted to the stretches that show: not
    // under the shapes drawn later (see [`Place::hidden`]) nor past the window
    let outer = (0..shape.loops.len()).max_by(|&a, &b| {
        let area = |k: usize| crate::tracer::signed_area(&shape.loops[k]).abs();
        area(a).total_cmp(&area(b))
    });
    if let Some(outer) = outer.filter(|_| what.primitives) {
        let hidden = place.hidden(ctx);
        let is_hidden = |p: Vec2| {
            let (x, y) = (p.x.floor() - window.x as f32, p.y.floor() - window.y as f32);
            x < 0.
                || y < 0.
                || (x as usize) >= window.width
                || (y as usize) >= window.height
                || hidden[y as usize * window.width + x as usize]
        };
        let all: Vec<Vec2> = shape.loops[outer]
            .iter()
            .flat_map(|c| (0..SAMPLES).map(move |i| eval(c, i as f32 / SAMPLES as f32)))
            .collect();
        let visible: Vec<Vec2> = all.iter().copied().filter(|&p| !is_hidden(p)).collect();
        let enough = visible.len() >= MIN_VISIBLE
            && visible.len() as f32 >= MIN_VISIBLE_SHARE * all.len() as f32;
        if enough
            && let Some(primitive) =
                crate::tracer::primitives::fit_primitive(&visible, &all, what.primitive_tolerance)
        {
            let (refined_error, refined) = refine_primitive(cost, shape, outer, primitive, origin);
            proposals.push((refined_error - what.primitive_allowance, refined));
        }
    }
    // A proposal is judged on its place's pixels only, so geometry more than a frame's size past
    // the frame is never judged and is refused, as a circle fitted to a nearly straight stretch
    // whose huge radius would ask for a raster of terabytes
    let (fw, fh) = (ctx.width as f32, ctx.height as f32);
    let within =
        |s: &Shape| s.min.x >= -fw && s.min.y >= -fh && s.max.x <= 2. * fw && s.max.y <= 2. * fh;
    // A proposal stays in the window (but where that meets the frame), since nothing judges it
    // past it: a primitive fitted to the stretch that shows would draw there unmeasured
    let (x0, y0) = (window.x as f32, window.y as f32);
    let (x1, y1) = (x0 + window.width as f32, y0 + window.height as f32);
    let judged = |s: &Shape| {
        s.loops.iter().flatten().all(|c| {
            (0..=SAMPLES).all(|i| {
                let p = eval(c, i as f32 / SAMPLES as f32);
                (p.x >= x0 || window.x == 0)
                    && (p.y >= y0 || window.y == 0)
                    && (p.x <= x1 || x1 >= fw)
                    && (p.y <= y1 || y1 >= fh)
            })
        })
    };
    for (proposal_error, proposal) in proposals {
        if within(&proposal)
            && judged(&proposal)
            && total(proposal_error, &proposal) < total(error, shape)
        {
            *shape = proposal;
            error = proposal_error;
        }
    }
    error
}

/// What [`complete`] proposes besides the bridges across hidden stretches.
#[derive(Clone, Copy, Debug)]
pub struct Completion {
    /// A circle, ellipse or rounded rectangle for the outer loop; their discount is meant for
    /// single-color objects, where such shapes are what an artist drew, not for patches of fur.
    pub primitives: bool,
    /// Mean distance in pixels the outline may lie from a primitive proposed for it: a pixel,
    /// or half a block for pixel art rendering a smooth shape in blocks.
    pub primitive_tolerance: f32,
    /// Error a primitive is allowed over the outline it replaces: what the blocks of pixel art
    /// rendering a smooth shape cost against the smooth shape itself, which is the rendering's
    /// noise rather than the shape's structure. Zero for an anti-aliased edge.
    pub primitive_allowance: f32,
}

/// A closed loop with every run of cubics through free pixels replaced by one cubic that leaves
/// and enters the rest of the loop smoothly. A loop that is free all around is kept as it is.
pub(crate) fn bridge(cubics: &[Cubic], is_free: &impl Fn(Vec2) -> bool) -> Vec<Cubic> {
    let n = cubics.len();
    let free: Vec<bool> = cubics
        .iter()
        .map(|c| (1..SAMPLES).all(|i| is_free(eval(c, i as f32 / SAMPLES as f32))))
        .collect();
    let Some(start) = (0..n).find(|&i| !free[i]) else {
        return cubics.to_vec();
    };
    // Walk the loop from a cubic that shows, so no run wraps around the start
    let mut out = Vec::with_capacity(n);
    let mut k = 0;
    while k < n {
        let i = (start + k) % n;
        if !free[i] {
            out.push(cubics[i]);
            k += 1;
            continue;
        }
        let mut len = 0;
        while k + len < n && free[(start + k + len) % n] {
            len += 1;
        }
        let first = &cubics[i];
        let last = &cubics[(i + len - 1) % n];
        let before = &cubics[(i + n - 1) % n];
        let after = &cubics[(i + len) % n];
        out.push(smooth_join(
            first[0],
            tangent_out(before),
            last[3],
            tangent_in(after),
        ));
        k += len;
    }
    out
}

/// The turn at the joint before cubic `j` of the loop `cubics`; None beside a cubic of no length,
/// which has no direction (in practice a loop of one point, a single such cubic): no corner.
pub(crate) fn joint_turn(cubics: &[Cubic], j: usize) -> Option<f32> {
    let i = (j + cubics.len() - 1) % cubics.len();
    crate::tracer::angle::turn(tangent_out(&cubics[i]), tangent_in(&cubics[j]))
}

/// The direction a cubic ends in.
pub(crate) fn tangent_out(c: &Cubic) -> Vec2 {
    [c[2], c[1], c[0]]
        .into_iter()
        .map(|p| c[3] - p)
        .find(|d| d.length_squared() > 1e-8)
        .unwrap_or(Vec2::ZERO)
        .normalize_or_zero()
}

/// The direction a cubic starts in.
pub(crate) fn tangent_in(c: &Cubic) -> Vec2 {
    [c[1], c[2], c[3]]
        .into_iter()
        .map(|p| p - c[0])
        .find(|d| d.length_squared() > 1e-8)
        .unwrap_or(Vec2::ZERO)
        .normalize_or_zero()
}

/// The cubic from `a` leaving in direction `ta` to `b` arriving in direction `tb`, with handles
/// sized so that it is a circular arc when the directions allow one.
fn smooth_join(a: Vec2, ta: Vec2, b: Vec2, tb: Vec2) -> Cubic {
    let chord = a.distance(b);
    // an end without a direction joins with the straight case's handles
    let handle = match crate::tracer::angle::turn(ta, tb).map(f32::abs) {
        Some(turn) if turn >= 1e-3 => {
            // An arc turning by `turn` over the chord has radius chord / (2 sin(turn / 2)),
            // and its cubic has handles of 4/3 tan(turn / 4) radii
            let radius = chord / (2. * (turn / 2.).sin());
            (4. / 3. * (turn / 4.).tan() * radius).min(chord)
        }
        _ => chord / 3.,
    };
    [a, a + ta * handle, b - tb * handle, b]
}

/// Optimizer steps at most for a primitive's parameters.
const PRIMITIVE_STEPS: usize = 40;
/// Step size for a primitive's lengths in pixels, roughly; angles move by the same arc.
const PRIMITIVE_RATE: f32 = 0.1;

/// The `primitive` for loop `outer` of `shape` with its parameters moved to lower `cost`, since
/// fitted to outline samples alone it lies up to a pixel off. Returns the error and the shape
/// with the loop replaced.
pub fn refine_primitive(
    cost: &CoverageCost,
    shape: &Shape,
    outer: usize,
    primitive: Primitive,
    origin: Vec2,
) -> (f32, Shape) {
    let with_loop = |p: &Primitive| with_primitive(shape, outer, p);
    let mut params = primitive.params();
    let n = params.len();
    // Lengths step in pixels, an angle by the same arc at the primitive's size
    let size = params[2..n.min(4)].iter().fold(1f32, |a, &b| a.max(b));
    let is_angle = |j: usize| {
        matches!(primitive, Primitive::Ellipse { .. } if j == 4)
            || matches!(primitive, Primitive::RoundedRect { .. } if j == 5)
    };
    let rate = |j: usize| {
        if is_angle(j) {
            PRIMITIVE_RATE / size
        } else {
            PRIMITIVE_RATE
        }
    };

    let mut best = (f32::INFINITY, params.clone());
    let mut since_best = 0;
    let mut scale = 1f32;
    let (mut m, mut v) = (vec![0f32; n], vec![0f32; n]);
    for step in 1..=PRIMITIVE_STEPS {
        let current = primitive.with_params(&params);
        let candidate = with_loop(&current);
        let (value, grad) = cost.evaluate(&candidate.path(origin));
        if value < best.0 * (1. - 1e-3) {
            since_best = 0;
        } else {
            since_best += 1;
        }
        if value < best.0 {
            best = (value, params.clone());
        }
        if since_best >= 5 {
            scale *= 0.5;
            since_best = 0;
            if scale < 1. / 16. {
                break;
            }
        }
        // The gradient by the parameters: the points move with each parameter by differences,
        // taken one-sided where a step changes the outline's cubics (a corner arc appearing)
        let points = &grad[outer];
        let here = current.cubics();
        let mut g = vec![0f32; n];
        for (j, gj) in g.iter_mut().enumerate() {
            let h = if is_angle(j) { 1e-3 } else { 1e-2 };
            let moved = |by: f32| {
                let mut p = params.clone();
                p[j] += by;
                Some(primitive.with_params(&p).cubics()).filter(|c| c.len() == here.len())
            };
            let (hi, lo, span) = match (moved(h), moved(-h)) {
                (Some(up), Some(down)) => (up, down, 2. * h),
                (Some(up), None) => (up, here.clone(), h),
                (None, Some(down)) => (here.clone(), down, h),
                (None, None) => continue,
            };
            for (k, (cu, cd)) in hi.iter().zip(&lo).enumerate() {
                for i in 0..3 {
                    *gj += points[3 * k + i].dot((cu[i] - cd[i]) / span);
                }
            }
        }
        // Adam, with bias correction
        for j in 0..n {
            m[j] = 0.9 * m[j] + 0.1 * g[j];
            v[j] = 0.999 * v[j] + 0.001 * g[j] * g[j];
            let m_hat = m[j] / (1. - 0.9f32.powi(step as i32));
            let v_hat = v[j] / (1. - 0.999f32.powi(step as i32));
            params[j] -= scale * rate(j) * m_hat / (v_hat.sqrt() + 1e-6);
        }
    }
    let refined = primitive.with_params(&best.1);
    (best.0, with_loop(&refined))
}

/// `shape` with loop `outer` replaced by `primitive`.
fn with_primitive(shape: &Shape, outer: usize, primitive: &Primitive) -> Shape {
    let mut loops = shape.loops.clone();
    loops[outer] = primitive.cubics();
    Shape {
        primitive: Some((outer, *primitive)),
        ..shape.successor(loops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rounded rectangle whose corner arcs or sides come and go within the difference step
    /// (radius at the arcs' threshold, or at half the side) refines without panicking.
    #[test]
    fn primitive_refines_where_its_cubics_change() {
        let (w, h) = (40usize, 40usize);
        let inside = |i: usize| {
            let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
            ((10.0..30.0).contains(&x) && (12.0..28.0).contains(&y)) as u8 as f32
        };
        let a = vec![1f32; w * h];
        let b: Vec<f32> = (0..w * h).map(|i| -2. * inside(i)).collect();
        let c: Vec<f32> = (0..w * h).map(inside).collect();
        let cost = CoverageCost::new(w, h, a, b, &c);
        for (half, radius) in [(Vec2::new(10., 8.), 0.295), (Vec2::new(8., 8.), 7.995)] {
            let primitive = Primitive::RoundedRect {
                center: Vec2::new(20., 20.),
                half,
                radius,
                angle: 0.,
            };
            let shape = Shape::new(
                vec![primitive.cubics()],
                crate::Fill::solid(glam::Vec3::ZERO),
            );
            let (error, refined) = refine_primitive(&cost, &shape, 0, primitive, Vec2::ZERO);
            assert!(error.is_finite() && refined.loops.len() == 1, "{error}");
        }
    }

    /// An end without a direction joins with the straight case's handles, and a loop of one
    /// point has no turn at its joint.
    #[test]
    fn no_direction_is_no_turn() {
        let (a, b) = (Vec2::ZERO, Vec2::new(6., 0.));
        let join = smooth_join(a, Vec2::ZERO, b, Vec2::X);
        assert_eq!(join, [a, a, b - Vec2::new(2., 0.), b]);
        assert_eq!(joint_turn(&[[b; 4]], 0), None);
    }
}
