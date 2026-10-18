//! Cleanup of fitted outlines where the free fit left noise: nearly straight cubics made
//! straight and snapped to the axes, collinear lines merged, rounded corners restored, kinks
//! smoothed. A step is kept when error plus segment cost rises by at most a small share.

use crate::tracer::Cubic;
use glam::Vec2;

use crate::render::coverage_cost::CoverageCost;

use crate::context::Context;
use crate::refine::{MARGIN, Place};
use crate::scene::{Scene, Shape};

/// How far a handle may lie from the chord for the cubic to be a line, how far a line may lean
/// from an axis to be snapped to it, and how far a joint may lie off the line through its
/// neighbors to be merged away, in pixels.
const TOLERANCE: f32 = 0.5;
/// How far a corner may move to the point where its sides, extended, meet.
const CORNER_REACH: f32 = 3. * TOLERANCE;
/// How far from a joint each side is taken to extend it: beyond where the fit rounds a corner
/// off, within the radius of the smallest rounded corner meant to stay round.
const CORNER_WINDOW: f32 = 1.;
/// How far an axis-aligned line may lie from the pixel grid to be put on it: a pixel edge.
const GRID: f32 = 0.25;
/// Turning angle above which a joint between two lines is a corner rather than a bend.
const CORNER_ANGLE: f32 = crate::outline::CORNER_ANGLE;
/// Turning angle below which a joint is meant to be smooth and only kinked by the fit. Between
/// this and the corner angle a joint is left as it is: a circle meeting the edge of the square
/// on it turns by 60 degrees there, and aligning those handles bends both curves.
const SMOOTH_ANGLE: f32 = 20f32.to_radians();
/// Share of a shape's error its cleanup may add.
const SLACK: f64 = 0.02;

/// Cleans every shape of `scene` at its place, each step kept where its cost rises by at most
/// [`SLACK`] of the error, see [`clean_shape`].
pub fn cleanup(ctx: &Context, scene: &mut Scene) {
    let mut cleaned = 0;
    let mut removed = 0;
    for k in 1..scene.shapes.len() {
        if crate::progress::poll() {
            break;
        }
        if scene.shapes[k].primitive.is_some() {
            continue;
        }
        // Out of the scene while it is measured, so it is not above itself; the change count
        // moves with it, since the whole-scene render is cached by it
        let shape = scene.shapes.remove(k);
        scene.change();
        let Some(place) = Place::new(ctx, scene, &shape, k, MARGIN) else {
            scene.shapes.insert(k, shape);
            scene.change();
            continue;
        };
        let cost = place.cost(ctx, shape.fill);
        let origin = place.window.origin();
        let error = cost.value(&shape.path(origin));
        let (candidate, _) = clean_shape(ctx, &cost, origin, &shape, error);
        if candidate.loops != shape.loops {
            cleaned += 1;
            removed += shape.segments() - candidate.segments();
            let change = scene.change();
            scene.shapes.insert(
                k,
                Shape {
                    id: shape.id,
                    changed: change,
                    fitted: change,
                    ..candidate
                },
            );
        } else {
            scene.shapes.insert(k, shape);
            scene.change();
        }
    }
    if crate::timing::enabled() {
        eprintln!("cleanup: {cleaned} shapes cleaned, {removed} segments removed");
    }
}

/// Cleans `shape` where `cost` measures it, between the fit and the simplification, so the
/// latter sees corners and lines the fit rounded off. Returns the error of the shape kept,
/// given `error` for the shape as it is.
pub fn clean_in(
    ctx: &Context,
    cost: &CoverageCost,
    origin: Vec2,
    shape: &mut Shape,
    error: f32,
) -> f32 {
    let (cleaned, error) = clean_shape(ctx, cost, origin, shape, error);
    *shape = cleaned;
    error
}

/// `shape` cleaned step by step (the lines, each corner on its own, the joints), each step kept
/// when the error plus segment cost rises by at most [`SLACK`] of the error, so one overreaching
/// step does not take the harmless ones down with it. Returns the shape and its error.
fn clean_shape(
    ctx: &Context,
    cost: &CoverageCost,
    origin: Vec2,
    shape: &Shape,
    error: f32,
) -> (Shape, f32) {
    let mut current = shape.clone();
    let mut error = error;
    let total = |e: f32, s: &Shape| e as f64 + s.cost_segments() * ctx.segment_cost;
    let try_step = |loops: Vec<Vec<Cubic>>, current: &mut Shape, error: &mut f32| {
        if loops == current.loops {
            return;
        }
        let candidate = Shape {
            clip: current.clip,
            id: current.id,
            smooth: current.smooth,
            antialiased: current.antialiased,
            node: current.node,
            layer: current.layer,
            blur: current.blur,
            ..Shape::new(loops, current.fill)
        }
        .with_corners(current.corners.clone());
        let cleaned = cost.value(&candidate.path(origin));
        let keep = total(cleaned, &candidate) <= total(*error, current) + SLACK * *error as f64;
        if keep {
            *current = candidate;
            *error = cleaned;
        }
    };
    // Lines
    let loops: Vec<Vec<Cubic>> = current.loops.iter().map(|l| clean_lines(l)).collect();
    try_step(loops, &mut current, &mut error);
    // Corners, one at a time
    for l in 0..current.loops.len() {
        let n = current.loops[l].len();
        for j in 0..n {
            let remembered = current.corner_flag(l, j, CORNER_ANGLE);
            let mut loops = current.loops.clone();
            if !meet_corner(&mut loops[l], j, remembered) {
                continue;
            }
            try_step(loops, &mut current, &mut error);
        }
    }
    // Joints
    let loops: Vec<Vec<Cubic>> = current
        .loops
        .iter()
        .enumerate()
        .map(|(l, cubics)| {
            let mut c = cubics.clone();
            let straight: Vec<bool> = c.iter().map(is_line).collect();
            let corners = current.corner_flags(l, CORNER_ANGLE);
            smooth_joints(&mut c, &straight, &corners);
            c
        })
        .collect();
    try_step(loops, &mut current, &mut error);
    (current, error)
}

/// A closed loop with its lines cleaned up: tiny cubics dropped, collinear lines merged, lines
/// snapped to the axes and their handles put at thirds.
pub fn clean_lines(cubics: &[Cubic]) -> Vec<Cubic> {
    let mut c = cubics.to_vec();
    drop_tiny(&mut c);
    let mut straight: Vec<bool> = c.iter().map(is_line).collect();
    merge_collinear(&mut c, &mut straight);
    snap_axes(&mut c, &straight);
    thirds(&mut c, &straight);
    c
}

/// Puts a line's handles at thirds of its chord.
fn thirds(c: &mut [Cubic], straight: &[bool]) {
    for (cubic, &line) in c.iter_mut().zip(straight) {
        if line {
            cubic[1] = cubic[0].lerp(cubic[3], 1. / 3.);
            cubic[2] = cubic[0].lerp(cubic[3], 2. / 3.);
        }
    }
}

/// Whether the cubic's handles lie within the tolerance of its chord.
fn is_line(c: &Cubic) -> bool {
    let chord = c[3] - c[0];
    let length = chord.length();
    if length < 1e-3 {
        return true;
    }
    let normal = chord.perp() / length;
    [c[1], c[2]]
        .iter()
        .all(|&p| (p - c[0]).dot(normal).abs() <= TOLERANCE)
}

/// Removes cubics that stay within the tolerance of their start point, joining their neighbors.
fn drop_tiny(c: &mut Vec<Cubic>) {
    while c.len() > 1 {
        let Some(i) = c
            .iter()
            .position(|q| q[1..].iter().all(|&p| (p - q[0]).length() <= TOLERANCE))
        else {
            return;
        };
        let end = c[i][3];
        let n = c.len();
        c[(i + n - 1) % n][3] = end;
        c.remove(i);
    }
}

/// Joins neighboring lines whose joint lies within the tolerance of the line through their
/// outer ends.
fn merge_collinear(c: &mut Vec<Cubic>, straight: &mut Vec<bool>) {
    while c.len() > 1 {
        let n = c.len();
        let Some(i) = (0..n).find(|&i| {
            let j = (i + 1) % n;
            if !straight[i] || !straight[j] {
                return false;
            }
            let (a, b, joint) = (c[i][0], c[j][3], c[i][3]);
            let chord = b - a;
            let length = chord.length();
            length > 1e-3 && (joint - a).perp_dot(chord).abs() / length <= TOLERANCE
        }) else {
            return;
        };
        let j = (i + 1) % n;
        c[i][3] = c[j][3];
        c.remove(j);
        straight.remove(j);
    }
}

/// Moves joint `j` to `to`, taking the handles on both sides of it along.
fn move_joint(c: &mut [Cubic], j: usize, to: Vec2) {
    let n = c.len();
    let i = (j + n - 1) % n;
    let delta = to - c[j][0];
    c[j][0] = to;
    c[j][1] += delta;
    c[i][3] = to;
    c[i][2] += delta;
}

/// Makes lines that lean less than the tolerance from an axis exactly horizontal or vertical,
/// moving both their ends to the mean, and onto the pixel grid when that is within reach.
fn snap_axes(c: &mut [Cubic], straight: &[bool]) {
    let n = c.len();
    for i in 0..n {
        if !straight[i] {
            continue;
        }
        let d = c[i][3] - c[i][0];
        let j = (i + 1) % n;
        if d.x.abs() <= TOLERANCE && d.y.abs() > d.x.abs() {
            let x = grid((c[i][0].x + c[i][3].x) / 2.);
            move_joint(c, i, Vec2::new(x, c[i][0].y));
            move_joint(c, j, Vec2::new(x, c[i][3].y));
        } else if d.y.abs() <= TOLERANCE && d.x.abs() > d.y.abs() {
            let y = grid((c[i][0].y + c[i][3].y) / 2.);
            move_joint(c, i, Vec2::new(c[i][0].x, y));
            move_joint(c, j, Vec2::new(c[i][3].x, y));
        }
    }
}

/// `v` rounded to the pixel grid when within reach of it.
fn grid(v: f32) -> f32 {
    if (v - v.round()).abs() <= GRID {
        v.round()
    } else {
        v
    }
}

/// Puts the corner at joint `j` where its sides, taken a [`CORNER_WINDOW`] back (past the fit's
/// rounding) and extended, meet; says whether it moved. Bends and meetings out of reach stay; a
/// `remembered` corner from the trace counts whatever its angle. Lines extend along their chord.
fn meet_corner(c: &mut [Cubic], j: usize, remembered: bool) -> bool {
    let n = c.len();
    let i = (j + n - 1) % n;
    if i == j {
        return false;
    }
    let straight = [is_line(&c[i]), is_line(&c[j])];
    let joint = c[j][0];
    let (pa, ta) = if straight[0] {
        (c[i][3], (c[i][3] - c[i][0]).normalize_or_zero())
    } else {
        trim(&c[i], true)
    };
    let (pb, tb) = if straight[1] {
        (c[j][0], (c[j][3] - c[j][0]).normalize_or_zero())
    } else {
        trim(&c[j], false)
    };
    // past the zero tests the turn is None only for coordinates that are not finite
    let turn = crate::tracer::angle::turn(ta, tb).map(f32::abs);
    if ta == Vec2::ZERO
        || tb == Vec2::ZERO
        || (!remembered && turn.is_some_and(|t| t < CORNER_ANGLE))
        || turn.is_some_and(|t| t < SMOOTH_ANGLE)
    {
        return false;
    }
    let denominator = ta.perp_dot(tb);
    if denominator.abs() < 1e-6 {
        return false;
    }
    // Ahead of the first side's trim point and behind the second's, within reach
    let s = (pb - pa).perp_dot(tb) / denominator;
    let u = (pb - pa).perp_dot(ta) / denominator;
    let meet = pa + s * ta;
    if s < 0. || u > 0. || (meet - joint).length() > CORNER_REACH || meet.distance(joint) < 1e-4 {
        return false;
    }
    let arm_a = (c[i][3] - c[i][2]).length().max(CORNER_WINDOW / 3.);
    let arm_b = (c[j][1] - c[j][0]).length().max(CORNER_WINDOW / 3.);
    c[i][3] = meet;
    c[i][2] = meet - ta * arm_a;
    c[j][0] = meet;
    c[j][1] = meet + tb * arm_b;
    if straight[0] {
        c[i][1] = c[i][0].lerp(c[i][3], 1. / 3.);
        c[i][2] = c[i][0].lerp(c[i][3], 2. / 3.);
    }
    if straight[1] {
        c[j][1] = c[j][0].lerp(c[j][3], 1. / 3.);
        c[j][2] = c[j][0].lerp(c[j][3], 2. / 3.);
    }
    true
}

/// The point a [`CORNER_WINDOW`] along `c` from its end (`from_end`) or its start, with the
/// unit tangent there in the direction of travel. A cubic shorter than the window gives its
/// far end.
fn trim(c: &Cubic, from_end: bool) -> (Vec2, Vec2) {
    const STEPS: usize = 16;
    let at = |t: f32| crate::tracer::fit::eval(c, t);
    let tangent = |t: f32| {
        let mt = 1. - t;
        let d =
            3. * mt * mt * (c[1] - c[0]) + 6. * mt * t * (c[2] - c[1]) + 3. * t * t * (c[3] - c[2]);
        let d = if d.length_squared() > 1e-8 {
            d
        } else {
            c[3] - c[0]
        };
        d.normalize_or_zero()
    };
    let mut walked = 0.;
    let mut t = if from_end { 1. } else { 0. };
    let step = 1. / STEPS as f32;
    for k in 0..STEPS {
        let (t0, t1) = if from_end {
            (1. - k as f32 * step, 1. - (k + 1) as f32 * step)
        } else {
            (k as f32 * step, (k + 1) as f32 * step)
        };
        let length = (at(t1) - at(t0)).length();
        if walked + length >= CORNER_WINDOW {
            let f = if length > 1e-6 {
                (CORNER_WINDOW - walked) / length
            } else {
                1.
            };
            t = t0 + (t1 - t0) * f;
            return (at(t), tangent(t));
        }
        walked += length;
        t = t1;
    }
    (at(t), tangent(t))
}

/// Aligns the handles on both sides of every joint that turns by less than [`SMOOTH_ANGLE`],
/// so the outline has no kinks. A line keeps its direction and the curve next to it follows;
/// two curves meet in the mean of their directions. Handles keep their lengths.
fn smooth_joints(c: &mut [Cubic], straight: &[bool], corners: &[bool]) {
    let n = c.len();
    for j in 0..n {
        let i = (j + n - 1) % n;
        if i == j || (straight[i] && straight[j]) || corners[j] {
            continue;
        }
        let joint = c[j][0];
        let (d_in, d_out) = (joint - c[i][2], c[j][1] - joint);
        if d_in.length_squared() < 1e-8 || d_out.length_squared() < 1e-8 {
            continue;
        }
        // both handles have a length here, so the turn is None only for coordinates not finite
        if crate::tracer::angle::turn(d_in, d_out).is_some_and(|t| t.abs() > SMOOTH_ANGLE) {
            continue;
        }
        let direction = match (straight[i], straight[j]) {
            (true, _) => d_in.normalize(),
            (_, true) => d_out.normalize(),
            _ => (d_in.normalize() + d_out.normalize()).normalize_or_zero(),
        };
        if direction == Vec2::ZERO {
            continue;
        }
        if !straight[i] {
            c[i][2] = joint - direction * d_in.length();
        }
        if !straight[j] {
            c[j][1] = joint + direction * d_out.length();
        }
    }
}
