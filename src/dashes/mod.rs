//! Dashed strokes: strokes at one width that continue one another (each end on the other's
//! continuation within half a width) are linked, the cheapest vertex-disjoint chains become one
//! stroke through them, and the pattern is read off as a dash and gap or a repeating period.

use glam::Vec2;

use crate::tracer::fit::eval;

use crate::refine::{MARGIN, Place, composite_error};
use crate::stroke::{
    CENTERLINE_TOLERANCE, Cap, Dashes, KINK, MAX_WIDTH, Stroke, WIDTH_QUANTUM, WIDTH_SLACK,
    extend_cubic, fit, refit, tangent,
};
use crate::{Context, Scene, Shape, Window};

mod dots;
mod leaves;
mod pattern;
mod profile;

use dots::*;
pub use leaves::from_leaves;
use pattern::*;
use profile::*;

/// Fewest strokes that make a dashed stroke.
const MIN_DASHES: usize = 2;
/// Longest period of a repeating pattern looked for, in dashes.
const MAX_PERIOD: usize = 4;
/// Fewest dots that make a dotted stroke: two spacings, to tell a chain from any two discs.
const MIN_DOTS: usize = 3;
/// The shortest dash drawn under a butt or square cap, in pixels: a butt dash of no length
/// paints nothing and a square one has no direction; half a pixel keeps the entries positive.
const DOT_LENGTH: f32 = 0.5;

/// How far a dotted stroke's centerline runs on past its end dots as fitted, so a dot is no
/// dash on an end, which the pattern does not draw (lacebobbins' last): where the half-pixel
/// dashes dots were drawn with reached; stretched farther, the end cubics move the dots.
const DOT_MARGIN: f32 = DOT_LENGTH / 2.;

/// The shortest dash drawn under `cap`: none under a round cap, which draws it as a dot.
fn shortest(cap: Cap) -> f32 {
    if cap == Cap::Round { 0. } else { DOT_LENGTH }
}
/// How far apart two leaves' colours may lie and be one paint: the contrast a pattern's dashes
/// must have against their gaps ([`profile`]'s), in units of the noise.
const ALIKE_COLOR: f32 = 3. * crate::translucency::NOISE;

/// An end of a dash: which dash, and which of its two ends.
type End = (usize, usize);

/// A stroke of the scene that may be a dash, with its two ends: where its paint ends (a square
/// cap paints half a width past the centerline, as a butt cap that much farther would), which
/// way the path leaves there, and the cubic to continue it by.
struct Dash {
    position: usize,
    /// The shape's stroke, or the stroke a rectangle is.
    stroke: Stroke,
    ends: [(Vec2, Vec2, [Vec2; 4]); 2],
    /// How far the paint reaches past each end of the centerline along it.
    reach: f32,
    length: f32,
    width: f32,
    /// The image's colour of a dash read off the picture, which links only to alike ones.
    color: Option<glam::Vec3>,
}

impl Dash {
    /// The dash `stroke` is, at `position`, with the image's `color` where read off it.
    fn new(position: usize, stroke: Stroke, color: Option<glam::Vec3>) -> Self {
        let (first, last) = (
            stroke.centerline[0],
            stroke.centerline[stroke.centerline.len() - 1],
        );
        let ends = stroke.arc_lengths();
        let reach = if stroke.cap == Cap::Square {
            stroke.width / 2.
        } else {
            0.
        };
        // Each end's cubic is turned to run out of the stroke at that end
        let reversed = [first[3], first[2], first[1], first[0]];
        let (h0, h1) = (-tangent(&first, 0.), tangent(&last, 1.));
        Dash {
            position,
            ends: [
                (first[0] + h0 * reach, h0, reversed),
                (last[3] + h1 * reach, h1, last),
            ],
            reach,
            length: ends[ends.len() - 1] + 2. * reach,
            width: stroke.width,
            stroke,
            color,
        }
    }
}

/// Groups the scene's strokes into dashed strokes where that costs less in error plus segments
/// and paths. A chain keeps its members' stack order against the shapes they overlap
/// ([`chain_slot`]) and draws them where they are ([`off_phase`]).
pub fn group(ctx: &Context, scene: &mut Scene) {
    let dashes: Vec<Dash> = scene
        .shapes
        .iter()
        .enumerate()
        .filter_map(|(position, shape)| {
            // Gradients are not dashes
            if shape.clip.is_some() || shape.fill.gradient.is_some() {
                return None;
            }
            let stroke = stroke_of(shape)?;
            if stroke.closed || stroke.dashes.is_some() {
                return None;
            }
            Some(Dash::new(position, stroke, None))
        })
        .collect();
    // A short stroke may be a dash or a dot: the chains of dashes take theirs first, the
    // dots chain among the rest, so no two chains share a member, and all are planned on
    // the scene as it is
    let mut planned: Vec<(Vec<usize>, Stroke, &str)> = Vec::new();
    if dashes.len() >= MIN_DASHES {
        for (positions, dashed) in dash_chains(&dashes) {
            planned.push((positions, dashed, "dashes"));
        }
    }
    let mut taken = vec![false; scene.shapes.len()];
    for (positions, _, _) in &planned {
        for &p in positions {
            taken[p] = true;
        }
    }
    for (positions, dotted) in dot_chains(scene, &taken) {
        planned.push((positions, dotted, "dots"));
    }
    let mut replacements: Vec<(Vec<usize>, usize, Shape)> = Vec::new();
    for (positions, dashed, kind) in planned {
        if let Some((slot, shape)) = try_chain(ctx, scene, &positions, dashed, kind) {
            replacements.push((positions, slot, shape));
        }
    }
    // The stack rebuilt in one sweep, every chain's stroke in at its slot and its members out,
    // since positions are those of the scene as planned on and chains interleave (strokes
    // share a placeholder id, so ids are no help)
    let n = scene.shapes.len();
    let mut member_of: Vec<Option<usize>> = vec![None; n];
    let mut at_slot: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for (r, (members, slot, _)) in replacements.iter().enumerate() {
        for &p in members {
            member_of[p] = Some(r);
        }
        at_slot[*slot].push(r);
    }
    let mut shapes: Vec<Shape> = Vec::with_capacity(n);
    let mut strokes: Vec<Option<Shape>> =
        replacements.into_iter().map(|(_, _, s)| Some(s)).collect();
    let change = scene.change();
    let mut place = |slot: usize, shapes: &mut Vec<Shape>| {
        for &r in &at_slot[slot] {
            let mut stroke = strokes[r].take().expect("each chain placed once");
            stroke.changed = change;
            stroke.fitted = change;
            shapes.push(stroke);
        }
    };
    for (p, shape) in scene.shapes.drain(..).enumerate() {
        place(p, &mut shapes);
        if member_of[p].is_none() {
            shapes.push(shape);
        }
    }
    place(n, &mut shapes);
    scene.shapes = shapes;
    scene.change();
}

/// The point `by` past the end of `c` on the circle its end osculates (its tangent and its
/// curvature there), which unlike the cubic's own extrapolation does not run off past its span.
fn onward(c: &[Vec2; 4], by: f32) -> Vec2 {
    let d1 = 3. * (c[3] - c[2]);
    let d2 = 6. * (c[3] - 2. * c[2] + c[1]);
    let speed = d1.length();
    let t = tangent(c, 1.);
    if speed < 1e-6 {
        return c[3] + t * by;
    }
    let curvature = d1.perp_dot(d2) / (speed * speed * speed);
    let turn = curvature * by;
    if turn.abs() < 1e-4 {
        return c[3] + t * by;
    }
    c[3] + t * (turn.sin() / curvature) + t.perp() * ((1. - turn.cos()) / curvature)
}

/// The longest stretch along a stroke of length `total` where the pattern's dashes (`drawn`,
/// arc length spans) and those seen (`members`, the chain's or the image's) disagree, since a
/// dashed stroke must draw each dash and gap where it is (the ticked line of `b006-subwaymap-24`).
fn off_phase(drawn: &[(f32, f32)], members: &[(f32, f32)], total: f32, closed: bool) -> f32 {
    if total <= 0. {
        return 0.;
    }
    let on = |s: f32, spans: &[(f32, f32)]| {
        let s = if closed { s.rem_euclid(total) } else { s };
        spans.iter().any(|&(a, b)| {
            // a member's span may run past the end of a closed stroke and on from its start
            (a <= s && s <= b) || (closed && b > total && s + total <= b)
        })
    };
    // Sampled every half pixel along the stroke, the longest run of disagreement
    const STEP: f32 = 0.5;
    let (mut run, mut longest) = (0f32, 0f32);
    let mut s = 0.;
    while s <= total {
        if on(s, drawn) != on(s, members) {
            run += STEP;
            longest = longest.max(run);
        } else {
            run = 0.;
        }
        s += STEP;
    }
    longest
}

/// Whether two shapes overlap: some pixel is covered by half or more by each, so that which
/// of them is on top shows (shapes that only meet share their anti-aliased rims, at less).
fn overlap(a: &Shape, b: &Shape) -> bool {
    let (ra, rb) = (a.raster(), b.raster());
    let (x0, y0) = (ra.x0.max(rb.x0), ra.y0.max(rb.y0));
    let (x1, y1) = (
        (ra.x0 + ra.w).min(rb.x0 + rb.w),
        (ra.y0 + ra.h).min(rb.y0 + rb.h),
    );
    (y0..y1).any(|y| {
        (x0..x1).any(|x| {
            ra.values[(y - ra.y0) * ra.w + x - ra.x0] >= 0.5
                && rb.values[(y - rb.y0) * rb.w + x - rb.x0] >= 0.5
        })
    })
}

/// The position (in the scene as planned on) a chain's stroke goes in below: where its highest
/// member was, or lower, so every shape a member overlaps keeps its side of it (as the crossing
/// dashed lines of `b003-crossing-01`). None where no place keeps every side.
fn chain_slot(scene: &Scene, positions: &[usize]) -> Option<usize> {
    let highest = *positions.iter().max()?;
    // The stroke goes in above `lo` (the highest shape below a member it overlaps) and at
    // or below `hi` (the lowest shape above a member it overlaps)
    let (mut lo, mut hi) = (0, scene.shapes.len());
    for (s, shape) in scene.shapes.iter().enumerate() {
        if positions.contains(&s) {
            continue;
        }
        for &p in positions {
            // only what changes the bounds is measured
            let matters = if s < p { s > lo } else { s < hi };
            if !matters || !overlap(shape, &scene.shapes[p]) {
                continue;
            }
            if s < p {
                lo = s;
            } else {
                hi = s;
            }
        }
    }
    let slot = (highest + 1).min(hi);
    (slot > lo).then_some(slot)
}

/// A chain's dashed stroke `shape` fitted over `scene` at `top`: its centerline to the members,
/// then the pattern read two ways (the members' and the image's profile) and settled on the
/// error; as proposed where no window holds it, none where no reading keeps the dashes seen.
fn fitted(
    ctx: &Context,
    scene: &Scene,
    top: usize,
    shape: &Shape,
    mut dashed: Stroke,
    kind: &str,
) -> Option<Shape> {
    let Some(place) = Place::new(ctx, scene, shape, top, MARGIN) else {
        return Some(shape.clone());
    };
    let cost = place.cost(ctx, shape.fill);
    let origin = place.window.origin();
    let mappings = dashed.dash_mappings();
    fit(&cost, &mut dashed, origin, 1.);
    // The fit held the dashes at their parameters; the pattern is read off again
    // where those now lie along the fitted centerline
    let ends = dashed.arc_lengths();
    let spans: Vec<(f32, f32)> = mappings
        .iter()
        .map(|m| {
            let (k0, t0, _) = m[0];
            let (k1, _, t1) = m[m.len() - 1];
            let (a, mut b) = (dashed.arc_at(k0, t0, &ends), dashed.arc_at(k1, t1, &ends));
            if b < a {
                b += ends[ends.len() - 1];
            }
            (a, b)
        })
        .collect();
    let total = ends[ends.len() - 1];
    let lengths: Vec<f32> = spans.iter().map(|(a, b)| b - a).collect();
    let gaps: Vec<f32> = (0..spans.len())
        .filter(|&k| dashed.closed || k + 1 < spans.len())
        .map(|k| {
            let gap = spans[(k + 1) % spans.len()].0 - spans[k].1;
            if gap < 0. { gap + total } else { gap }
        })
        .collect();
    // Two readings of the pattern, each settled on the error: the members' (their medians
    // where alike, see [`pattern`]), and the period and phase of the image's own profile
    // along the fitted centerline, which stay in step over a long chain
    let reach = if dashed.cap == Cap::Butt {
        0.
    } else {
        dashed.width / 2.
    };
    // Lengths along a dashed stroke are told apart to the quantum its pattern is read
    // with ([`pattern`] calls two dashes alike within it); a dot's place is its centre,
    // fitted to its whole outline and held to a quarter of that ([`dotted_stroke`])
    let quantum = if kind == "dots" {
        WIDTH_QUANTUM / 4.
    } else {
        WIDTH_QUANTUM
    };
    // each with the dashes it must draw where they are: the members', or the image's
    let mut readings = vec![(pattern(&lengths, &gaps, spans[0].0), spans.clone())];
    let (x, noise) = profile(ctx, &place, &dashed, shape.fill.color);
    for r in read(&x, noise, dashed.closed, quantum) {
        readings.push((
            dashes_of(&r, reach, shortest(dashed.cap)),
            seen_spans(&r, reach, shortest(dashed.cap)),
        ));
    }
    let mut best: Option<(f64, Stroke)> = None;
    for (reading, seen) in readings {
        let mut trial = dashed.clone();
        trial.dashes = Some(reading);
        let e = tune_pattern(&cost, &mut trial, origin) as f64
            + trial.segments() as f64 * ctx.segment_cost;
        // Each dash and gap drawn where seen, compared by their paint with the caps' reach, so a
        // dot (a dash of no length) lies where its member does (dice's pips); then the ends made
        // whole, which moves the arc lengths
        let ends = trial.arc_lengths();
        let painted = |spans: &[(f32, f32)]| -> Vec<(f32, f32)> {
            spans.iter().map(|&(a, b)| (a - reach, b + reach)).collect()
        };
        let drawn = painted(&trial.dash_spans(&ends));
        let off = off_phase(&drawn, &painted(&seen), ends[ends.len() - 1], trial.closed);
        if off > quantum {
            continue;
        }
        whole_end_dashes(&mut trial);
        if best.as_ref().is_none_or(|b| e < b.0) {
            best = Some((e, trial));
        }
    }
    let (_, dashed) = best?;
    let mut fitted_shape = dashed.shape(shape.fill);
    fitted_shape.smooth = true;
    Some(fitted_shape)
}

/// The chains of dashes, each with the dashed stroke through them: links between the
/// dashes' ends, every end taking its cheapest, and the chains along them.
fn dash_chains(dashes: &[Dash]) -> Vec<(Vec<usize>, Stroke)> {
    // Candidate links between ends of different dashes, with a cost: each end lies on the
    // other's continuation, and the two run toward each other
    let mut links: Vec<(f32, End, End)> = Vec::new();
    for (i, a) in dashes.iter().enumerate() {
        for (j, b) in dashes.iter().enumerate().skip(i + 1) {
            // Caps are not compared (butt and square render alike on short dashes; the chain
            // takes the majority), nor fitted colors (the acceptance judges a chain's single
            // paint); the image's own colors of two leaves are, one paint within the noise
            if let (Some(ca), Some(cb)) = (a.color, b.color)
                && ca.distance(cb) > ALIKE_COLOR
            {
                continue;
            }
            let width = a.width.max(b.width);
            if (a.width - b.width).abs() > (WIDTH_SLACK * width).max(WIDTH_QUANTUM) {
                continue;
            }
            for ea in 0..2 {
                for eb in 0..2 {
                    let (pa, ha, ca) = a.ends[ea];
                    let (pb, hb, cb) = b.ends[eb];
                    let gap = pa.distance(pb);
                    if gap > 3. * a.length.max(b.length) + 2. * width {
                        continue;
                    }
                    if ha.dot(hb) > -(2. * KINK).cos() {
                        continue;
                    }
                    let onward_a = onward(&ca, a.reach + gap);
                    let onward_b = onward(&cb, b.reach + gap);
                    let (miss_a, miss_b) = (onward_a.distance(pb), onward_b.distance(pa));
                    let ok = miss_a <= width / 2. + 1. && miss_b <= width / 2. + 1.;
                    if !ok {
                        continue;
                    }
                    let cost = (miss_a + miss_b) / width + gap / (a.length + b.length + 1.);
                    links.push((cost, (i, ea), (j, eb)));
                }
            }
        }
    }
    links.sort_by(|x, y| x.0.total_cmp(&y.0));
    // Each end takes one link
    let mut partner: Vec<[Option<(usize, usize)>; 2]> = vec![[None, None]; dashes.len()];
    for &(_, (i, ea), (j, eb)) in &links {
        if partner[i][ea].is_none() && partner[j][eb].is_none() {
            partner[i][ea] = Some((j, eb));
            partner[j][eb] = Some((i, ea));
        }
    }
    // Chains: from a dash with a free end, out through the other end, turning each member
    // so the path runs from its end 0 to its end 1; what is left closes on itself
    let mut used = vec![false; dashes.len()];
    let mut chains: Vec<(Vec<(usize, bool)>, bool)> = Vec::new();
    let walk = |start: usize, enter: usize, used: &mut Vec<bool>| -> Vec<(usize, bool)> {
        // `enter` is the end the path comes in by, so it leaves by the other
        let mut chain = Vec::new();
        let (mut i, mut from) = (start, enter);
        loop {
            used[i] = true;
            chain.push((i, from == 1));
            let leave = 1 - from;
            match partner[i][leave] {
                Some((j, e)) if !used[j] => {
                    i = j;
                    from = e;
                }
                _ => break,
            }
        }
        chain
    };
    for start in 0..dashes.len() {
        if used[start] {
            continue;
        }
        let free = (0..2).find(|&e| partner[start][e].is_none());
        if let Some(e) = free {
            let chain = walk(start, e, &mut used);
            chains.push((chain, false));
        }
    }
    for start in 0..dashes.len() {
        if !used[start] {
            let chain = walk(start, 0, &mut used);
            chains.push((chain, true));
        }
    }
    let mut planned = Vec::new();
    for (chain, closed) in chains {
        if chain.len() < MIN_DASHES {
            continue;
        }
        let positions: Vec<usize> = chain.iter().map(|&(i, _)| dashes[i].position).collect();
        let member_strokes: Vec<(&Stroke, bool)> = chain
            .iter()
            .map(|&(i, reversed)| (&dashes[i].stroke, reversed))
            .collect();
        if let Some(dashed) = dashed_stroke(&member_strokes, closed) {
            planned.push((positions, dashed));
        }
    }
    planned
}

/// The slot and fitted shape of a chain's dashed stroke where it costs less error plus
/// segments than its members (at `positions` in the scene, in order along the stroke).
fn try_chain(
    ctx: &Context,
    scene: &Scene,
    positions: &[usize],
    dashed: Stroke,
    kind: &str,
) -> Option<(usize, Shape)> {
    let slot = chain_slot(scene, positions)?;
    let shape = {
        // The members' median color
        let mut channels: Vec<Vec<f32>> = vec![Vec::new(); 3];
        for &p in positions {
            let c = scene.shapes[p].fill.color;
            for (k, v) in [c.x, c.y, c.z].into_iter().enumerate() {
                channels[k].push(v);
            }
        }
        let median = |v: &mut Vec<f32>| {
            v.sort_by(f32::total_cmp);
            v[v.len() / 2]
        };
        let color = glam::Vec3::new(
            median(&mut channels[0]),
            median(&mut channels[1]),
            median(&mut channels[2]),
        );
        let mut shape = dashed.shape(crate::Fill::solid(color));
        shape.smooth = true;
        shape
    };
    let window = Window::around(shape.min, shape.max, MARGIN, ctx.width, ctx.height);
    if window.is_empty() {
        return None;
    }
    let before = {
        let drawn = scene.render(window, 0..scene.shapes.len(), [1.; 3]);
        composite_error(ctx, window, &drawn, &drawn, None, None)
    };
    let cost_before: f64 = positions
        .iter()
        .map(|&p| scene.shapes[p].cost_segments() * ctx.segment_cost + ctx.path_cost)
        .sum();
    let mut trial = scene.clone();
    let mut sorted = positions.to_vec();
    sorted.sort_unstable();
    for &p in sorted.iter().rev() {
        trial.shapes.remove(p);
    }
    trial.change();
    let top = slot - sorted.iter().filter(|&&p| p < slot).count();
    trial.change();
    // Fitted as one stroke in the members' place: the centerline through the members
    // is theirs, but the pattern read off them drifts along the chain
    let shape = fitted(ctx, &trial, top, &shape, dashed, kind)?;
    trial.shapes.insert(top, shape.clone());
    trial.change();
    let after = {
        let drawn = trial.render(window, 0..trial.shapes.len(), [1.; 3]);
        composite_error(ctx, window, &drawn, &drawn, None, None)
    };
    let cost_after = shape.cost_segments() * ctx.segment_cost + ctx.path_cost;
    let taken = after as f64 + cost_after < before as f64 + cost_before;
    taken.then_some((slot, shape))
}
