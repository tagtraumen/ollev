//! The draw order: a shape occluded by an earlier sibling goes below it.

use super::*;

/// Pixels around an outline sample within which the other shape has to lie for the sample to
/// be on their shared boundary.
pub(super) const JUNCTION_REACH: i32 = 1;
/// Mean distance in pixels within which a shape's outline, bridged smoothly across the stretch
/// it shares with a sibling, still follows that stretch: the shape owns the boundary.
pub(super) const OWNS: f64 = 1.;
/// Mean distance by which a shape's bridge has to miss the shared stretch more than the
/// sibling's for the stretch to be a notch cut into the shape (a square on a circle of radius
/// 10 cuts a chord missed by 2 px).
pub(super) const NOTCH_DEPTH: f64 = 1.;
/// And by what factor it has to miss more.
pub(super) const NOTCH_RATIO: f64 = 2.;
/// Pixels along a sibling's outline on either side of a junction over which its turning is
/// measured, to tell it running smoothly through from having a corner there.
pub(super) const SMOOTH_REACH: f32 = 2.;

/// For each planned shape, the earlier sibling that occludes it, if any: of two siblings
/// sharing a boundary, the one whose outline bridged across it still follows it owns it, and
/// the other goes below (a circle under a square; two blocks sharing a side keep their order).
pub(super) fn occluders(
    hierarchy: &Hierarchy,
    plan: &[Drawn],
    outlines: &HashMap<u32, Vec<Vec<Cubic>>>,
    width: usize,
    height: usize,
) -> Vec<Option<usize>> {
    use crate::tracer::fit::eval;
    const SAMPLES: usize = 8;
    let near = |p: Vec2, other: u32| {
        let (cx, cy) = (p.x.floor() as i32, p.y.floor() as i32);
        (cy - JUNCTION_REACH..=cy + JUNCTION_REACH).any(|y| {
            (cx - JUNCTION_REACH..=cx + JUNCTION_REACH).any(|x| {
                x >= 0
                    && y >= 0
                    && (x as usize) < width
                    && (y as usize) < height
                    && hierarchy.contains(other, y as usize * width + x as usize)
            })
        })
    };
    // How far the outline of `node`, bridged across what it shares with `other`, lies from the
    // shared stretch: the mean over the stretch's samples, with the ends of the stretches, or
    // `None` without a shared stretch; with `split`, the stretch is read between samples
    let miss = |node: u32, other: u32, split: bool| -> Option<(f64, Vec<Vec2>)> {
        let (mut total, mut count) = (0f64, 0usize);
        let mut ends = Vec::new();
        for cubics in &outlines[&node] {
            let pieces;
            let cubics = if split {
                pieces = split_where(cubics, &|p| near(p, other));
                &pieces
            } else {
                cubics
            };
            let n = cubics.len();
            let shared: Vec<bool> = cubics
                .iter()
                .map(|c| (1..SAMPLES).all(|i| near(eval(c, i as f32 / SAMPLES as f32), other)))
                .collect();
            if !shared.iter().any(|&s| s) || shared.iter().all(|&s| s) {
                continue;
            }
            for i in 0..n {
                if shared[i] && !shared[(i + n - 1) % n] {
                    ends.push(cubics[i][0]);
                }
                if shared[i] && !shared[(i + 1) % n] {
                    ends.push(cubics[i][3]);
                }
            }
            let bridged = crate::complete::bridge(cubics, &|p| near(p, other));
            let bridge_samples: Vec<Vec2> = bridged
                .iter()
                .flat_map(|c| {
                    (0..SAMPLES * 4).map(move |i| eval(c, i as f32 / (SAMPLES * 4) as f32))
                })
                .collect();
            for (c, _) in cubics.iter().zip(&shared).filter(|(_, s)| **s) {
                for i in 0..SAMPLES {
                    let p = eval(c, i as f32 / SAMPLES as f32);
                    let d = bridge_samples
                        .iter()
                        .map(|q| p.distance(*q))
                        .fold(f32::INFINITY, f32::min);
                    total += d as f64;
                    count += 1;
                }
            }
        }
        (count > 0).then(|| (total / count as f64, ends))
    };
    // Whether the outline of `node` runs smoothly through the point `p` on it: it turns by less
    // than the corner angle between a reach before and after its nearest point
    let smooth_through = |node: u32, p: Vec2| -> bool {
        let (mut best, mut at) = (f32::INFINITY, None);
        for (l, cubics) in outlines[&node].iter().enumerate() {
            for (k, c) in cubics.iter().enumerate() {
                for i in 0..SAMPLES * 2 {
                    let t = i as f32 / (SAMPLES * 2) as f32;
                    let d = eval(c, t).distance(p);
                    if d < best {
                        best = d;
                        at = Some((l, k, t));
                    }
                }
            }
        }
        let Some((l, k, t)) = at.filter(|_| best <= JUNCTION_REACH as f32 + 1.) else {
            return false;
        };
        let cubics = &outlines[&node][l];
        let n = cubics.len();
        // Walk the outline from the nearest point by about the reach, either way
        let walk = |direction: f32| {
            let (mut k, mut t) = (k, t);
            let mut from = eval(&cubics[k], t);
            let mut walked = 0f32;
            let step = direction / (SAMPLES * 2) as f32;
            while walked < SMOOTH_REACH {
                t += step;
                if t > 1. {
                    k = (k + 1) % n;
                    t -= 1.;
                } else if t < 0. {
                    k = (k + n - 1) % n;
                    t += 1.;
                }
                let to = eval(&cubics[k], t);
                walked += from.distance(to);
                from = to;
            }
            from
        };
        let (before, after) = (walk(-1.), walk(1.));
        let here = eval(&cubics[k], t);
        // a walk that stands still (no direction) is not smooth there
        crate::tracer::angle::turn(here - before, after - here)
            .is_some_and(|t| t.abs() < crate::outline::CORNER_ANGLE)
    };

    let mut under = vec![None; plan.len()];
    for k in 1..plan.len() {
        let mut best: Option<(usize, f64)> = None;
        for s in 1..k {
            if plan[s].parent != plan[k].parent {
                continue;
            }
            let (a, b) = (
                &hierarchy.nodes[plan[k].node as usize],
                &hierarchy.nodes[plan[s].node as usize],
            );
            if a.min.x > b.max.x || b.min.x > a.max.x || a.min.y > b.max.y || b.min.y > a.max.y {
                continue;
            }
            let Some((own, ends)) = miss(plan[k].node, plan[s].node, false) else {
                continue;
            };
            // The sibling either has a shared stretch of its own, or its outline runs through
            // both ends of ours and misses by nothing, unless its outline read between samples
            // (a coarse fit shares no whole cubic) shows it does not own ours (the fox's ear,
            // four cubics around its inner ear)
            let theirs = match miss(plan[s].node, plan[k].node, false) {
                Some((theirs, _)) => theirs,
                None if ends.iter().all(|&p| smooth_through(plan[s].node, p)) => {
                    let m = miss(plan[s].node, plan[k].node, true).map_or(0., |m| m.0);
                    if m < OWNS { 0. } else { m }
                }
                None => continue,
            };
            if crate::diagnostics::debug() {
                eprintln!(
                    "order: node {} misses {own:.2} px across node {}, which misses {theirs:.2}",
                    plan[k].node, plan[s].node
                );
            }
            if theirs < OWNS
                && own - theirs > NOTCH_DEPTH
                && own > NOTCH_RATIO * theirs
                && best.is_none_or(|(_, m)| own > m)
            {
                best = Some((s, own));
            }
        }
        under[k] = best.map(|(s, _)| s);
    }
    under
}

/// The loop with each cubic split where the outline enters or leaves the pixels `inside`,
/// between two of its samples, so a stretch an outline shares is whole cubics however coarse
/// its fit (the fox's ear, four cubics around the inner ear's notch).
fn split_where(cubics: &[Cubic], inside: &impl Fn(Vec2) -> bool) -> Vec<Cubic> {
    use crate::tracer::fit::eval;
    const SAMPLES: usize = 16;
    let mut out = Vec::with_capacity(cubics.len());
    for c in cubics {
        let at: Vec<bool> = (0..=SAMPLES)
            .map(|i| inside(eval(c, i as f32 / SAMPLES as f32)))
            .collect();
        let cuts: Vec<f32> = (0..SAMPLES)
            .filter(|&i| at[i] != at[i + 1])
            .map(|i| (i as f32 + 0.5) / SAMPLES as f32)
            .collect();
        let (mut rest, mut from) = (*c, 0.);
        for t in cuts {
            let (a, b) = split(&rest, (t - from) / (1. - from));
            out.push(a);
            (rest, from) = (b, t);
        }
        out.push(rest);
    }
    out
}

/// A cubic split at `t` by de Casteljau.
fn split(c: &Cubic, t: f32) -> (Cubic, Cubic) {
    let ab = c[0].lerp(c[1], t);
    let bc = c[1].lerp(c[2], t);
    let cd = c[2].lerp(c[3], t);
    let abc = ab.lerp(bc, t);
    let bcd = bc.lerp(cd, t);
    let m = abc.lerp(bcd, t);
    ([c[0], ab, abc, m], [m, bcd, cd, c[3]])
}

/// The order to draw the plan in: its own order, except that a shape occluded by an earlier
/// sibling (`under`) comes right before that sibling, recursively.
pub(super) fn draw_order(under: &[Option<usize>]) -> Vec<usize> {
    fn visit(k: usize, under: &[Option<usize>], placed: &mut [bool], order: &mut Vec<usize>) {
        if placed[k] {
            return;
        }
        placed[k] = true;
        for j in k + 1..under.len() {
            if under[j] == Some(k) {
                visit(j, under, placed, order);
            }
        }
        order.push(k);
    }
    let mut order = Vec::with_capacity(under.len());
    let mut placed = vec![false; under.len()];
    for k in 0..under.len() {
        visit(k, under, &mut placed, &mut order);
    }
    order
}
