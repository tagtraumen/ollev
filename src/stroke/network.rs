//! Branching skeletons as networks of strokes cut at their knots.

use super::*;

/// An arm of a network: a chain between junctions, walked from one end to the other.
struct Arm {
    chain: Vec<usize>,
    /// the knot each end touches, if any
    knots: [Option<usize>; 2],
}

impl Arm {
    /// The pixel at end `e` (0 the chain's first).
    fn end(&self, e: usize) -> usize {
        if e == 0 {
            self.chain[0]
        } else {
            self.chain[self.chain.len() - 1]
        }
    }
}

/// The lines of a branching skeleton: cut at its junction knots, with arms that run on
/// through a knot (turning under a kink, aligned within its width) joined, straightest first.
/// Arms no longer than the width there are the knot's bulk and left out.
pub(super) fn network(
    skeleton: &[bool],
    distance: &[f32],
    w: usize,
    h: usize,
    top_left: IVec2,
) -> Vec<Skeleton> {
    let neighbors = |m: &[bool], i: usize| neighbors(m, w, h, i);
    let width_at = |i: usize| 2. * distance[i].sqrt();
    // The junctions: knots of the pixels with three or more neighbours
    let junction: Vec<bool> = (0..w * h)
        .map(|i| skeleton[i] && neighbors(skeleton, i).len() > 2)
        .collect();
    let mut knot = vec![usize::MAX; w * h];
    let mut knots: Vec<Vec<usize>> = Vec::new();
    for i in 0..w * h {
        if !junction[i] || knot[i] != usize::MAX {
            continue;
        }
        let id = knots.len();
        let mut members = vec![i];
        knot[i] = id;
        let mut at = 0;
        while at < members.len() {
            for j in neighbors(&junction, members[at]) {
                if knot[j] == usize::MAX {
                    knot[j] = id;
                    members.push(j);
                }
            }
            at += 1;
        }
        knots.push(members);
    }
    // The arms: the chains between the junctions, each walked from one end to the other
    let arm_mask: Vec<bool> = (0..w * h).map(|i| skeleton[i] && !junction[i]).collect();
    let mut seen = vec![false; w * h];
    let mut arms: Vec<Arm> = Vec::new();
    let mut lines: Vec<Skeleton> = Vec::new();
    let as_skeleton = |chain: &[usize], closed: bool| Skeleton {
        chain: chain
            .iter()
            .map(|&i| IVec2::new((i % w) as i32 - 1, (i / w) as i32 - 1))
            .collect(),
        closed,
        widths: chain.iter().map(|&i| width_at(i)).collect(),
        top_left,
    };
    for i in 0..w * h {
        if !arm_mask[i] || seen[i] {
            continue;
        }
        let mut members = vec![i];
        seen[i] = true;
        let mut at = 0;
        while at < members.len() {
            for j in neighbors(&arm_mask, members[at]) {
                if !seen[j] {
                    seen[j] = true;
                    members.push(j);
                }
            }
            at += 1;
        }
        let farthest = |from: usize| farthest(&arm_mask, w, h, from, None);
        let ends: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&i| neighbors(&arm_mask, i).len() == 1)
            .collect();
        if ends.is_empty() {
            // A ring with no junction on it: a closed stroke of its own among the lines
            let (e1, _) = farthest(members[0]);
            let (e2, parent) = farthest(e1);
            let chain = walk_back(&parent, e2);
            if chain.len() >= MIN_SKELETON {
                lines.push(as_skeleton(&chain, true));
            }
            continue;
        }
        let (e1, _) = farthest(ends[0]);
        let (e2, parent) = farthest(e1);
        let mut chain = walk_back(&parent, e2);
        chain.reverse();
        let touching = |end: usize| neighbors(&junction, end).first().map(|&j| knot[j]);
        let knots = [touching(chain[0]), touching(chain[chain.len() - 1])];
        arms.push(Arm { chain, knots });
    }
    // Each arm's heading into the knot at each end, read over twice the width (past where the
    // thinning hooks toward the knot's corners) and at least twice the corner window, so the
    // pixel staircase of a thin line does not tilt it by nearly a kink
    let at = |i: usize| Vec2::new((i % w) as f32, (i / w) as f32);
    let heading = |arm: &Arm, end: usize| -> Vec2 {
        let n = arm.chain.len();
        let last = arm.end(end);
        let span = ((2. * width_at(last)).ceil() as usize)
            .max(2 * crate::outline::FIT.corner_window)
            .min(n - 1);
        let back = if end == 0 {
            arm.chain[span]
        } else {
            arm.chain[n - 1 - span]
        };
        (at(last) - at(back)).normalize_or_zero()
    };
    // The arms paired across each knot: those that run on through it, the straightest first
    let mut partner: Vec<[Option<(usize, usize)>; 2]> = vec![[None, None]; arms.len()];
    for (id, members) in knots.iter().enumerate() {
        let reach = members.iter().map(|&i| width_at(i)).fold(0f32, f32::max);
        let slots: Vec<(usize, usize)> = arms
            .iter()
            .enumerate()
            .flat_map(|(a, arm)| {
                (0..2)
                    .filter(move |&e| arm.knots[e] == Some(id))
                    .map(move |e| (a, e))
            })
            .collect();
        let mut options: Vec<(f32, usize, usize)> = Vec::new();
        for s in 0..slots.len() {
            for t in s + 1..slots.len() {
                let ((a, ea), (b, eb)) = (slots[s], slots[t]);
                if a == b {
                    continue;
                }
                let (da, db) = (heading(&arms[a], ea), heading(&arms[b], eb));
                // an arm without a heading (a pixel long) runs on through nothing
                let Some(turn) = crate::tracer::angle::turn(da, -db).map(f32::abs) else {
                    continue;
                };
                if turn > KINK {
                    continue;
                }
                let ab = at(arms[b].end(eb)) - at(arms[a].end(ea));
                let offset = da.perp_dot(ab).abs();
                if offset > reach {
                    continue;
                }
                options.push((turn, s, t));
            }
        }
        options.sort_by(|x, y| x.0.total_cmp(&y.0));
        for (_, s, t) in options {
            let ((a, ea), (b, eb)) = (slots[s], slots[t]);
            if partner[a][ea].is_some() || partner[b][eb].is_some() {
                continue;
            }
            partner[a][ea] = Some((b, eb));
            partner[b][eb] = Some((a, ea));
        }
    }
    // The lines: the arms walked through their partners, bridged straight across each knot
    let bridge = |from: usize, to: usize, chain: &mut Vec<usize>| {
        let (x0, y0) = ((from % w) as i32, (from / w) as i32);
        let (x1, y1) = ((to % w) as i32, (to / w) as i32);
        let steps = (x1 - x0).abs().max((y1 - y0).abs());
        for k in 1..steps {
            let t = k as f32 / steps as f32;
            let x = (x0 as f32 + t * (x1 - x0) as f32).round() as usize;
            let y = (y0 as f32 + t * (y1 - y0) as f32).round() as usize;
            let i = y * w + x;
            if chain.last() != Some(&i) {
                chain.push(i);
            }
        }
    };
    let mut used = vec![false; arms.len()];
    let walk = |start: (usize, usize), used: &mut Vec<bool>| -> (Vec<usize>, bool) {
        let mut chain: Vec<usize> = Vec::new();
        let (mut a, mut e) = start;
        loop {
            used[a] = true;
            let arm = &arms[a];
            let oriented: Vec<usize> = if e == 0 {
                arm.chain.clone()
            } else {
                arm.chain.iter().rev().copied().collect()
            };
            if let Some(&last) = chain.last() {
                bridge(last, oriented[0], &mut chain);
            }
            chain.extend(oriented);
            if chain.len() > w * h {
                // longer than the box holds: a pairing that cycles, which the partners'
                // one-to-one matching rules out; kept open rather than walked again
                return (chain, false);
            }
            match partner[a][1 - e] {
                None => return (chain, false),
                Some((b, eb)) => {
                    if b == start.0 && eb == start.1 {
                        // around a loop: back at the start, bridged across the last knot
                        let first = chain[0];
                        bridge(*chain.last().unwrap(), first, &mut chain);
                        return (chain, true);
                    }
                    (a, e) = (b, eb);
                }
            }
        }
    };
    for a in 0..arms.len() {
        for (e, slot) in partner[a].iter().enumerate() {
            if !used[a] && slot.is_none() {
                let (chain, closed) = walk((a, e), &mut used);
                let typical = chain.iter().map(|&i| width_at(i)).fold(0f32, f32::max);
                if chain.len() >= MIN_SKELETON && chain.len() as f32 > typical {
                    lines.push(as_skeleton(&chain, closed));
                }
            }
        }
    }
    for a in 0..arms.len() {
        if !used[a] {
            let (chain, closed) = walk((a, 0), &mut used);
            if chain.len() >= MIN_SKELETON {
                lines.push(as_skeleton(&chain, closed));
            }
        }
    }
    // One pen: the network's lines are drawn at their median width, and a line outside its
    // slack is left to the thick remainder
    let median = |mut v: Vec<f32>| -> f32 {
        v.sort_by(f32::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(0.)
    };
    let pen = median(
        lines
            .iter()
            .flat_map(|l| l.widths.iter().copied())
            .collect(),
    );
    let slack = (WIDTH_SLACK * pen).max(WIDTH_QUANTUM);
    lines.retain(|l| (median(l.widths.clone()) - pen).abs() <= slack);
    lines
}

/// The second moments of points about their mean, and the larger one's sum and axis.
pub(super) struct Spread {
    pub(super) mean: Vec2,
    pub(super) xx: f32,
    pub(super) xy: f32,
    pub(super) yy: f32,
    pub(super) top: f32,
    pub(super) axis: Vec2,
}

pub(super) fn spread(points: &[Vec2]) -> Spread {
    let mean = points.iter().sum::<Vec2>() / points.len() as f32;
    let (mut xx, mut xy, mut yy) = (0f32, 0f32, 0f32);
    for p in points {
        let d = *p - mean;
        xx += d.x * d.x;
        xy += d.x * d.y;
        yy += d.y * d.y;
    }
    let top = 0.5 * (xx + yy) + (0.25 * (xx - yy).powi(2) + xy * xy).sqrt();
    let axis = if xy.abs() > 1e-6 {
        Vec2::new(xy, top - xx).normalize()
    } else if xx >= yy {
        Vec2::X
    } else {
        Vec2::Y
    };
    Spread {
        mean,
        xx,
        xy,
        yy,
        top,
        axis,
    }
}

/// The chain of a short band along its principal axis through its centroid, as far as the
/// mask reaches either way, with the distance transform's widths; `None` where the band is
/// round (no axis) or shorter than a few pixels.
pub(super) fn axis_chain(
    mask: &[bool],
    distance: &[f32],
    w: usize,
    h: usize,
    top_left: IVec2,
) -> Option<Skeleton> {
    let inside: Vec<Vec2> = (0..w * h)
        .filter(|&i| mask[i])
        .map(|i| Vec2::new((i % w) as f32, (i / w) as f32))
        .collect();
    if inside.len() < 4 {
        return None;
    }
    let Spread {
        mean,
        xx,
        xy,
        yy,
        top,
        axis,
    } = spread(&inside);
    let bottom = 0.5 * (xx + yy) - (0.25 * (xx - yy).powi(2) + xy * xy).sqrt();
    // No axis where the spread is the same every way (a dot)
    if top < 1.5 * bottom.max(1e-3) {
        return None;
    }
    let covered = |p: Vec2| {
        let (x, y) = (p.x.round() as isize, p.y.round() as isize);
        x >= 0
            && y >= 0
            && (x as usize) < w
            && (y as usize) < h
            && mask[y as usize * w + x as usize]
    };
    let reach = |direction: Vec2| -> usize {
        let mut s = 0;
        while covered(mean + direction * (s as f32 + 1.)) {
            s += 1;
        }
        s
    };
    let (back, forward) = (reach(-axis), reach(axis));
    let mut chain: Vec<IVec2> = Vec::new();
    for k in -(back as i32)..=(forward as i32) {
        let p = mean + axis * k as f32;
        let pixel = IVec2::new(p.x.round() as i32, p.y.round() as i32);
        if chain.last() != Some(&pixel) {
            chain.push(pixel);
        }
    }
    if chain.len() < 3 {
        return None;
    }
    let widths = chain
        .iter()
        .map(|p| 2. * distance[p.y as usize * w + p.x as usize].sqrt())
        .collect();
    Some(Skeleton {
        chain: chain.iter().map(|p| *p - 1).collect(),
        closed: false,
        widths,
        top_left,
    })
}
