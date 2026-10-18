//! Completion of a layer's beneath across the holes the opaque parts above it leave
//! ([`super::Candidate::holes`]): the pieces meeting a hole continue their outlines across it
//! by their tangents, and the hole takes each piece's colour on its side of those outlines.

use crate::HashMap;
use std::collections::VecDeque;

use super::raster::{Conn, DVec3, SHOW_THROUGH, deep_in, dilate, length_px};

pub type V2 = [f64; 2];
/// A crack between two pieces: its midpoint and its two pixels (the lower label's first).
type Crack = (V2, usize, usize);
/// A crack's end at a hole: its pair, the corner, the direction along the crack, its two
/// pixels and the hole.
type RawEnd = ((u32, u32), V2, V2, usize, usize, u32);
/// One side of a crack: the corner, the direction past it and the two pixels it runs between.
type Side = (V2, V2, [(i64, i64); 2]);

fn sub(a: V2, b: V2) -> V2 {
    [a[0] - b[0], a[1] - b[1]]
}
fn add(a: V2, b: V2) -> V2 {
    [a[0] + b[0], a[1] + b[1]]
}
fn mul(a: V2, s: f64) -> V2 {
    [a[0] * s, a[1] * s]
}
fn dot(a: V2, b: V2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn cross(a: V2, b: V2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn norm(a: V2) -> f64 {
    dot(a, a).sqrt()
}

/// A boundary between two pieces where it meets a hole.
#[derive(Clone, Debug)]
pub struct End {
    pub p: V2,
    /// into the hole, unit
    pub d: V2,
    pub hole: u32,
    /// the colours left and right of `d` (left: cross(d, x - p) > 0)
    pub left: DVec3,
    pub right: DVec3,
}

/// What the completion did, for the log.
#[derive(Default, Debug)]
pub struct Report {
    pub holes: usize,
    pub ends: usize,
    pub pairs: usize,
    pub barrier_px: usize,
    pub filled_px: usize,
    /// hole pixels in thick parts, filled flat
    pub flat_px: usize,
    pub ends_at: Vec<End>,
}

fn step(a: DVec3, b: DVec3) -> f64 {
    (a - b).abs().max_element()
}

/// 4-connected components of the hole.
fn components(hole: &[bool], w: usize, h: usize) -> (Vec<u32>, usize) {
    let mut comp = vec![u32::MAX; w * h];
    let mut n = 0;
    let mut stack = Vec::new();
    for s in 0..w * h {
        if !hole[s] || comp[s] != u32::MAX {
            continue;
        }
        comp[s] = n;
        stack.push(s);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            let mut go = |j: usize| {
                if hole[j] && comp[j] == u32::MAX {
                    comp[j] = n;
                    stack.push(j);
                }
            };
            if x > 0 {
                go(i - 1);
            }
            if x + 1 < w {
                go(i + 1);
            }
            if y > 0 {
                go(i - w);
            }
            if y + 1 < h {
                go(i + w);
            }
        }
        n += 1;
    }
    (comp, n as usize)
}

/// The contour ends at the holes: every crack between two pieces outside the holes whose
/// line runs on into a hole, its step past the show-through, with its tangent from the cracks
/// within the reach and the colours either side; and the count of hole components.
pub fn contour_ends(
    img: &[DVec3],
    label: &[u32],
    hole: &[bool],
    w: usize,
    h: usize,
) -> (Vec<End>, usize) {
    let (comp, holes) = components(hole, w, h);
    let at = |x: i64, y: i64| -> Option<usize> {
        (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h)
            .then(|| y as usize * w + x as usize)
    };
    let in_hole = |x: i64, y: i64| at(x, y).filter(|&i| hole[i]);
    // every crack between two pieces outside the hole, by its pair: its midpoint and the two
    // pixels (the first of the pair's lower label)
    let mut cracks: HashMap<(u32, u32), Vec<Crack>> = HashMap::default();
    // candidate ends: (pair, corner point, initial direction, pixel of the lower label, other)
    let mut raw: Vec<RawEnd> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if hole[i] {
                continue;
            }
            // right neighbour: a vertical crack at x + 1 from y to y + 1
            for (dx, dy) in [(1usize, 0usize), (0, 1)] {
                let (x2, y2) = (x + dx, y + dy);
                if x2 >= w || y2 >= h {
                    continue;
                }
                let j = y2 * w + x2;
                if hole[j] || label[i] == label[j] || step(img[i], img[j]) <= SHOW_THROUGH {
                    continue;
                }
                let (a, b, pa, pb) = if label[i] < label[j] {
                    (label[i], label[j], i, j)
                } else {
                    (label[j], label[i], j, i)
                };
                let (xf, yf) = (x as f64, y as f64);
                let mid = if dx == 1 {
                    [xf + 1.0, yf + 0.5]
                } else {
                    [xf + 0.5, yf + 1.0]
                };
                cracks.entry((a, b)).or_default().push((mid, pa, pb));
                // the crack's line runs on past both of its corners
                let (xi, yi) = (x as i64, y as i64);
                let sides: [Side; 2] = if dx == 1 {
                    [
                        (
                            [xf + 1.0, yf],
                            [0.0, -1.0],
                            [(xi, yi - 1), (xi + 1, yi - 1)],
                        ),
                        (
                            [xf + 1.0, yf + 1.0],
                            [0.0, 1.0],
                            [(xi, yi + 1), (xi + 1, yi + 1)],
                        ),
                    ]
                } else {
                    [
                        (
                            [xf, yf + 1.0],
                            [-1.0, 0.0],
                            [(xi - 1, yi), (xi - 1, yi + 1)],
                        ),
                        (
                            [xf + 1.0, yf + 1.0],
                            [1.0, 0.0],
                            [(xi + 1, yi), (xi + 1, yi + 1)],
                        ),
                    ]
                };
                for (corner, d, past) in sides {
                    if let Some(k) = past.iter().find_map(|&(px, py)| in_hole(px, py)) {
                        raw.push(((a, b), corner, d, pa, pb, comp[k]));
                    }
                }
            }
        }
    }
    // tangents, and one end per pair and hole within the anti-aliasing scale
    let reach = length_px(w, 4.0) as f64;
    let mut ends: Vec<End> = Vec::new();
    // one end per pair, hole, place and direction: a crack's two corners give the same end
    // twice, but at a convex corner of a piece two cracks end at one point in two directions,
    // and both are contours ending there
    let mut seen: Vec<((u32, u32), u32, V2, V2)> = Vec::new();
    for (pair, p, d0, _, _, hole_id) in raw {
        if seen.iter().any(|(q, c, s, sd)| {
            *q == pair && *c == hole_id && norm(sub(*s, p)) <= 1.5 && dot(*sd, d0) > 0.0
        }) {
            continue;
        }
        seen.push((pair, hole_id, p, d0));
        let near: Vec<&Crack> = cracks[&pair]
            .iter()
            .filter(|(m, pa, pb)| {
                // the stretch this end belongs to: the pair's cracks within the reach that run the
                // way the end's own crack runs, so the two stretches meeting at a convex corner do
                // not pool into one diagonal tangent
                norm(sub(*m, p)) <= reach && ((pa.abs_diff(*pb) == 1) == (d0[0] == 0.0))
            })
            .collect();
        let mut c = [0.0, 0.0];
        for (m, _, _) in &near {
            c = add(c, *m);
        }
        c = mul(c, 1.0 / near.len() as f64);
        let chord = sub(p, c);
        let d = if norm(chord) >= reach / 4.0 {
            mul(chord, 1.0 / norm(chord))
        } else if near.len() <= 2 {
            d0
        } else {
            // the contour passes the hole by on both sides: no end
            continue;
        };
        if dot(d, d0) < 0.0 {
            continue;
        }
        // the colours either side, averaged over the near cracks
        let (mut sa, mut sb, mut side) = (DVec3::ZERO, DVec3::ZERO, 0.0);
        for (m, pa, pb) in &near {
            sa += img[*pa];
            sb += img[*pb];
            let pa_xy = [(*pa % w) as f64 + 0.5, (*pa / w) as f64 + 0.5];
            side += cross(d, sub(pa_xy, *m)).signum();
        }
        let (ca, cb) = (sa / near.len() as f64, sb / near.len() as f64);
        let (left, right) = if side > 0.0 { (ca, cb) } else { (cb, ca) };
        ends.push(End {
            p,
            d,
            hole: hole_id,
            left,
            right,
        });
    }
    (ends, holes)
}

/// The relatable pairs among `ends` on one hole each (Kellman and Shipley: a turn of at most
/// a right angle or collinear within tolerance, the colours matching crosswise), joined by
/// the tangent cubic staying in the holes, best first: the curves and the barrier they draw.
pub fn pair_ends(ends: &[End], hole: &[bool], w: usize, h: usize) -> (Vec<[V2; 4]>, Vec<bool>) {
    let at = |x: i64, y: i64| -> Option<usize> {
        (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h)
            .then(|| y as usize * w + x as usize)
    };
    let in_hole = |x: i64, y: i64| at(x, y).filter(|&i| hole[i]);
    let near_hole = |q: V2| -> bool {
        let (x, y) = (q[0].floor() as i64, q[1].floor() as i64);
        (-1..=1).any(|dy| (-1..=1).any(|dx| in_hole(x + dx, y + dy).is_some()))
    };
    let curve = |a: &End, b: &End| -> [V2; 4] {
        let chord = norm(sub(b.p, a.p));
        let arrive = mul(b.d, -1.0);
        let turn = (cross(a.d, arrive).atan2(dot(a.d, arrive))).abs();
        let handle = if turn < 1e-3 {
            chord / 3.0
        } else {
            let radius = chord / (2.0 * (turn / 2.0).sin());
            (4.0 / 3.0 * (turn / 4.0).tan() * radius).min(chord)
        };
        [
            a.p,
            add(a.p, mul(a.d, handle)),
            add(b.p, mul(b.d, handle)),
            b.p,
        ]
    };
    let eval = |c: &[V2; 4], t: f64| -> V2 {
        let u = 1.0 - t;
        let mut q = mul(c[0], u * u * u);
        q = add(q, mul(c[1], 3.0 * u * u * t));
        q = add(q, mul(c[2], 3.0 * u * t * t));
        add(q, mul(c[3], t * t * t))
    };
    let mut options: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..ends.len() {
        for j in i + 1..ends.len() {
            let (a, b) = (&ends[i], &ends[j]);
            if a.hole != b.hole {
                continue;
            }
            let step_a = step(a.left, a.right);
            let step_b = step(b.left, b.right);
            let tol = 0.5 * step_a.min(step_b);
            if step(a.left, b.right) > tol || step(a.right, b.left) > tol {
                continue;
            }
            let arrive = mul(b.d, -1.0);
            let turn = (cross(a.d, arrive).atan2(dot(a.d, arrive))).abs();
            if turn > std::f64::consts::FRAC_PI_2 {
                continue;
            }
            let ab = sub(b.p, a.p);
            let dist = norm(ab);
            if dist < 1e-9 {
                continue;
            }
            let offset = cross(a.d, ab).abs();
            let det = cross(a.d, b.d);
            let meet = det.abs() > 1e-6 && {
                let s = cross(ab, b.d) / det;
                let t = cross(ab, a.d) / det;
                s >= 0.0 && t >= 0.0
            };
            let collinear = dot(a.d, ab) > 0.0 && offset <= 1.5 + 0.2 * dist;
            if !meet && !collinear {
                continue;
            }
            let c = curve(a, b);
            let samples = (dist * 2.0).ceil().max(4.0) as usize;
            let pts: Vec<V2> = (0..=samples)
                .map(|k| eval(&c, k as f64 / samples as f64))
                .collect();
            if !pts.iter().all(|q| near_hole(*q)) {
                continue;
            }
            // The pairs are taken by the curve's bending energy (Euler's elastica: arc length plus
            // the squared curvature, weighed by the reach squared to share a unit), shortest and
            // straightest first, so a near pair across a strand wins over a far one across the hair
            let mut length = 0.0;
            let mut bending = 0.0;
            let mut last: Option<V2> = None;
            for k in 1..pts.len() {
                let seg = sub(pts[k], pts[k - 1]);
                let ds = norm(seg);
                if ds < 1e-9 {
                    continue;
                }
                let t = mul(seg, 1.0 / ds);
                if let Some(prev) = last {
                    let dphi = cross(prev, t).atan2(dot(prev, t));
                    bending += dphi * dphi / ds;
                }
                length += ds;
                last = Some(t);
            }
            let reach = length_px(w, 4.0) as f64;
            options.push((length + reach * reach * bending, i, j));
        }
    }
    options.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut used = vec![false; ends.len()];
    let mut barrier = vec![false; w * h];
    let mut curves = Vec::new();
    for (_, i, j) in options {
        if used[i] || used[j] {
            continue;
        }
        used[i] = true;
        used[j] = true;
        let c = curve(&ends[i], &ends[j]);
        let samples = (norm(sub(c[3], c[0])) * 8.0).ceil().max(8.0) as usize;
        for k in 0..=samples {
            let q = eval(&c, k as f64 / samples as f64);
            let (x, y) = (q[0].floor() as i64, q[1].floor() as i64);
            if let Some(m) = in_hole(x, y) {
                barrier[m] = true;
            } else {
                // the curve may run along the hole's rim just outside it: the barrier is then the
                // hole pixels beside it, or the curve cuts nothing
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    if let Some(m) = in_hole(x + dx, y + dy) {
                        barrier[m] = true;
                    }
                }
            }
        }
        curves.push(c);
    }
    (curves, barrier)
}

/// Fills the holes from their rims by a wave that does not cross barriers, each hole pixel
/// taking the colour its wave's piece shows near the rim; thick parts ([`thin_part`]) filled
/// flat. `blend` pixels (a hole's anti-aliased rim) are filled alike and lend no colour.
pub fn complete_holes_with_blend(
    img: &[DVec3],
    label: &[u32],
    hole_core: &[bool],
    blend: &[bool],
    w: usize,
    h: usize,
) -> (Vec<DVec3>, Vec<bool>, Report) {
    let hole_all: Vec<bool> = hole_core.iter().zip(blend).map(|(&a, &b)| a || b).collect();
    let hole = &hole_all[..];
    let (ends, holes) = contour_ends(img, label, hole, w, h);
    let (curves, barrier) = pair_ends(&ends, hole, w, h);
    let at = |x: i64, y: i64| -> Option<usize> {
        (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h)
            .then(|| y as usize * w + x as usize)
    };
    let reach = length_px(w, 4.0) as f64;
    let mut report = Report {
        holes,
        ends: ends.len(),
        pairs: curves.len(),
        ..Report::default()
    };
    report.barrier_px = barrier.iter().filter(|b| **b).count();
    // the wave from the rim, each rim pixel carrying its piece's mean colour within the reach,
    // not its own: a rim pixel at a junction or on an anti-aliased edge sends streaks of blends
    let r = reach as i64;
    let piece_colour = |s: usize| -> DVec3 {
        let (x, y) = ((s % w) as i64, (s / w) as i64);
        let (mut sum, mut n) = (DVec3::ZERO, 0.0);
        for yy in (y - r).max(0)..=(y + r).min(h as i64 - 1) {
            for xx in (x - r).max(0)..=(x + r).min(w as i64 - 1) {
                let j = yy as usize * w + xx as usize;
                if !hole[j] && label[j] == label[s] {
                    sum += img[j];
                    n += 1.0;
                }
            }
        }
        sum / n
    };
    let mut out = img.to_vec();
    // the piece each hole pixel's wave came from
    let mut src: Vec<u32> = label.to_vec();
    let mut done: Vec<bool> = hole.iter().map(|h| !h).collect();
    let mut queue = VecDeque::new();
    for i in 0..w * h {
        if !hole[i] || barrier[i] {
            continue;
        }
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        if let Some(s) = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            .iter()
            .filter_map(|&(a, b)| at(a, b))
            .find(|&j| !hole[j])
        {
            out[i] = piece_colour(s);
            src[i] = label[s];
            done[i] = true;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        for (a, b) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
            if let Some(j) = at(a, b)
                && !done[j]
                && !barrier[j]
            {
                out[j] = out[i];
                src[j] = src[i];
                done[j] = true;
                queue.push_back(j);
            }
        }
    }
    // the barriers and anything they closed off, from any filled neighbour
    loop {
        let mut changed = false;
        for i in 0..w * h {
            if done[i] {
                continue;
            }
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            if let Some(j) = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                .iter()
                .filter_map(|&(a, b)| at(a, b))
                .find(|&j| done[j])
            {
                out[i] = out[j];
                src[i] = src[j];
                done[i] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // each thick part flat, in the colour of the piece whose wave reached most of it, without
    // the streaks where waves meet; a thick part a barrier crosses keeps the wave, as the
    // contours on its sides say what lies under it (glassware's stand rod)
    let thin = thin_part(hole_core, w, h);
    let thick: Vec<bool> = (0..w * h).map(|i| hole_core[i] && !thin[i]).collect();
    let (part, parts) = components(&thick, w, h);
    let mut votes: Vec<HashMap<u32, (usize, DVec3)>> = vec![HashMap::default(); parts];
    let mut crossed = vec![false; parts];
    for i in 0..w * h {
        if thick[i] {
            let e = votes[part[i] as usize].entry(src[i]).or_default();
            e.0 += 1;
            e.1 += out[i];
            if barrier[i] {
                crossed[part[i] as usize] = true;
            }
        }
    }
    let flat: Vec<Option<DVec3>> = votes
        .iter()
        .zip(&crossed)
        .map(|(v, crossed)| {
            if *crossed {
                return None;
            }
            v.values()
                .max_by_key(|(n, _)| *n)
                .map(|(n, sum)| *sum / *n as f64)
        })
        .collect();
    for i in 0..w * h {
        if thick[i]
            && let Some(c) = flat[part[i] as usize]
        {
            out[i] = c;
            report.flat_px += 1;
        }
    }
    report.filled_px = hole.iter().filter(|h| **h).count();
    report.ends_at = ends;
    (out, barrier, report)
}

/// The widest a line is, in units: half the side of the square whose opening keeps a hole's
/// thick part (a strand of hair, glassware's 3 px outlines: up to six units across).
pub const THIN: f64 = 3.0;

/// The thin parts of `hole`: what an opening by a square of half side [`THIN`] units takes
/// off. The rest is its thick part (the hair's mass).
pub fn thin_part(hole: &[bool], w: usize, h: usize) -> Vec<bool> {
    let r = length_px(w, THIN);
    let thick = dilate(&deep_in(hole, w, h, r), w, h, Conn::Eight, r);
    hole.iter().zip(&thick).map(|(&a, &b)| a && !b).collect()
}
