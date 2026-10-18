//! The missing lines as chains: ridges followed, chains linked across gaps, their smoothness.

use super::*;

/// The missing lines followed from each chain end along its tangent past where the
/// hysteresis drops them, through gaps of the reach and through what the render draws, until
/// another chain, the border or the ridge is lost (so a faint fold is one stroke, not shards).
pub(super) fn follow_ridges(img: &Image, lines: &[bool]) -> Vec<bool> {
    let (w, h) = (img.width, img.height);
    let (strength, _) = colour_lines_at(img, unit_of(w));
    let mut out = lines.to_vec();
    let sk = skeletonize(lines, w, h);
    let reach = REACH as i64;
    let back = 3 * (REACH + 1);
    let at = |q: DVec2| -> Option<usize> {
        let (x, y) = (q.x.round(), q.y.round());
        (x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h)
            .then(|| y as usize * w + x as usize)
    };
    // whether pixel i is a ridge of the colour line test across the walk's normal `n`, above
    // the noise (the walk's own direction, not the test's: on a faint line the direction
    // that finds the strongest pair wanders, the line's does not)
    let ridge = |i: usize, n: DVec2| -> bool {
        if strength[i] <= NOISE {
            return false;
        }
        let q = DVec2::new((i % w) as f64, (i / w) as f64);
        [1.0, -1.0]
            .iter()
            .all(|s| at(q + s * n).is_none_or(|j| strength[i] >= strength[j]))
    };
    for c in trace_chains(&sk, w, h, 2 * REACH + 1) {
        let p: Vec<DVec2> = c
            .iter()
            .map(|&(y, x)| DVec2::new(x as f64, y as f64))
            .collect();
        if p.len() < 2 {
            continue;
        }
        let own: crate::HashSet<usize> = c.iter().map(|&(y, x)| y * w + x).collect();
        for reversed in [false, true] {
            let mut trail: Vec<DVec2> = if reversed {
                p.iter().rev().copied().collect()
            } else {
                p.clone()
            };
            let mut gap = 0i64;
            let mut pending: Vec<usize> = Vec::new();
            let mut q = trail[trail.len() - 1];
            for _ in 0..w + h {
                let k = back.min(trail.len() - 1);
                let t = (trail[trail.len() - 1] - trail[trail.len() - 1 - k]).normalize_or_zero();
                let ahead = q + t;
                if t == DVec2::ZERO || at(ahead).is_none() {
                    break;
                }
                let n = DVec2::new(-t.y, t.x);
                let best = (-reach..=reach)
                    .filter_map(|o| {
                        let r = ahead + o as f64 * n;
                        at(r).filter(|&i| ridge(i, n)).map(|i| (i, r.round()))
                    })
                    .max_by(|a, b| strength[a.0].total_cmp(&strength[b.0]));
                match best {
                    Some((i, r)) => {
                        gap = 0;
                        q = r;
                        trail.push(r);
                        // a bridged gap joins the line only once the ridge is found again
                        for j in pending.drain(..) {
                            out[j] = true;
                        }
                        let joined = lines[i] && !own.contains(&i);
                        out[i] = true;
                        if joined {
                            break;
                        }
                    }
                    None => {
                        gap += 1;
                        if gap > reach {
                            break;
                        }
                        q = ahead;
                        pending.extend(at(ahead));
                    }
                }
            }
        }
    }
    out
}

/// A thin mask's chains: skeletonized, cut at junctions, spurs shorter than the anti-aliased
/// band's width (a pixel beyond the reach either side) dropped, a dash a few pixels long
/// kept; fitted as the lines module's chains of the kind given.
pub(super) fn chains_of(img: &Image, mask: &[bool], ridge: bool) -> Vec<Chain> {
    let (w, h) = (img.width, img.height);
    let sk = skeletonize(mask, w, h);
    let l = img.luminance();
    let mut out = Vec::new();
    for c in trace_chains(&sk, w, h, 2 * REACH + 1) {
        let mut p: Vec<DVec2> = c
            .iter()
            .map(|&(y, x)| DVec2::new(x as f64, y as f64))
            .collect();
        // the skeleton's staircase is the grid's
        if p.len() >= 3 {
            let q = p.clone();
            for i in 1..p.len() - 1 {
                p[i] = 0.25 * q[i - 1] + 0.5 * q[i] + 0.25 * q[i + 1];
            }
        }
        out.push(make_chain(img, &l, p, ridge));
    }
    out
}

/// A linked line: its members in order, each as (chain, reversed), whether it closes on
/// itself (a ring), and whether it is a ridge by most of its length.
#[derive(Clone, Debug)]
pub(super) struct Linked {
    pub(super) members: Vec<(usize, bool)>,
    pub(super) closed: bool,
    pub(super) ridge: bool,
}

pub(super) fn linked(chains: &[Chain]) -> Vec<Linked> {
    let n = chains.len();
    let mut partner = vec![None; 2 * n];
    for l in find_links(chains, LinkMode::Angles) {
        let (a, b) = (2 * l.a.0 + l.a.1, 2 * l.b.0 + l.b.1);
        partner[a] = Some(b);
        partner[b] = Some(a);
    }
    let mut used = vec![false; n];
    let mut out = Vec::new();
    let walk = |c0: usize, e0: usize, used: &mut Vec<bool>| {
        let (mut c, mut e) = (c0, e0);
        let mut members = Vec::new();
        let closed = loop {
            used[c] = true;
            // entered by end e: walked from e to the other end, reversed when e is the last
            members.push((c, e == 1));
            match partner[2 * c + 1 - e] {
                Some(q) if !used[q / 2] => {
                    c = q / 2;
                    e = q % 2;
                }
                Some(q) => break q / 2 == c0,
                None => break false,
            }
        };
        let (mut ridge, mut all) = (0.0, 0.0);
        for &(m, _) in &members {
            all += chains[m].length;
            if chains[m].kind == Kind::Ridge {
                ridge += chains[m].length;
            }
        }
        Linked {
            members,
            closed,
            ridge: ridge > 0.5 * all,
        }
    };
    for c in 0..n {
        if !used[c] && (partner[2 * c].is_none() || partner[2 * c + 1].is_none()) {
            let e = if partner[2 * c].is_none() { 0 } else { 1 };
            out.push(walk(c, e, &mut used));
        }
    }
    for c in 0..n {
        if !used[c] {
            out.push(walk(c, 0, &mut used));
        }
    }
    out
}

/// A member's points in SVG units in the line's order.
pub(super) fn points(ch: &Chain, reversed: bool) -> Vec<DVec2> {
    let mut p: Vec<DVec2> = ch.p.iter().map(|q| *q + DVec2::splat(0.5)).collect();
    if reversed {
        p.reverse();
    }
    p
}

/// A linked line's members' points and the whole line through its gaps (closed for a ring).
pub(super) fn polylines(chains: &[Chain], line: &Linked) -> (Vec<Vec<DVec2>>, Vec<DVec2>) {
    let parts: Vec<Vec<DVec2>> = line
        .members
        .iter()
        .map(|&(c, r)| points(&chains[c], r))
        .collect();
    let mut through: Vec<DVec2> = parts.iter().flatten().copied().collect();
    let ring = line.closed
        || (parts.len() == 1
            && through.len() > 3
            && through[0].distance(through[through.len() - 1]) < 2.0);
    if ring {
        through.push(through[0]);
    }
    (parts, through)
}

pub(super) fn arc_length(p: &[DVec2]) -> f64 {
    p.windows(2).map(|q| q[0].distance(q[1])).sum()
}

/// A line's simplicity: its length per cubic fitted to the lines module's tolerance, over
/// the shortest chain that module takes, at most 1 (a jagged texture chain scores low).
pub fn smoothness(p: &[DVec2], width: usize) -> f64 {
    let u = unit_of(width);
    let shortest = length_px(width, MIN_CHAIN as f64) as f64;
    if p.len() < 2 {
        return 0.0;
    }
    let cubics = fit_cubics(p, FIT_TOL * u).len().max(1) as f64;
    (arc_length(p) / cubics / shortest).min(1.0)
}

/// The target's weak edges: suppressed gradient ridges at least 3 NOISE of step, held by a
/// strong edge or not; a texture's lines are mostly below the strong level, so its density
/// shows only here.
pub fn weak_edges(img: &Image) -> Vec<bool> {
    let (w, h) = (img.width, img.height);
    let (mag, theta) = di_zenzo(img, SIGMA);
    let unit = 1.0 / (SIGMA * (2.0 * std::f64::consts::PI).sqrt());
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let (dy, dx) = (theta[i].sin(), theta[i].cos());
            mag[i] >= 3.0 * NOISE * unit
                && mag[i] >= map1(&mag, w, h, x + dx, y + dy)
                && mag[i] >= map1(&mag, w, h, x - dx, y - dy)
        })
        .collect()
}

/// How alone a line runs: per point, the weak edges a ray across meets on each side within
/// four side offsets; one per side is the line's own shape, each beyond it a texture's next
/// line (lena's straw weave), and the point counts by 1 / (1 + extra / 2), averaged.
pub fn isolation(p: &[DVec2], field: &EdgeField, weak: &[bool]) -> f64 {
    let (w, h) = (field.width, field.height);
    let side = SIDE_OFFSET * field.unit;
    let nrm = normals(p);
    let (mut sum, mut all) = (0.0, 0usize);
    for (q, n) in p.iter().zip(&nrm) {
        let mut extra = 0usize;
        for s in [1.0, -1.0] {
            // edges met along the ray, each counted where the ray enters it
            let (mut met, mut inside) = (0usize, false);
            let mut d = side;
            while d <= 4.0 * side {
                let r = *q + s * d * *n;
                if r.x < 0.0 || r.y < 0.0 || r.x as usize >= w || r.y as usize >= h {
                    break;
                }
                let on = weak[r.y as usize * w + r.x as usize];
                if on && !inside {
                    met += 1;
                }
                inside = on;
                d += 0.5;
            }
            extra += met.saturating_sub(1);
        }
        sum += 1.0 / (1.0 + 0.5 * extra as f64);
        all += 1;
    }
    if all == 0 { 1.0 } else { sum / all as f64 }
}

/// The discount on a line's edge credit: its simplicity times how alone it runs (a short
/// fragment is discounted by its length, not zeroed).
pub(super) fn discount(p: &[DVec2], field: &EdgeField, weak: &[bool]) -> f64 {
    smoothness(p, field.width) * isolation(p, field, weak)
}
