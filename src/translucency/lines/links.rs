//! Links between chain ends across gaps.

use glam::DVec2;

use super::{
    Bezier, Chain, FIT_TOL, JUNCTION_GAP, Kind, WIDTH_QUANTUM, WIDTH_SLACK, along_circle,
    bezier_ls, chord_params, curvature, dot, end_tangents, errors, fit, hyp, kink, max_of, newton,
};
use crate::translucency::raster::linspace;

pub(super) fn widths_alike(a: &Chain, b: &Chain) -> bool {
    if a.kind == Kind::Ridge && b.kind == Kind::Ridge {
        let wmax = a.width.max(b.width);
        return (a.width - b.width).abs() <= (WIDTH_SLACK * wmax).max(WIDTH_QUANTUM * a.unit);
    }
    true
}

/// The cost of end `eb` of chain b continuing end `ea` of chain a: each end continued along
/// its curve lands within half a width plus one of the other (plus its heading's uncertainty),
/// headings within the kink tolerance, curvatures and ridge widths alike.
pub fn link_test(a: &Chain, ea: usize, b: &Chain, eb: usize) -> Option<f64> {
    let u = a.unit;
    // the kinds may differ: under a tint a contour's soft dark rim falls below the ridge
    // test while the step stays
    if !widths_alike(a, b) {
        return None;
    }
    let (x, y) = (a.ends[ea], b.ends[eb]);
    let g = hyp(y.p - x.p);
    // a curve is continued no farther than its end cubic's own span, but always across the
    // break suppression leaves at a crossing
    if g > (JUNCTION_GAP * a.unit).max(x.span.min(y.span)) {
        return None;
    }
    let width = a.width.max(b.width);
    let mut cost = 0.0;
    for (s, q) in [(x, y), (y, x)] {
        let (on, h_on) = along_circle(s.p, s.heading, s.k, g);
        let unc = 2.0 * FIT_TOL * u / s.span.max(1.0);
        let miss = hyp(on - q.p);
        if miss > width / 2.0 + a.unit + g * unc {
            return None;
        }
        // q's outward heading reversed is the way the line runs on at q; within twice the
        // kink tolerance (each end's direction is uncertain by a kink)
        let turn = dot(h_on, -q.heading).clamp(-1.0, 1.0).acos();
        if turn > 2.0 * kink() + unc {
            return None;
        }
        cost += miss / (width / 2.0 + a.unit);
    }
    // curvature: along the path from a into b, a's outward curvature is the path's and b's
    // outward curvature is the reversed path's, so the path's at b is -kb
    let lshort = x.span.min(y.span).max(1.0);
    if (x.k - (-y.k)).abs() * lshort.powi(2) / 8.0 > FIT_TOL * u {
        return None;
    }
    Some(cost + g / (a.length + b.length + 1.0))
}

/// One cubic through the points with the end tangents fixed, Newton-improved, and its
/// largest miss.
pub(super) fn one_cubic(p: &[DVec2], tol: f64) -> (Bezier, f64) {
    let (t1, t2) = end_tangents(p, 3.0 * tol);
    let mut c = fit(p, t1, t2, 1e9, 0)[0];
    let mut u = chord_params(p);
    for _ in 0..8 {
        u = newton(&c, p, &u);
        c = bezier_ls(p, &u, t1, t2);
    }
    (c, max_of(&errors(&c, p, &u)))
}

/// The link test by a joint fit: the parts' pixels near the gap lie on one cubic within the
/// fit tolerance that bends at most twice the parts' worst (ends can differ in angle yet join
/// smoothly). The cost prefers the smoother join.
pub fn link_test_joint(a: &Chain, ea: usize, b: &Chain, eb: usize) -> Option<f64> {
    const REACH_MULT: f64 = 2.0;
    if !widths_alike(a, b) {
        return None;
    }
    let (x, y) = (a.ends[ea], b.ends[eb]);
    let g = hyp(y.p - x.p);
    if g > (JUNCTION_GAP * a.unit).max(REACH_MULT * x.span.max(y.span)) {
        return None;
    }
    if g > 1e-6 {
        let chord = (y.p - x.p) / g;
        if dot(chord, x.heading) < 0.0 || dot(-chord, y.heading) < 0.0 {
            return None;
        }
    }
    let pa: Vec<DVec2> = if ea == 1 {
        a.p.clone()
    } else {
        a.p.iter().rev().copied().collect()
    };
    let pb: Vec<DVec2> = if eb == 1 {
        b.p.iter().rev().copied().collect()
    } else {
        b.p.clone()
    };
    let n = 4usize.max((3.0 * g) as usize);
    let mut p: Vec<DVec2> = pa[pa.len().saturating_sub(n)..].to_vec();
    p.extend_from_slice(&pb[..n.min(pb.len())]);
    let (c, miss) = one_cubic(&p, FIT_TOL * a.unit);
    if miss > FIT_TOL * a.unit {
        return None;
    }
    let ts = linspace(0.0, 1.0, 33);
    let curv = |c: &Bezier| {
        ts.iter()
            .map(|&t| curvature(c, t).abs())
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let kj = curv(&c);
    let kp = a
        .cubics
        .iter()
        .chain(&b.cubics)
        .map(curv)
        .fold(f64::NEG_INFINITY, f64::max);
    if kj > 2.0 * kp.max(1.0 / (4.0 * g + 1e-9)) {
        return None;
    }
    Some(miss / (FIT_TOL * a.unit) + g / (a.length + b.length + 1.0))
}

/// Which link test pairs the ends: the angles and curvatures of the ends, or the joint fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkMode {
    Angles,
    Joint,
}

impl LinkMode {
    pub(super) fn test(self, a: &Chain, ea: usize, b: &Chain, eb: usize) -> Option<f64> {
        match self {
            LinkMode::Angles => link_test(a, ea, b, eb),
            LinkMode::Joint => link_test_joint(a, ea, b, eb),
        }
    }
}

/// A link between end `a.1` of chain `a.0` and end `b.1` of chain `b.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Link {
    pub cost: f64,
    pub a: (usize, usize),
    pub b: (usize, usize),
}

/// Every pair of ends of different chains within `reach` that passes the test, cheapest
/// first (pairs in index order before the sort, which keeps ties so).
pub(super) fn candidate_links(chains: &[Chain], reach: f64, mode: LinkMode) -> Vec<Link> {
    let ends: Vec<DVec2> = chains
        .iter()
        .flat_map(|c| [c.ends[0].p, c.ends[1].p])
        .collect();
    let r2 = reach * reach;
    let mut order: Vec<usize> = (0..ends.len()).collect();
    order.sort_by(|&a, &b| ends[a].x.total_cmp(&ends[b].x));
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (oi, &u) in order.iter().enumerate() {
        for &v in &order[oi + 1..] {
            if ends[v].x - ends[u].x > reach {
                break;
            }
            let d = ends[v] - ends[u];
            if d.x * d.x + d.y * d.y <= r2 {
                pairs.push((u.min(v), u.max(v)));
            }
        }
    }
    pairs.sort_unstable();
    let mut cand: Vec<Link> = Vec::new();
    for (u, v) in pairs {
        let ((i, ea), (j, eb)) = ((u / 2, u % 2), (v / 2, v % 2));
        if i == j {
            continue;
        }
        if let Some(cost) = mode.test(&chains[i], ea, &chains[j], eb) {
            cand.push(Link {
                cost,
                a: (i, ea),
                b: (j, eb),
            });
        }
    }
    cand.sort_by(|a, b| a.cost.total_cmp(&b.cost));
    cand
}

pub(super) fn reach_of(chains: &[Chain]) -> f64 {
    chains
        .iter()
        .map(|c| c.ends[0].span.max(c.ends[1].span))
        .fold(
            JUNCTION_GAP * chains.first().map_or(1.0, |c| c.unit),
            f64::max,
        )
}

/// Candidate links between ends of different chains, and the cover: cheapest first, each
/// end takes one.
pub fn find_links(chains: &[Chain], mode: LinkMode) -> Vec<Link> {
    if chains.is_empty() {
        return Vec::new();
    }
    let mut reach = reach_of(chains);
    if mode == LinkMode::Joint {
        reach *= 2.0;
    }
    let mut taken = vec![false; 2 * chains.len()];
    let mut links = Vec::new();
    for l in candidate_links(chains, reach, mode) {
        let (x, y) = (l.a.0 * 2 + l.a.1, l.b.0 * 2 + l.b.1);
        if taken[x] || taken[y] {
            continue;
        }
        taken[x] = true;
        taken[y] = true;
        links.push(l);
    }
    links
}

/// [`find_links`]' candidates without the cover, by the angles test, cheapest first: a short
/// T join can take an end from the smooth continuation, so the proposal counts every one.
pub fn all_links(chains: &[Chain]) -> Vec<Link> {
    if chains.is_empty() {
        return Vec::new();
    }
    candidate_links(chains, reach_of(chains), LinkMode::Angles)
}
