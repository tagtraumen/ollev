//! Hypotheses of relations `o = k b + m` from the leaves' contact map: the plane pass (a Hough
//! over contact points) and the equation graph (slope pairs and crossings scored as cuts),
//! merged into one list; and the shading pass, shadows and highlights as their own class.

use crate::{HashMap, HashSet};

use glam::{IVec3, Vec3};

use super::leaves::{Leaves, Plane, slope_fits};
use super::{NOISE, in_gamut};

mod graph;
mod shading;

pub use graph::*;
pub use shading::*;

/// A relation `o = k b + m` with `m = (1 - k) c`, and its evidence.
#[derive(Clone, Copy, Debug)]
pub struct Hypothesis {
    pub k: f32,
    pub c: Vec3,
    pub score: f32,
}

/// The values of `k` the Hough passes vote at: 0.05 to 0.95 in steps of 0.025.
fn k_grid() -> impl Iterator<Item = f32> {
    (0..=36).map(|i| (0.05 + 0.025 * i as f64) as f32)
}

/// A color quantized to steps of `levels` 8-bit levels, for the Hough cells and bins.
fn bin(c: Vec3, levels: f32) -> IVec3 {
    (c * 255. / levels).round().as_ivec3()
}

/// Sorts by score, highest first, stably: ties are frequent and the order decides them.
fn by_score(hyps: &mut [Hypothesis]) {
    hyps.sort_by(|a, b| b.score.total_cmp(&a.score));
}

/// The hypotheses in order, each dropped if one kept before it lies within `dk` in `k` and
/// `dc` in every channel of `c`.
fn merge_near(hyps: &[Hypothesis], dk: f32, dc: f32) -> Vec<Hypothesis> {
    let mut out: Vec<Hypothesis> = Vec::new();
    for h in hyps {
        if out
            .iter()
            .any(|e| (h.k - e.k).abs() <= dk && (h.c - e.c).abs().max_element() <= dc)
        {
            continue;
        }
        out.push(*h);
    }
    out
}

/// A contact point as the Hough passes see it.
struct Vote {
    b: Vec3,
    o: Vec3,
    w: f32,
    inside: Plane,
    outside: Plane,
}

/// The contact points that can be evidence: of a pair whose inside has an interior, where
/// the two sides differ.
fn votes(lv: &Leaves) -> Vec<Vote> {
    let mut out = Vec::new();
    for p in &lv.pairs {
        if !lv.interior[p.x as usize] {
            continue;
        }
        for q in &lv.points[p.points.clone()] {
            if (q.o - q.b).abs().max_element() > 2. * NOISE {
                out.push(Vote {
                    b: q.b,
                    o: q.o,
                    w: q.w,
                    inside: lv.plane[p.x as usize],
                    outside: lv.plane[p.y as usize],
                });
            }
        }
    }
    out
}

/// The adjacency Hough over contact points (b, o), each fitting only where the slopes do: per
/// `k` a point votes for `m = o - k b` in cells of 6 levels, a cell needing two distinct
/// backgrounds. Scored by contact, near duplicates merged, then refined.
pub fn plane_pass(lv: &Leaves) -> Vec<Hypothesis> {
    let votes = votes(lv);
    if votes.len() < 2 {
        return Vec::new();
    }
    let mut hyps = Vec::new();
    for k in k_grid() {
        // The cells in the order of their first vote: the sum of their m, their contact,
        // their distinct backgrounds and their votes
        let mut cells: HashMap<IVec3, usize> = HashMap::default();
        let mut members: Vec<(Vec3, f32, HashSet<IVec3>, usize)> = Vec::new();
        for v in &votes {
            let m = v.o - k * v.b;
            let c = m / (1. - k);
            if c.min_element() < -NOISE
                || c.max_element() > 1. + NOISE
                || !slope_fits(&v.inside, &v.outside, k)
            {
                continue;
            }
            let next = members.len();
            let cell = *cells.entry(bin(m, 6.)).or_insert(next);
            if cell == next {
                members.push((Vec3::ZERO, 0., HashSet::default(), 0));
            }
            let entry = &mut members[cell];
            entry.0 += m;
            entry.1 += v.w;
            entry.2.insert(bin(v.b, 4.));
            entry.3 += 1;
        }
        if members.iter().map(|e| e.3).sum::<usize>() < 2 {
            continue;
        }
        for (sum, w, backgrounds, n) in &members {
            if backgrounds.len() >= 2 {
                hyps.push(Hypothesis {
                    k,
                    c: *sum / *n as f32 / (1. - k),
                    score: *w,
                });
            }
        }
    }
    by_score(&mut hyps);
    let merged = merge_near(&hyps, 0.05, 12. / 255.);
    refine(&merged, &votes, 2. * NOISE)
}

/// Rounds of the refinement, and how many of a list it refines.
const REFINE_ROUNDS: usize = 2;
const REFINE_TOP: usize = 40;

/// Each hypothesis refitted by weighted least squares over the points that fit it (colors
/// and slopes), a couple of times, then scored by their contact times `k`, near duplicates
/// merged.
fn refine(hyps: &[Hypothesis], votes: &[Vote], tol: f32) -> Vec<Hypothesis> {
    let fits = |v: &Vote, k: f32, m: Vec3| {
        (v.o - (k * v.b + m)).abs().max_element() <= tol && slope_fits(&v.inside, &v.outside, k)
    };
    let mut out = Vec::new();
    for h in hyps.iter().take(REFINE_TOP) {
        let (mut k, mut m) = (h.k, (1. - h.k) * h.c);
        for _ in 0..REFINE_ROUNDS {
            let fit: Vec<&Vote> = votes.iter().filter(|v| fits(v, k, m)).collect();
            if fit.len() < 2 {
                break;
            }
            let w: f32 = fit.iter().map(|v| v.w).sum();
            let mb = fit.iter().map(|v| v.w * v.b).sum::<Vec3>() / w;
            let mo = fit.iter().map(|v| v.w * v.o).sum::<Vec3>() / w;
            let den: f32 = fit.iter().map(|v| v.w * (v.b - mb).length_squared()).sum();
            if den <= 0. {
                break;
            }
            let k2 = fit
                .iter()
                .map(|v| v.w * (v.b - mb).dot(v.o - mo))
                .sum::<f32>()
                / den;
            if !(0.05..=0.95).contains(&k2) {
                break;
            }
            k = k2;
            m = mo - k2 * mb;
        }
        let fit: Vec<&Vote> = votes.iter().filter(|v| fits(v, k, m)).collect();
        let c = m / (1. - k);
        if !in_gamut(c) {
            continue;
        }
        // The points that fit have to show two backgrounds and two images: one image color
        // over anything is an opaque shape
        let bs: HashSet<IVec3> = fit.iter().map(|v| bin(v.b, 16.)).collect();
        let os: HashSet<IVec3> = fit.iter().map(|v| bin(v.o, 16.)).collect();
        if bs.len() < 2 || os.len() < 2 {
            continue;
        }
        // A color is a background or an image of the relation, not both (a chain of bands
        // each a little darker than the last is shading, see the equation graph)
        if bs.intersection(&os).count() as f32 > 0.3 * bs.len().min(os.len()) as f32 {
            continue;
        }
        // The evidence for translucency is the show-through, k times the edge: a near-opaque
        // relation (k = 0.05, c the tile color) fits every boundary of a tile and says nothing
        out.push(Hypothesis {
            k,
            c,
            score: fit.iter().map(|v| v.w).sum::<f32>() * k,
        });
    }
    by_score(&mut out);
    merge_near(&out, 0.04, 10. / 255.)
}
