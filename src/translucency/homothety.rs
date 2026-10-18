//! Whether a region holds the image of an outside material under a layer: under opacity k and
//! colour c a material's colours b are seen as `o = k b + (1 - k) c`, one homothety, fitted by
//! one-to-one matching of density-resampled clouds alternated with least squares for k.

use std::fmt::Write;

use crate::HashMap;

use super::inner::pin_grid;
use super::materials::{Palette, clear_of};
use super::raster::{Conn, DVec3, Image, NOISE, SHOW_THROUGH, cheb, dilate, length_px, median};

/// The count of colours each side is resampled to for the coarse fit.
pub const SAMPLES: usize = 48;
/// The counts a pair that passes at [`SAMPLES`] is refitted at, in turn, each from the
/// relation the one before found (all the colours where a side has fewer).
pub const REFINED: [usize; 2] = [128, 256];
/// The starts of k.
const STARTS: [f64; 4] = [0.2, 0.4, 0.6, 0.8];
/// Rounds of assignment and least squares from one start.
const ROUNDS: usize = 6;
/// The residual an image is fitted to: two colours each known to the noise (the matched
/// inside colour and the image of its outside colour), 4 levels.
pub const RESIDUAL: f64 = 2.0 * NOISE;

/// A relation `o = k b + m` fitted to an inside cloud over an outside one, and its residual:
/// the median Chebyshev distance of the matched pairs (0..1, as the colours).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    pub k: f64,
    pub m: DVec3,
    pub residual: f64,
}

impl Fit {
    /// The layer's colour, `m / (1 - k)`.
    pub fn c(&self) -> DVec3 {
        self.m / (1.0 - self.k)
    }
}

/// `n` colours of `x` (weights `w`) by pixel density: systematic sampling over the colours
/// in code order, a colour drawn as often as its pixels hold shares. Exact colours, as a
/// bucketed mean would move each by up to a level, the residual's scale.
pub fn sample(x: &[DVec3], w: &[f64], n: usize) -> Vec<DVec3> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| {
        let (p, q) = (x[a], x[b]);
        p.x.total_cmp(&q.x)
            .then(p.y.total_cmp(&q.y))
            .then(p.z.total_cmp(&q.z))
    });
    let total: f64 = w.iter().sum();
    let mut out = Vec::with_capacity(n);
    let (mut at, mut cum) = (0usize, 0.0f64);
    for draw in 0..n {
        let pos = (draw as f64 + 0.5) * total / n as f64;
        // the first colour whose cumulative pixels pass the draw's position
        while at + 1 < order.len() && cum + w[order[at]] <= pos {
            cum += w[order[at]];
            at += 1;
        }
        out.push(x[order[at]]);
    }
    out
}

/// The assignment of least total cost on the `n` by `m` matrix `cost` (row major, `n <= m`):
/// per row its column, each column taken at most once. The Hungarian method with
/// potentials, O(n^2 m).
pub fn assign(cost: &[f64], n: usize, m: usize) -> Vec<usize> {
    debug_assert!(n <= m && cost.len() == n * m);
    let inf = f64::INFINITY;
    let (mut u, mut v) = (vec![0.0f64; n + 1], vec![0.0f64; m + 1]);
    // p[j]: the row (1-based) assigned to column j; column 0 the free one
    let (mut p, mut way) = (vec![0usize; m + 1], vec![0usize; m + 1]);
    let (mut minv, mut used) = (vec![inf; m + 1], vec![false; m + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        minv.fill(inf);
        used.fill(false);
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let row = &cost[(i0 - 1) * m..i0 * m];
            let (mut delta, mut j1) = (inf, 0usize);
            for j in 1..=m {
                if !used[j] {
                    let cur = row[j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = vec![0usize; n];
    for j in 1..=m {
        if p[j] != 0 {
            out[p[j] - 1] = j - 1;
        }
    }
    out
}

fn mean(x: &[DVec3]) -> DVec3 {
    x.iter().copied().sum::<DVec3>() / x.len().max(1) as f64
}

/// The mean of `x` weighted by `w`.
fn weighted_mean(x: &[DVec3], w: &[f64]) -> DVec3 {
    let total: f64 = w.iter().sum();
    x.iter().zip(w).map(|(x, w)| *x * *w).sum::<DVec3>() / total.max(1e-12)
}

/// The least squares relation over matched pairs `(o, b)`, one k for the three channels:
/// `k = cov(o, b) / var(b)` summed over channels, `m = mean(o) - k mean(b)`; k clamped
/// inside (0, 1) by a step of the grid's scale.
fn least_squares(o: &[DVec3], b: &[DVec3]) -> (f64, DVec3) {
    let (mo, mb) = (mean(o), mean(b));
    let (mut num, mut den) = (0.0, 0.0);
    for (o, b) in o.iter().zip(b) {
        let (d, e) = (*b - mb, *o - mo);
        num += d.dot(e);
        den += d.dot(d);
    }
    let k = (num / den.max(1e-12)).clamp(0.02, 0.98);
    (k, mo - k * mb)
}

/// The matched pairs of one fit, kept for a joint fit over several pairs.
#[derive(Clone, Debug, Default)]
pub struct Matched {
    pub o: Vec<DVec3>,
    pub b: Vec<DVec3>,
}

/// One start of the alternation on the samples `o` (inside) and `b` (outside), equal in
/// count: per round the assignment under the current relation, its residual, then the least
/// squares over the matched pairs; the round of least residual and its pairs.
fn icp(o: &[DVec3], b: &[DVec3], k0: f64) -> (Fit, Matched) {
    let (n, nb) = (o.len(), b.len());
    let mut k = k0;
    let mut m = mean(o) - k * mean(b);
    let mut best = (
        Fit {
            k,
            m,
            residual: f64::INFINITY,
        },
        Matched::default(),
    );
    let mut cost = vec![0.0f64; n * nb];
    let mut d = vec![0.0f64; n];
    for _ in 0..ROUNDS {
        let img: Vec<DVec3> = b.iter().map(|&b| k * b + m).collect();
        for (i, &oi) in o.iter().enumerate() {
            for (j, &q) in img.iter().enumerate() {
                cost[i * nb + j] = cheb(oi, q);
            }
        }
        let a = assign(&cost, n, nb);
        for (i, &j) in a.iter().enumerate() {
            d[i] = cost[i * nb + j];
        }
        let residual = median(&mut d);
        let bo: Vec<DVec3> = o.to_vec();
        let bb: Vec<DVec3> = a.iter().map(|&j| b[j]).collect();
        let (k2, m2) = least_squares(&bo, &bb);
        if residual < best.0.residual {
            best = (Fit { k, m, residual }, Matched { o: bo, b: bb });
        }
        // settled: k within a ten-thousandth, m within a twentieth of a level
        if (k2 - k).abs() < 1e-4 && (m2 - m).abs().max_element() < 0.05 / 255.0 {
            break;
        }
        (k, m) = (k2, m2);
    }
    best
}

/// The fit `from` refitted at each count of [`REFINED`] in turn from the relation before,
/// counts capped at the larger side's colours and a count no larger than the last skipped.
/// Per count taken its fit; the last fit's matched pairs (the given ones if none is taken).
pub fn refine(
    inside: &[DVec3],
    w_in: &[f64],
    outside: &[DVec3],
    w_out: &[f64],
    from: (Fit, Matched),
) -> (Vec<(usize, Fit)>, Matched) {
    let cap = inside.len().max(outside.len());
    let (mut at, mut k) = (SAMPLES, from.0.k);
    let mut steps = Vec::new();
    let mut matched = from.1;
    for &n in &REFINED {
        let n = n.min(cap);
        if n <= at {
            continue;
        }
        let (f, mt) = icp(&sample(inside, w_in, n), &sample(outside, w_out, n), k);
        (k, matched, at) = (f.k, mt, n);
        steps.push((n, f));
    }
    (steps, matched)
}

/// What a pair's fit is read against: the inside cluster's spread (its widest channel, 5th
/// to 95th percentile by pixels) and the outside material's mean.
#[derive(Clone, Copy, Debug)]
pub struct Clouds {
    pub spread: f64,
    pub bmean: DVec3,
}

/// The `p`-th percentile of one channel of `x` by the pixels `w`.
fn weighted_percentile(x: &[DVec3], w: &[f64], ch: usize, p: f64) -> f64 {
    let mut v: Vec<(f64, f64)> = x.iter().zip(w).map(|(x, w)| (x[ch], *w)).collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = w.iter().sum();
    let mut cum = 0.0;
    for (x, w) in &v {
        cum += w;
        if cum >= p / 100.0 * total {
            return *x;
        }
    }
    v.last().map_or(0.0, |v| v.0)
}

impl Clouds {
    pub fn of(inside: &[DVec3], w_in: &[f64], outside: &[DVec3], w_out: &[f64]) -> Self {
        let spread = (0..3)
            .map(|ch| {
                weighted_percentile(inside, w_in, ch, 95.0)
                    - weighted_percentile(inside, w_in, ch, 5.0)
            })
            .fold(0.0, f64::max);
        let bmean = weighted_mean(outside, w_out);
        Clouds { spread, bmean }
    }
}

/// Whether a fit reads as a layer the pin can take: [`in_range`] and the residual within
/// [`RESIDUAL`].
pub fn is_layer(f: &Fit, clouds: &Clouds) -> bool {
    in_range(f, clouds) && f.residual <= RESIDUAL
}

/// Whether a fit is a relation the pin could take: k in [`pin_grid`]'s range, c in gamut loosely
/// (a white over the layer pushes it past), the inside cloud spread by more than the
/// show-through (or k stays open), and the material moved past it (seen bare is no evidence).
pub fn in_range(f: &Fit, clouds: &Clouds) -> bool {
    let g = pin_grid();
    let (lo, hi) = (g[0], g[g.len() - 1]);
    if !(lo - 1e-9..=hi + 1e-9).contains(&f.k) {
        return false;
    }
    let c = f.c();
    let margin = SHOW_THROUGH / (1.0 - f.k);
    c.min_element() >= -margin
        && c.max_element() <= 1.0 + margin
        && clouds.spread > SHOW_THROUGH
        && cheb(f.k * clouds.bmean + f.m, clouds.bmean) > SHOW_THROUGH
}

/// An inside cluster matched to an outside material.
#[derive(Clone, Debug)]
pub struct Pair {
    /// the inside cluster (its index in the inside palette) and its pixels
    pub cluster: usize,
    pub px: f64,
    /// the outside material (its rank among the outside palette's materials)
    pub material: usize,
    /// whether the material was read as the ring sees it ([`Ring`])
    pub in_ring: bool,
    /// the outside material's mean
    pub bmean: DVec3,
    /// the fit at [`SAMPLES`] (the best start in range), and per refined count its fit
    pub coarse: Fit,
    pub steps: Vec<(usize, Fit)>,
    /// the fit the gate reads: the last refined one, or the coarse one where none was taken
    pub fit: Fit,
    pub matched: Matched,
}

/// The relation a region carries: the group of passing pairs whose relations agree within
/// their residuals, of the most distinct outside materials, then the most inside pixels,
/// fitted jointly over all its matched pairs.
#[derive(Clone, Debug)]
pub struct Relation {
    pub k: f64,
    pub m: DVec3,
    /// the joint fit's residual over the group's matched pairs
    pub residual: f64,
    /// the pairs of the group (indices into [`Gate::pairs`]), distinct outside materials
    pub pairs: Vec<usize>,
    pub materials: usize,
    pub px: f64,
    /// the spread of the members' relations about the joint one: the largest move of a
    /// member's material image, and the range of their k. A drift past the show-through is a
    /// beneath not flat under the layer.
    pub drift: f64,
    pub k_range: (f64, f64),
}

impl Relation {
    pub fn c(&self) -> DVec3 {
        self.m / (1.0 - self.k)
    }
}

/// The gate on one region's palettes.
#[derive(Clone, Debug, Default)]
pub struct Gate {
    /// every pair that passes at [`SAMPLES`], refitted; `passing` those whose refined fit
    /// passes too
    pub pairs: Vec<Pair>,
    pub passing: Vec<usize>,
    pub relation: Option<Relation>,
    /// the pairs tried, and the closest fit in range of all of them (the reason a region is
    /// rejected) and the closest at any k
    pub tried: usize,
    /// the inside clusters not fitted, their spread within the show-through
    pub flat: usize,
    pub closest_in_range: Option<(usize, usize, f64, Fit)>,
    pub closest: Option<(usize, usize, f64, Fit)>,
}

impl Gate {
    /// Whether some inside cluster is the image of some outside material: one pair is enough,
    /// as one material's image can carry a whole layer.
    pub fn passes(&self) -> bool {
        !self.passing.is_empty()
    }

    /// One line per pair passing at [`SAMPLES`] (its residual, k and c at every count), the
    /// relation and its group, or the closest fits where none passes; levels throughout.
    pub fn describe(&self) -> String {
        let lv = |f: &Fit| {
            format!(
                "res {:.1} k {:.2} c {:?}",
                f.residual * 255.0,
                f.k,
                (f.c() * 255.0).round().to_array()
            )
        };
        let mut s = format!(
            "{} pairs tried ({} flat clusters left out)",
            self.tried, self.flat
        );
        if self.pairs.is_empty() {
            if let Some((cl, mat, px, f)) = &self.closest_in_range {
                let _ = write!(
                    s,
                    "; none passes, closest in range cl{cl} ({px:.0} px) on m{mat}: {}",
                    lv(f)
                );
            } else {
                s += "; none passes, no fit in range";
            }
            if let Some((cl, mat, px, f)) = &self.closest {
                let _ = write!(
                    s,
                    "; closest at any k cl{cl} ({px:.0} px) on m{mat}: {}",
                    lv(f)
                );
            }
            return s;
        }
        for (i, p) in self.pairs.iter().enumerate() {
            let steps: Vec<String> = p
                .steps
                .iter()
                .map(|(n, f)| format!("{n}: {}", lv(f)))
                .collect();
            let _ = write!(
                s,
                "\n    cl{} ({:.0} px) on m{}{}: 48: {}{}{}{}",
                p.cluster,
                p.px,
                p.material,
                if p.in_ring { " (ring)" } else { "" },
                lv(&p.coarse),
                if steps.is_empty() { "" } else { "; " },
                steps.join("; "),
                if self.passing.contains(&i) {
                    ""
                } else {
                    ", refit fails"
                }
            );
        }
        if let Some(r) = &self.relation {
            let _ = write!(
                s,
                "\n    relation k {:.3} c {:?} res {:.1}: {} materials, {:.0} px, pairs {:?}, drift \
                 {:.1} levels, k {:.2} to {:.2}{}",
                r.k,
                (r.c() * 255.0).round().to_array(),
                r.residual * 255.0,
                r.materials,
                r.px,
                r.pairs,
                r.drift * 255.0,
                r.k_range.0,
                r.k_range.1,
                if r.drift > SHOW_THROUGH {
                    " (past the show-through)"
                } else {
                    ""
                }
            );
        }
        s
    }
}

/// How far two pairs' relations lie apart where they are read: the largest move, over both
/// pairs' outside material means, of the image under one relation against the other.
fn apart(a: &Pair, b: &Pair) -> f64 {
    [a.bmean, b.bmean]
        .iter()
        .map(|&x| cheb(a.fit.k * x + a.fit.m, b.fit.k * x + b.fit.m))
        .fold(0.0, f64::max)
}

/// The clusters that carry a region: its materials (at least `min_material_px` pixels),
/// largest first, until they hold half its pixels. A small cluster beside them is no
/// evidence: a few colours of small clusters fit by chance.
pub fn carrying(inside: &Palette) -> Vec<usize> {
    let mut out = Vec::new();
    let mut held = 0.0;
    for ci in inside.materials() {
        if held >= inside.total / 2.0 {
            break;
        }
        out.push(ci);
        held += inside.clusters[ci].px;
    }
    out
}

fn code(c: DVec3) -> u32 {
    let q = |v: f64| (v * 255.0).round_ties_even().clamp(0.0, 255.0) as u32;
    (q(c.x) * 256 + q(c.y)) * 256 + q(c.z)
}

/// The colours seen in a ring about a region, with their pixels: those past the outside
/// palette's clearance ([`clear_of`]) and within the reach a layer's region grows by
/// (20 px at 330), where the beneath runs on out of the layer.
pub struct Ring {
    counts: HashMap<u32, f64>,
}

impl Ring {
    pub fn of(img: &Image, region: &[bool], rim: &[bool]) -> Self {
        let (w, h) = (img.width, img.height);
        let clear = clear_of(img);
        let near = dilate(region, w, h, Conn::Four, clear);
        let far = dilate(region, w, h, Conn::Four, clear + length_px(w, 20.0));
        let mut counts: HashMap<u32, f64> = HashMap::default();
        for i in 0..img.len() {
            if far[i] && !near[i] && !rim[i] {
                *counts.entry(code(img.px[i])).or_insert(0.0) += 1.0;
            }
        }
        Ring { counts }
    }

    /// A material's colours as the ring sees them (weights its ring pixels), or `None`
    /// where the ring holds fewer than [`SAMPLES`] of its pixels.
    fn part(&self, x: &[DVec3]) -> Option<(Vec<DVec3>, Vec<f64>)> {
        let (rx, rw): (Vec<DVec3>, Vec<f64>) = x
            .iter()
            .filter_map(|c| self.counts.get(&code(*c)).map(|w| (*c, *w)))
            .unzip();
        (rw.iter().sum::<f64>() >= SAMPLES as f64).then_some((rx, rw))
    }
}

/// Whether [`gate`] can pass on `inside` whatever the outside: a [`carrying`] cluster with
/// spread to pair, or two flat ones to fix k together.
pub fn may_pass(inside: &Palette) -> bool {
    let (mut spread, mut flat) = (0, 0);
    for ci in carrying(inside) {
        let cl = &inside.clusters[ci];
        let (x, w) = (
            &inside.colors[cl.start..cl.end],
            &inside.weights[cl.start..cl.end],
        );
        if Clouds::of(x, w, &x[..1], &w[..1]).spread <= SHOW_THROUGH {
            flat += 1;
        } else {
            spread += 1;
        }
    }
    spread > 0 || flat >= 2
}

/// The gate on a region: each [`carrying`] cluster against each outside material, passing
/// pairs refitted ([`refine`]) and grouped. A `ring` adds each material as the ring sees it
/// ([`Ring::part`]): a layer may cover only the part of a varying material next to it.
pub fn gate(inside: &Palette, outside: &Palette, ring: Option<&Ring>) -> Gate {
    // per outside material its readings: whole, and as the ring sees it
    type Reading = (usize, bool, Vec<DVec3>, Vec<f64>, Vec<DVec3>, DVec3);
    let mut g = Gate::default();
    let side = |p: &Palette, ci: usize| {
        let cl = &p.clusters[ci];
        (
            p.colors[cl.start..cl.end].to_vec(),
            p.weights[cl.start..cl.end].to_vec(),
        )
    };
    let mut outs: Vec<Reading> = Vec::new();
    for (mat, ci) in outside.materials().enumerate() {
        let whole = side(outside, ci);
        let part = ring.and_then(|r| r.part(&whole.0));
        for (in_ring, (x, w)) in std::iter::once((false, whole)).chain(part.map(|p| (true, p))) {
            let b = sample(&x, &w, SAMPLES);
            let mean = weighted_mean(&x, &w);
            outs.push((mat, in_ring, x, w, b, mean));
        }
    }
    for ci in carrying(inside) {
        let (x, w) = side(inside, ci);
        let px = inside.clusters[ci].px;
        let spread = Clouds::of(&x, &w, &x[..1], &w[..1]).spread;
        // a cluster without spread leaves k open (see `in_range`): no pair of it is fitted
        if spread <= SHOW_THROUGH {
            g.flat += 1;
            continue;
        }
        let o = sample(&x, &w, SAMPLES);
        for &(mat, in_ring, ref bx, ref bw, ref b, bmean) in &outs {
            g.tried += 1;
            let clouds = Clouds { spread, bmean };
            // the pair's relation is its closest fit from any start, at any k: a pair that fits
            // closer at a k out of the pin's range is that, not a layer the pin takes
            let (best, best_matched) = STARTS
                .iter()
                .map(|&k0| icp(&o, b, k0))
                .min_by(|a, b| a.0.residual.total_cmp(&b.0.residual))
                .expect("the starts");
            if g.closest
                .as_ref()
                .is_none_or(|c| best.residual < c.3.residual)
            {
                g.closest = Some((ci, mat, px, best));
            }
            if !in_range(&best, &clouds) {
                continue;
            }
            if g.closest_in_range
                .as_ref()
                .is_none_or(|c| best.residual < c.3.residual)
            {
                g.closest_in_range = Some((ci, mat, px, best));
            }
            if best.residual > RESIDUAL {
                continue;
            }
            // every pair within the tolerance at the coarse count is refitted, not only the
            // best: the refit and the grouping decide
            let (steps, matched) = refine(&x, &w, bx, bw, (best, best_matched));
            let fit = steps.last().map_or(best, |s| s.1);
            if is_layer(&fit, &clouds) {
                g.passing.push(g.pairs.len());
            }
            g.pairs.push(Pair {
                cluster: ci,
                px,
                material: mat,
                bmean,
                in_ring,
                coarse: best,
                steps,
                fit,
                matched,
            });
        }
    }
    // Two flat clusters over two outside materials fix k, though one alone leaves it open:
    // o1 - o2 = k (b1 - b2), the same k in every channel, m from either. Two materials under
    // one relation is a layer, not one material's shading.
    let flats: Vec<(usize, DVec3, f64)> = carrying(inside)
        .into_iter()
        .filter_map(|ci| {
            let (x, w) = side(inside, ci);
            let spread = Clouds::of(&x, &w, &x[..1], &w[..1]).spread;
            (spread <= SHOW_THROUGH).then(|| (ci, weighted_mean(&x, &w), inside.clusters[ci].px))
        })
        .collect();
    // The flat pairs are read against the materials the ring sees beside the region where a
    // ring is given; a flat cluster within the show-through of one of them is that material
    // seen bare, no image.
    let ringed = outs.iter().any(|o| o.1);
    let mats: Vec<(usize, DVec3)> = outs
        .iter()
        .filter(|o| o.1 == ringed)
        .map(|o| (o.0, o.5))
        .collect();
    let flats: Vec<(usize, DVec3, f64)> = flats
        .into_iter()
        .filter(|f| !outs.iter().any(|o| cheb(f.1, o.5) <= SHOW_THROUGH))
        .collect();
    let grid = pin_grid();
    let (lo, hi) = (grid[0], grid[grid.len() - 1]);
    if crate::diagnostics::debug() && !flats.is_empty() {
        let lv = |v: DVec3| (v * 255.0).round().as_ivec3().to_array();
        eprintln!(
            "gate: flat clusters {:?}; outside materials {:?}",
            flats
                .iter()
                .map(|f| (lv(f.1), f.2 as usize))
                .collect::<Vec<_>>(),
            mats.iter().map(|m| lv(m.1)).collect::<Vec<_>>()
        );
    }
    for (ai, &(ci, oi, pxi)) in flats.iter().enumerate() {
        for &(cj, oj, pxj) in &flats[ai + 1..] {
            // two images of one colour are one image
            if cheb(oi, oj) <= SHOW_THROUGH {
                continue;
            }
            for &(mat_a, ba) in &mats {
                for &(mat_b, bb) in &mats {
                    if mat_a == mat_b || cheb(ba, bb) <= SHOW_THROUGH {
                        continue;
                    }
                    g.tried += 1;
                    let (db, d_o) = (ba - bb, oi - oj);
                    let k = d_o.dot(db) / db.dot(db);
                    if !(lo - 1e-9..=hi + 1e-9).contains(&k) {
                        continue;
                    }
                    let m = oi - k * ba;
                    let residual = cheb(oj, k * bb + m);
                    let fit = Fit { k, m, residual };
                    let c = fit.c();
                    let margin = SHOW_THROUGH / (1.0 - k);
                    let in_gamut = c.min_element() >= -margin && c.max_element() <= 1.0 + margin;
                    // both materials moved: a colour is a background or an image, not both
                    let moved =
                        cheb(k * ba + m, ba) > SHOW_THROUGH && cheb(k * bb + m, bb) > SHOW_THROUGH;
                    if !in_gamut || !moved {
                        continue;
                    }
                    if g.closest.as_ref().is_none_or(|c| residual < c.3.residual) {
                        g.closest = Some((ci, mat_a, pxi, fit));
                    }
                    if g.closest_in_range
                        .as_ref()
                        .is_none_or(|c| residual < c.3.residual)
                    {
                        g.closest_in_range = Some((ci, mat_a, pxi, fit));
                    }
                    if residual > RESIDUAL {
                        continue;
                    }
                    for (cl, px, mat, bmean, o) in
                        [(ci, pxi, mat_a, ba, oi), (cj, pxj, mat_b, bb, oj)]
                    {
                        g.passing.push(g.pairs.len());
                        g.pairs.push(Pair {
                            cluster: cl,
                            px,
                            material: mat,
                            bmean,
                            in_ring: false,
                            coarse: fit,
                            steps: Vec::new(),
                            fit,
                            matched: Matched {
                                o: vec![o],
                                b: vec![bmean],
                            },
                        });
                    }
                }
            }
        }
    }
    g.relation = relation(&g);
    g
}

/// The passing pairs grouped: from each as a seed, the pairs within the sum of residuals
/// of it ([`apart`]), one per outside material and inside cluster; the group of the most
/// materials, then pixels, fitted jointly by least squares over its matched pairs.
fn relation(g: &Gate) -> Option<Relation> {
    let mut seeds: Vec<usize> = g.passing.clone();
    seeds.sort_by(|&a, &b| g.pairs[a].fit.residual.total_cmp(&g.pairs[b].fit.residual));
    let mut best: Option<((bool, usize, f64), Vec<usize>)> = None;
    for &s in &seeds {
        let seed = &g.pairs[s];
        let mut members = vec![s];
        for &o in &seeds {
            let p = &g.pairs[o];
            if o == s
                || members
                    .iter()
                    .any(|&m| g.pairs[m].material == p.material || g.pairs[m].cluster == p.cluster)
            {
                continue;
            }
            if apart(seed, p) <= seed.fit.residual + p.fit.residual {
                members.push(o);
            }
        }
        let mats = members.len();
        let px: f64 = members.iter().map(|&m| g.pairs[m].px).sum();
        // a group with a fitted cloud outranks one of flat pairs alone: a wrong pairing of two
        // flat colours is cheap to come by, a cloud's one-to-one match is its whole shape
        let fitted = members.iter().any(|&m| !g.pairs[m].steps.is_empty());
        let key = (fitted, mats, px);
        if best.as_ref().is_none_or(|(bk, _)| {
            key.0 && !bk.0 || key.0 == bk.0 && (key.1 > bk.1 || (key.1 == bk.1 && key.2 > bk.2))
        }) {
            best = Some((key, members));
        }
    }
    let ((_, materials, px), pairs) = best?;
    let (mut o, mut b) = (Vec::new(), Vec::new());
    for &p in &pairs {
        o.extend_from_slice(&g.pairs[p].matched.o);
        b.extend_from_slice(&g.pairs[p].matched.b);
    }
    let (k, m) = if pairs.len() == 1 {
        let f = g.pairs[pairs[0]].fit;
        (f.k, f.m)
    } else {
        least_squares(&o, &b)
    };
    let mut d: Vec<f64> = o
        .iter()
        .zip(&b)
        .map(|(o, b)| cheb(*o, k * *b + m))
        .collect();
    let residual = median(&mut d);
    let drift = pairs
        .iter()
        .map(|&p| {
            let p = &g.pairs[p];
            cheb(p.fit.k * p.bmean + p.fit.m, k * p.bmean + m)
        })
        .fold(0.0, f64::max);
    let ks = pairs.iter().map(|&p| g.pairs[p].fit.k);
    let k_range = (
        ks.clone().fold(f64::INFINITY, f64::min),
        ks.fold(f64::NEG_INFINITY, f64::max),
    );
    Some(Relation {
        k,
        m,
        residual,
        pairs,
        materials,
        px,
        drift,
        k_range,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Heap's algorithm: `f` on every permutation of `p[..k]`.
    fn heap(k: usize, p: &mut Vec<usize>, f: &mut dyn FnMut(&[usize])) {
        if k == 1 {
            f(p);
            return;
        }
        for i in 0..k {
            heap(k - 1, p, f);
            let j = if k.is_multiple_of(2) { i } else { 0 };
            p.swap(j, k - 1);
        }
    }

    #[test]
    fn the_assignment_is_least() {
        // three rows: the identity costs 3, the best 0 + 1 + 1
        let cost = [5.0, 0.0, 9.0, 1.0, 9.0, 9.0, 9.0, 9.0, 1.0];
        assert_eq!(assign(&cost, 3, 3), vec![1, 0, 2]);
        // against every permutation on random 5 by 5 matrices
        let mut r = Lcg(3);
        for _ in 0..20 {
            let n = 5;
            let c: Vec<f64> = (0..n * n).map(|_| r.next()).collect();
            let a = assign(&c, n, n);
            let total = |p: &[usize]| {
                p.iter()
                    .enumerate()
                    .map(|(i, &j)| c[i * n + j])
                    .sum::<f64>()
            };
            let mut perm: Vec<usize> = (0..n).collect();
            let mut least = f64::INFINITY;
            heap(n, &mut perm, &mut |p| least = least.min(total(p)));
            assert!((total(&a) - least).abs() < 1e-12);
        }
        // three rows over five columns: every injection tried
        for _ in 0..20 {
            let (n, m) = (3, 5);
            let c: Vec<f64> = (0..n * m).map(|_| r.next()).collect();
            let a = assign(&c, n, m);
            let mut least = f64::INFINITY;
            for x in 0..m {
                for y in 0..m {
                    for z in 0..m {
                        if x != y && y != z && x != z {
                            least = least.min(c[x] + c[m + y] + c[2 * m + z]);
                        }
                    }
                }
            }
            let total: f64 = a.iter().enumerate().map(|(i, &j)| c[i * m + j]).sum();
            assert!((total - least).abs() < 1e-12);
        }
    }
}
