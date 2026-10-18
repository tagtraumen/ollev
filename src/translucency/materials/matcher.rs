//! The exact score of a relation between two palettes: the matcher.

use std::cell::OnceCell;

use glam::DVec3;

use super::{NOISE, Palette, SHOW_THROUGH, witnesses};
use crate::translucency::raster::{ChebTree, cheb, np_sum, rows_sum};

/// The exact score of one relation ([`Matcher::score`]).
#[derive(Clone, Debug)]
pub struct Score {
    /// distinct materials matched
    pub n: usize,
    pub sel: Vec<bool>,
    pub jj: Vec<usize>,
}

/// The veil's opacity over the layer, at most.
pub const VEIL_MAX: f64 = 0.9;

pub(super) struct Samples {
    pub(super) so: Vec<DVec3>,
    pub(super) sj: Vec<usize>,
    pub(super) nj: Vec<f64>,
    pub(super) sw: Vec<f64>,
    pub(super) sb: Vec<DVec3>,
    pub(super) si: Vec<usize>,
}

/// Exact scoring of relations between an inside palette and an outside one.
pub struct Matcher<'a> {
    pub inside: &'a Palette,
    pub outside: &'a Palette,
    pub tol: f64,
    pub multi: bool,
    pub(super) exclude_self: bool,
    pub(super) same: Vec<usize>,
    pub(super) b: Vec<DVec3>,
    pub(super) bmat: Vec<usize>,
    pub(super) bmean: Vec<DVec3>,
    pub(super) tree: ChebTree,
    pub(super) ocol: Vec<DVec3>,
    pub(super) ow: Vec<f64>,
    pub(super) ojc: Vec<usize>,
    pub(super) nj: usize,
    pub(super) wj: Vec<f64>,
    pub(super) samples: OnceCell<Samples>,
}

impl<'a> Matcher<'a> {
    pub fn new(
        inside: &'a Palette,
        outside: &'a Palette,
        tol: f64,
        exclude_self: bool,
        multi: bool,
    ) -> Self {
        // the outside colors that belong to a material, with that material's index
        let mut mat_of = vec![usize::MAX; outside.clusters.len()];
        for (r, ci) in outside.materials().enumerate() {
            mat_of[ci] = r;
        }
        let (mut b, mut bmat) = (Vec::new(), Vec::new());
        for (c, &l) in outside.colors.iter().zip(&outside.label) {
            if mat_of[l] != usize::MAX {
                b.push(*c);
                bmat.push(mat_of[l]);
            }
        }
        let bmean = outside
            .materials()
            .map(|ci| outside.clusters[ci].mean)
            .collect();
        // each inside material's colours as its witnesses (see [`witnesses`]), in the
        // palette's order of the materials
        let (mut ocol, mut ow, mut ojc) = (Vec::new(), Vec::new(), Vec::new());
        for (r, ci) in inside.materials().enumerate() {
            let cl = &inside.clusters[ci];
            let (x, w) = witnesses(
                &inside.colors[cl.start..cl.end],
                &inside.weights[cl.start..cl.end],
            );
            ojc.extend(std::iter::repeat_n(r, x.len()));
            ocol.extend(x);
            ow.extend(w);
        }
        let nj = inside.materials().len();
        let mut wj = vec![0.0; nj];
        for (j, w) in ojc.iter().zip(&ow) {
            wj[*j] += *w;
        }
        Matcher {
            inside,
            outside,
            tol,
            multi,
            exclude_self,
            // one palette on both sides (the whole image): a cluster cannot explain itself
            same: inside.materials().collect(),
            tree: ChebTree::new(&b),
            b,
            bmat,
            bmean,
            ocol,
            ow,
            ojc,
            nj,
            wj,
            samples: OnceCell::new(),
        }
    }

    /// Per inside color: the material of its nearest outside color under (k, m) within the
    /// tolerance, -1 if none or if the map does not move it, and that outside color.
    pub fn matches(&self, k: f64, m: DVec3) -> (Vec<i64>, Vec<usize>) {
        let n = self.ocol.len();
        if self.tree.is_empty() || n == 0 {
            return (vec![-1; n], vec![0; n]);
        }
        let (mut mat, mut jj) = (vec![-1i64; n], vec![0usize; n]);
        for i in 0..n {
            if let Some((_, j)) = self.tree.nearest((self.ocol[i] - m) / k, self.tol / k) {
                jj[i] = j;
                // the map moves it: an inside color within the show-through of its background
                // is the background itself, no evidence of a layer
                if cheb(self.ocol[i], self.b[j]) > SHOW_THROUGH {
                    mat[i] = self.bmat[j] as i64;
                }
            }
            if self.exclude_self && mat[i] == self.ojc[i] as i64 {
                mat[i] = -1;
            }
        }
        (mat, jj)
    }

    /// The relation's score: distinct materials, the inside colours matched and to what.
    pub fn score(&self, k: f64, m: DVec3) -> Score {
        let (mat, jj) = self.matches(k, m);
        let nm = self.bmean.len();
        let mut w = vec![vec![0.0f64; nm.max(1)]; self.nj];
        for ((&mt, &j), &wt) in mat.iter().zip(&self.ojc).zip(&self.ow) {
            if mt >= 0 {
                w[j][mt as usize] += wt;
            }
        }
        let mut corr: Vec<(usize, usize, f64)> = Vec::new();
        if nm > 0 {
            for (j, (row, &total)) in w.iter().zip(&self.wj).enumerate() {
                let mut best = 0;
                for (i, &v) in row.iter().enumerate().skip(1) {
                    if v > row[best] {
                        best = i;
                    }
                }
                let sh = row[best] / total.max(1e-9);
                if self.multi {
                    // every material whose image within an inside cluster has MIN_MATERIAL_PX
                    for (i, &v) in row.iter().enumerate() {
                        if v >= self.inside.min_px {
                            corr.push((j, i, (v / (0.5 * total)).min(1.0)));
                        }
                    }
                } else if sh >= 0.5 {
                    corr.push((j, best, sh));
                }
            }
        }
        if self.exclude_self {
            // a color is a background or an image, not both: a material that is matched as an
            // image drops out as a background (the larger share keeps its role)
            corr.sort_by(|a, b| b.2.total_cmp(&a.2));
            let (mut images, mut bgs, mut kept) = (Vec::new(), Vec::new(), Vec::new());
            for (j, i, s) in corr {
                let (ji, ii) = (self.same[j], self.same[i]);
                if bgs.contains(&ji) || images.contains(&ii) {
                    continue;
                }
                images.push(ji);
                bgs.push(ii);
                kept.push((j, i, s));
            }
            corr = kept;
        }
        // distinct materials: per outside material its best share; materials whose images
        // lie within the show-through of each other are one background
        let mut per: Vec<(usize, f64, usize)> = Vec::new();
        for &(j, i, s) in &corr {
            match per.iter_mut().find(|e| e.0 == i) {
                Some(e) => {
                    if s > e.1 {
                        e.1 = s;
                        e.2 = j;
                    }
                }
                None => {
                    if s > 0.0 {
                        per.push((i, s, j));
                    }
                }
            }
        }
        let mut order = per;
        order.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut chosen: Vec<(usize, f64, usize)> = Vec::new();
        let mut images: Vec<ChebTree> = Vec::new();
        for &(i, s, j) in &order {
            if !chosen
                .iter()
                .all(|c| (k * (self.bmean[i] - self.bmean[c.0])).abs().max_element() > SHOW_THROUGH)
            {
                continue;
            }
            // Two materials' images have to be apart by the show-through, or the inside holds
            // one continuum that either explains, and the second would pin a wrong k
            let ei: Vec<DVec3> = (0..mat.len())
                .filter(|&e| mat[e] == i as i64)
                .map(|e| self.ocol[e])
                .collect();
            if !ei.is_empty()
                && images.iter().any(|t| {
                    !t.is_empty()
                        && ei
                            .iter()
                            .map(|q| t.nearest_dist(*q))
                            .fold(f64::INFINITY, f64::min)
                            <= SHOW_THROUGH
                })
            {
                continue;
            }
            chosen.push((i, s, j));
            images.push(ChebTree::new(&ei));
        }
        let sel: Vec<bool> = (0..mat.len())
            .map(|e| {
                chosen
                    .iter()
                    .any(|&(i, _, j)| self.ojc[e] == j && mat[e] == i as i64)
            })
            .collect();
        Score {
            n: chosen.len(),
            sel,
            jj,
        }
    }

    /// m refitted at fixed k over the matched pairs, each inside material weighed equally.
    pub fn refit_m(&self, k: f64, mut m: DVec3, rounds: usize) -> (DVec3, Score) {
        for _ in 0..rounds {
            let s = self.score(k, m);
            if s.n == 0 {
                return (m, s);
            }
            let idx: Vec<usize> = (0..s.sel.len()).filter(|&e| s.sel[e]).collect();
            if idx.len() < 2 {
                return (m, s);
            }
            let w: Vec<f64> = idx
                .iter()
                .map(|&e| self.ow[e] / self.wj[self.ojc[e]])
                .collect();
            let m2 = rows_sum(
                idx.iter()
                    .zip(&w)
                    .map(|(&e, w)| *w * (self.ocol[e] - k * self.b[s.jj[e]])),
            ) / np_sum(&w);
            let settled = cheb(m2, m) < 0.25 / 255.0;
            m = m2;
            if settled {
                break;
            }
        }
        (m, self.score(k, m))
    }

    pub(super) fn samples(&self) -> &Samples {
        self.samples.get_or_init(|| {
            let (mut so, mut sj, mut nj, mut sw) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (j, ci) in self.inside.materials().enumerate() {
                let (s, w) = self.inside.samples(ci);
                nj.push(s.len() as f64);
                sj.extend(std::iter::repeat_n(j, s.len()));
                so.extend(s);
                sw.extend(w);
            }
            let (mut sb, mut si) = (Vec::new(), Vec::new());
            for (i, ci) in self.outside.materials().enumerate() {
                let (s, _) = self.outside.samples(ci);
                si.extend(std::iter::repeat_n(i, s.len()));
                sb.extend(s);
            }
            Samples {
                so,
                sj,
                nj,
                sw,
                sb,
                si,
            }
        })
    }
}

/// Whether the relation's color lies in gamut to the noise over the show-through.
pub fn in_gamut_c(k: f64, m: DVec3) -> bool {
    let c = m / (1.0 - k);
    let margin = NOISE / (1.0 - k);
    c.min_element() >= -margin && c.max_element() <= 1.0 + margin
}

pub(super) fn sorted_unique(mut v: Vec<i64>) -> Vec<i64> {
    v.sort_unstable();
    v.dedup();
    v
}

/// Indices by value, largest first, ties in index order.
pub(super) fn argsort_desc(v: &[f64]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..v.len()).collect();
    o.sort_by(|&a, &b| (-v[a]).total_cmp(&-v[b]));
    o
}
