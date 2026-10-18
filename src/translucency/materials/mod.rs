//! Correspondence of materials: one homothety `o = k b + m` mapping the material clusters
//! outside a region onto the color clusters inside it, found by votes on a k grid
//! ([`vote_cells`]) and scored exactly ([`Matcher::score`]); a color counts only if moved.

use crate::HashMap;
use std::cell::OnceCell;

use glam::DVec3;

use super::raster::{
    ChebTree, Conn, Image, NOISE, SHOW_THROUGH, area_px, cheb, dilate, gradient_magnitude, label,
    length_px, np_sum, reduceat_sum, rows_sum, unit_of,
};

mod matcher;
mod relations;

pub use matcher::*;
pub use relations::*;

/// A pixel is on an edge's anti-aliased rim where the color changes by more than this per
/// pixel: blurred shading ramps slower, an edge steps within a pixel. At 330 px wide;
/// [`rim_slope`] at the image's width.
pub const RIM_SLOPE: f64 = 8.0 * NOISE;
/// Colors link into one cluster when they lie in neighboring cells of this many levels (3
/// to 5 levels apart): the 8-bit noise of a smooth ramp, NOISE, plus a level of rounding.
pub const LINK_LEVELS: i64 = 3;
/// A color seen on one pixel only is noise or a rim the slope test missed (at 330;
/// [`min_seen`] at the image's width).
pub const MIN_SEEN: i64 = 2;
/// The match tolerance of a peeled inside color against an outside color: the noise plus
/// half a level of rounding either side. Clusters are thin, so a looser tolerance lets wrong
/// k through.
pub const MATCH_TOL: f64 = NOISE + 1.0 / 255.0;
/// A cluster of fewer pixels is not a material (a stray shading step, a speck), at 330;
/// [`min_material_px`] at the image's width.
pub const MIN_MATERIAL_PX: f64 = 16.0;
/// How many colors a cluster votes with: its colors grouped on a coarse grid, one per cell.
pub const VOTE_GRID: f64 = 8.0 / 255.0;
/// The cells of LINK_LEVELS along a channel.
const LINK_CELLS: i64 = 256 / LINK_LEVELS + 1;

/// [`RIM_SLOPE`] at this image's width: at s times the width a drawing's shading ramps s times
/// slower per pixel, while a step's anti-aliasing stays a pixel or two wide.
pub fn rim_slope(img: &Image) -> f64 {
    RIM_SLOPE / unit_of(img.width)
}

/// [`MIN_MATERIAL_PX`] at this image's width, an area: a stray shading step or speck covers
/// s^2 times the pixels at s times the width.
pub fn min_material_px(img: &Image) -> f64 {
    area_px(img.width, MIN_MATERIAL_PX)
}

/// [`MIN_SEEN`] at this image's width, an area: a mixed rim colour seen once at 330 is seen
/// four times at 660, and would link two materials into one cluster.
pub fn min_seen(img: &Image) -> f64 {
    area_px(img.width, MIN_SEEN as f64).round()
}

/// How far an outside palette starts clear of a region: 2 px at 330, in the unit.
pub fn clear_of(img: &Image) -> usize {
    length_px(img.width, 2.0)
}

/// The rims: pixels whose color changes by more than [`rim_slope`] per pixel.
pub fn rim_of(img: &Image) -> Vec<bool> {
    gradient_magnitude(img)
        .into_iter()
        .map(|g| g > rim_slope(img))
        .collect()
}

/// An 8-bit color as one integer.
fn code_of(c: DVec3) -> i64 {
    let q = |v: f64| (v * 255.0).round_ties_even() as i64;
    (q(c.x) * 256 + q(c.y)) * 256 + q(c.z)
}

fn decode(u: i64) -> [i64; 3] {
    [u / 65536, (u / 256) % 256, u % 256]
}

fn color_of(u: i64) -> DVec3 {
    let q = decode(u);
    DVec3::new(
        q[0] as f64 / 255.0,
        q[1] as f64 / 255.0,
        q[2] as f64 / 255.0,
    )
}

/// The distinct values, sorted, and their counts.
fn unique_counts(mut codes: Vec<i64>) -> (Vec<i64>, Vec<i64>) {
    codes.sort_unstable();
    let mut u: Vec<i64> = Vec::new();
    let mut w: Vec<i64> = Vec::new();
    for c in codes {
        if u.last() == Some(&c) {
            *w.last_mut().unwrap() += 1;
        } else {
            u.push(c);
            w.push(1);
        }
    }
    (u, w)
}

/// One cluster of a palette: its colors' range in the palette, its pixels and mean.
#[derive(Clone, Debug)]
pub struct Cluster {
    pub start: usize,
    pub end: usize,
    pub px: f64,
    pub mean: DVec3,
}

fn cluster_stats(x: &[DVec3], w: &[f64], start: usize) -> Cluster {
    let tot = np_sum(w);
    let mu = rows_sum(x.iter().zip(w).map(|(c, w)| *w * *c)) / tot;
    Cluster {
        start,
        end: start + x.len(),
        px: tot,
        mean: mu,
    }
}

/// The colors of a mask's pixels off the rims, in clusters largest first: `weights` in pixels,
/// `label` the cluster of each color. A material is a cluster of at least `min_px` pixels, so the
/// materials are the first `material_count` clusters ([`Palette::materials`]).
#[derive(Clone, Debug, Default)]
pub struct Palette {
    pub colors: Vec<DVec3>,
    pub weights: Vec<f64>,
    pub label: Vec<usize>,
    pub clusters: Vec<Cluster>,
    pub material_count: usize,
    pub total: f64,
    /// the least pixels of a material, as the palette was built with
    pub min_px: f64,
}

/// The codes of the 8-bit colors of the pixels of `sel`, in raster order.
fn codes_of(img: &Image, sel: &[bool]) -> Vec<i64> {
    img.px
        .iter()
        .zip(sel)
        .filter(|(_, s)| **s)
        .map(|(c, _)| code_of(*c))
        .collect()
}

/// Single linkage of 8-bit colors on a grid: occupied cells of LINK_LEVELS, 26-connected.
/// The component of each color, numbered in the order of each component's first cell.
fn link_colors(u: &[i64]) -> (Vec<usize>, usize) {
    let cell = |c: i64| {
        let q = decode(c);
        [q[0] / LINK_LEVELS, q[1] / LINK_LEVELS, q[2] / LINK_LEVELS]
    };
    let code = |g: [i64; 3]| (g[0] * LINK_CELLS + g[1]) * LINK_CELLS + g[2];
    let mut cells: Vec<i64> = u.iter().map(|&c| code(cell(c))).collect();
    cells.sort_unstable();
    cells.dedup();
    let index: HashMap<i64, usize> = cells.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let mut parent: Vec<usize> = (0..cells.len()).collect();
    fn find(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    for (i, &c) in cells.iter().enumerate() {
        let g = [
            c / (LINK_CELLS * LINK_CELLS),
            (c / LINK_CELLS) % LINK_CELLS,
            c % LINK_CELLS,
        ];
        for d0 in -1..=1 {
            for d1 in -1..=1 {
                for d2 in -1..=1 {
                    let n = [g[0] + d0, g[1] + d1, g[2] + d2];
                    if n.iter().any(|&v| !(0..LINK_CELLS).contains(&v)) {
                        continue;
                    }
                    if let Some(&j) = index.get(&code(n)) {
                        let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                        if a != b {
                            parent[a.max(b)] = a.min(b);
                        }
                    }
                }
            }
        }
    }
    // components numbered in the order of their first cell
    let mut comp = vec![usize::MAX; cells.len()];
    let mut n = 0;
    for i in 0..cells.len() {
        let r = find(&mut parent, i);
        if comp[r] == usize::MAX {
            comp[r] = n;
            n += 1;
        }
        comp[i] = comp[r];
    }
    (u.iter().map(|&c| comp[index[&code(cell(c))]]).collect(), n)
}

/// The most colours an inside material is scored by: each witness is searched in the outside
/// tree per relation, so they are the matcher's cost (the outside stays whole). A channel's
/// levels: a material at every step of a full ramp.
const MAX_WITNESSES: usize = 256;

/// A material's colours `x` with pixels `w` as witnesses: all where there are at most
/// [`MAX_WITNESSES`], else a systematic sample by pixels in code order, each draw carrying an
/// equal share. Colours stay exact, as a bucketed mean would move them.
fn witnesses(x: &[DVec3], w: &[f64]) -> (Vec<DVec3>, Vec<f64>) {
    let total: f64 = w.iter().sum();
    if x.len() <= MAX_WITNESSES || total <= 0.0 {
        return (x.to_vec(), w.to_vec());
    }
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by_key(|&i| code_of(x[i]));
    let n = MAX_WITNESSES;
    let share = total / n as f64;
    let (mut xs, mut ws): (Vec<DVec3>, Vec<f64>) = (Vec::new(), Vec::new());
    let (mut at, mut cum) = (0usize, 0.0f64);
    let mut last: Option<usize> = None;
    for draw in 0..n {
        // the colour holding the pixel at the draw's position in the cumulative distribution
        let pos = (draw as f64 + 0.5) * share;
        while at + 1 < order.len() && cum + w[order[at]] <= pos {
            cum += w[order[at]];
            at += 1;
        }
        if last == Some(at) {
            let i = ws.len() - 1;
            ws[i] += share;
        } else {
            xs.push(x[order[at]]);
            ws.push(share);
            last = Some(at);
        }
    }
    (xs, ws)
}

impl Palette {
    /// Clusters by color alone: the colors of `mask` off `rim` linked where they step by a
    /// few levels.
    pub fn new(img: &Image, mask: &[bool], rim: &[bool], min_px: f64) -> Self {
        let sel: Vec<bool> = mask.iter().zip(rim).map(|(m, r)| *m && !*r).collect();
        let (u, w) = unique_counts(codes_of(img, &sel));
        let (u, w): (Vec<i64>, Vec<i64>) = u
            .into_iter()
            .zip(w)
            .filter(|(_, w)| *w as f64 >= min_seen(img))
            .unzip();
        let mut groups = Vec::new();
        if !u.is_empty() {
            let (raw, nl) = link_colors(&u);
            for ci in 0..nl {
                let (gu, gw): (Vec<i64>, Vec<i64>) = u
                    .iter()
                    .zip(&w)
                    .zip(&raw)
                    .filter(|(_, r)| **r == ci)
                    .map(|((u, w), _)| (*u, *w))
                    .unzip();
                groups.push((gu, gw));
            }
        }
        Self::from_groups(groups, min_px)
    }

    fn from_groups(mut groups: Vec<(Vec<i64>, Vec<i64>)>, min_px: f64) -> Self {
        groups.sort_by_key(|g| -g.1.iter().sum::<i64>());
        let mut p = Palette::default();
        for (ci, (u, w)) in groups.iter().enumerate() {
            let x: Vec<DVec3> = u.iter().map(|&c| color_of(c)).collect();
            let wf: Vec<f64> = w.iter().map(|&v| v as f64).collect();
            let start = p.colors.len();
            p.clusters.push(cluster_stats(&x, &wf, start));
            p.colors.extend(x);
            p.weights.extend(wf);
            p.label.extend(std::iter::repeat_n(ci, u.len()));
        }
        p.material_count = p.clusters.iter().take_while(|c| c.px >= min_px).count();
        p.total = np_sum(&p.weights);
        p.min_px = min_px;
        p
    }

    /// The clusters that are materials, largest first.
    pub fn materials(&self) -> std::ops::Range<usize> {
        0..self.material_count
    }

    /// Clusters by place and color: the mask's pixels off the rims in 4-connected pieces,
    /// joined where most of the smaller one's colors are the other's, so a sky and pale skin
    /// that link by color alone stay apart.
    pub fn spatial(img: &Image, mask: &[bool], rim: &[bool], min_px: f64, overlap: f64) -> Self {
        let sel: Vec<bool> = mask.iter().zip(rim).map(|(m, r)| *m && !*r).collect();
        let (lab, n) = label(&sel, img.width, img.height, Conn::Four);
        Self::of_pieces(img, &lab, n, min_px, overlap)
    }

    /// Clusters by the opaque trace's shapes: the mask's pixels off the rims in the pieces
    /// `shapes` labels (1-based), joined as [`Palette::spatial`] joins its pieces. A shape is
    /// local where a color is not.
    pub fn of_shapes(
        img: &Image,
        mask: &[bool],
        rim: &[bool],
        shapes: &[u32],
        min_px: f64,
        overlap: f64,
    ) -> Self {
        let lab: Vec<u32> = (0..img.len())
            .map(|i| if mask[i] && !rim[i] { shapes[i] } else { 0 })
            .collect();
        let n = lab.iter().max().map_or(0, |&l| l as usize);
        Self::of_pieces(img, &lab, n, min_px, overlap)
    }

    /// The pieces (labels 1..=n, 0 off) joined into materials by their colors.
    fn of_pieces(img: &Image, lab: &[u32], n: usize, min_px: f64, overlap: f64) -> Self {
        if n == 0 {
            return Self::from_groups(Vec::new(), min_px);
        }
        // per piece its colors and counts
        let keys: Vec<i64> = img
            .px
            .iter()
            .zip(lab)
            .filter(|(_, l)| **l > 0)
            .map(|(c, l)| (*l as i64 - 1) * (1 << 24) + code_of(*c))
            .collect();
        let (uk, ck) = unique_counts(keys);
        let piece: Vec<usize> = uk.iter().map(|k| (k >> 24) as usize).collect();
        let col: Vec<i64> = uk.iter().map(|k| k & ((1 << 24) - 1)).collect();
        let mut psize = vec![0.0f64; n];
        for (p, c) in piece.iter().zip(&ck) {
            psize[*p] += *c as f64;
        }
        let big: Vec<usize> = (0..n).filter(|&p| psize[p] >= min_px).collect();
        // occupancy of grid cells per piece, the cells of each color dilated by one
        let cell: Vec<i64> = col
            .iter()
            .map(|&c| {
                let q = decode(c);
                ((q[0] / LINK_LEVELS) * LINK_CELLS + q[1] / LINK_LEVELS) * LINK_CELLS
                    + q[2] / LINK_LEVELS
            })
            .collect();
        // each piece's entries are a run of the sorted keys
        let mut range = vec![(0usize, 0usize); n];
        let mut i = 0;
        while i < piece.len() {
            let mut j = i;
            while j < piece.len() && piece[j] == piece[i] {
                j += 1;
            }
            range[piece[i]] = (i, j);
            i = j;
        }
        // per cell the big pieces holding colours in it, with their pixels there: a colour of
        // the smaller piece is the other's only where the other holds it as densely, as a
        // large piece's few anti-aliased pixels occupy cells too
        let mut owner: HashMap<i64, Vec<(usize, f64)>> = HashMap::default();
        for &p in &big {
            let (a, b) = range[p];
            for e in a..b {
                let v = owner.entry(cell[e]).or_default();
                match v.last_mut() {
                    Some(last) if last.0 == p => last.1 += ck[e] as f64,
                    _ => v.push((p, ck[e] as f64)),
                }
            }
        }
        let mut in_cell: HashMap<(usize, i64), f64> = HashMap::default();
        for &p in &big {
            let (a, b) = range[p];
            for e in a..b {
                *in_cell.entry((p, cell[e])).or_insert(0.0) += ck[e] as f64;
            }
        }
        let offs: Vec<i64> = (-1..=1)
            .flat_map(|a| {
                (-1..=1)
                    .flat_map(move |b| (-1..=1).map(move |c| (a * LINK_CELLS + b) * LINK_CELLS + c))
            })
            .collect();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        let mut by_size = big.clone();
        by_size.sort_by(|&a, &b| psize[a].total_cmp(&psize[b]));
        for &p in &by_size {
            let (a, b) = range[p];
            // the pixels of p whose cell or a neighboring cell another piece occupies, per piece
            let mut hits: Vec<(usize, f64)> = Vec::new();
            for e in a..b {
                let mine = in_cell[&(p, cell[e])];
                let mut seen: Vec<usize> = Vec::new();
                for o in &offs {
                    if let Some(qs) = owner.get(&(cell[e] + o)) {
                        for &(q, theirs) in qs {
                            if q != p
                                && !seen.contains(&q)
                                && psize[q] >= psize[p]
                                && theirs >= mine
                            {
                                seen.push(q);
                                match hits.iter_mut().find(|h| h.0 == q) {
                                    Some(h) => h.1 += ck[e] as f64,
                                    None => hits.push((q, ck[e] as f64)),
                                }
                            }
                        }
                    }
                }
            }
            // A piece joins the one larger piece holding most of its colours, not every piece
            // holding half: joined to each, a piece two materials share would chain them by
            // single linkage
            let best = hits
                .into_iter()
                .filter(|&(_, h)| h >= overlap * psize[p])
                .max_by(|a, b| {
                    a.1.total_cmp(&b.1)
                        .then(psize[a.0].total_cmp(&psize[b.0]))
                        .then(b.0.cmp(&a.0))
                });
            if let Some((q, _)) = best {
                let (ra, rb) = (find(&mut parent, p), find(&mut parent, q));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
        let mut roots: Vec<(usize, Vec<usize>)> = Vec::new();
        for &p in &big {
            let r = find(&mut parent, p);
            match roots.iter_mut().find(|e| e.0 == r) {
                Some(e) => e.1.push(p),
                None => roots.push((r, vec![p])),
            }
        }
        let mut groups = Vec::new();
        for (_, ps) in roots {
            let mut acc: Vec<(i64, i64)> = Vec::new();
            for &p in &ps {
                let (a, b) = range[p];
                acc.extend((a..b).map(|e| (col[e], ck[e])));
            }
            acc.sort_unstable();
            let mut u: Vec<i64> = Vec::new();
            let mut w: Vec<i64> = Vec::new();
            for (c, k) in acc {
                if u.last() == Some(&c) {
                    *w.last_mut().unwrap() += k;
                } else {
                    u.push(c);
                    w.push(k);
                }
            }
            let (u, w): (Vec<i64>, Vec<i64>) = u
                .into_iter()
                .zip(w)
                .filter(|(_, w)| *w as f64 >= min_seen(img))
                .unzip();
            if !u.is_empty() {
                groups.push((u, w));
            }
        }
        Self::from_groups(groups, min_px)
    }

    /// A cluster's colors grouped on the [`VOTE_GRID`], one mean color per cell, and the
    /// cells' pixels: the cluster's support covered evenly, whatever its pixels'
    /// distribution.
    pub fn samples(&self, ci: usize) -> (Vec<DVec3>, Vec<f64>) {
        let cl = &self.clusters[ci];
        let x = &self.colors[cl.start..cl.end];
        let w = &self.weights[cl.start..cl.end];
        let code: Vec<i64> = x
            .iter()
            .map(|c| {
                let g = (*c / VOTE_GRID).floor();
                (g.x as i64 * 64 + g.y as i64) * 64 + g.z as i64
            })
            .collect();
        let mut o: Vec<usize> = (0..x.len()).collect();
        o.sort_by_key(|&i| code[i]);
        let (mut means, mut ws) = (Vec::new(), Vec::new());
        let mut i = 0;
        while i < o.len() {
            let mut j = i;
            while j < o.len() && code[o[j]] == code[o[i]] {
                j += 1;
            }
            let seg = &o[i..j];
            let mut s = DVec3::ZERO;
            for c in 0..3 {
                let v: Vec<f64> = seg.iter().map(|&e| x[e][c] * w[e]).collect();
                s[c] = reduceat_sum(&v);
            }
            let wv: Vec<f64> = seg.iter().map(|&e| w[e]).collect();
            let ws_ = reduceat_sum(&wv);
            means.push(s / ws_);
            ws.push(ws_);
            i = j;
        }
        (means, ws)
    }
}
