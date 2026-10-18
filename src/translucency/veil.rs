//! Translucency proposed by contours continuing through it: two T junctions whose stems face
//! each other across a gap, with one step k times the other under `o = k b + m`, put the
//! low-contrast side under a layer (Metelli's episcotister). Their objects seed [`proposals`].

use crate::HashMap;

use super::find_root as find;
use super::materials::{LINK_LEVELS, Palette, min_material_px};
use super::raster::{DVec2, DVec3, Image, SHOW_THROUGH, forward, length_px, unit_of};
use super::region_graph::crack_cost;

/// The frame outside the picture, as a label.
const FRAME: u32 = u32::MAX;

/// The objects: leaves of one material joined where they touch.
pub struct Pieces {
    pub w: usize,
    pub h: usize,
    pub label: Vec<u32>,
    pub n: usize,
    pub area: Vec<usize>,
    pub mean: Vec<DVec3>,
}

/// The leaves joined into objects: adjacent leaves of one material (most of their pixels off
/// `rim`, [`super::materials::rim_of`]) or with no contour between them are one object; a
/// blend leaf of no material joins its nearer neighbour.
pub fn pieces(img: &Image, leaves: &[u32], rim: &[bool]) -> Pieces {
    let (w, h) = (img.width, img.height);
    let nl = leaves.iter().max().map_or(0, |&l| l as usize + 1);
    let shapes: Vec<u32> = leaves.iter().map(|&l| l + 1).collect();
    let all = vec![true; w * h];
    let pal = Palette::of_shapes(img, &all, rim, &shapes, min_material_px(img), 0.5);
    // a pixel's cluster: each LINK_LEVELS cell of the palette's grid takes the cluster of most
    // of its pixels, a pixel taking its own cell's or else a neighbouring cell's
    let cell = |c: DVec3| -> [i64; 3] {
        let q = |v: f64| (v * 255.0).round() as i64 / LINK_LEVELS;
        [q(c.x), q(c.y), q(c.z)]
    };
    let mut cell_votes: HashMap<[i64; 3], HashMap<usize, f64>> = HashMap::default();
    for ((c, &l), &wt) in pal.colors.iter().zip(&pal.label).zip(&pal.weights) {
        *cell_votes
            .entry(cell(*c))
            .or_default()
            .entry(l)
            .or_default() += wt;
    }
    let cell_of: HashMap<[i64; 3], usize> = cell_votes
        .into_iter()
        .map(|(k, v)| {
            let best = v
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
                .map_or(0, |(l, _)| l);
            (k, best)
        })
        .collect();
    let cluster_at = |c: DVec3| -> Option<usize> {
        let g = cell(c);
        if let Some(&l) = cell_of.get(&g) {
            return Some(l);
        }
        for d0 in -1..=1 {
            for d1 in -1..=1 {
                for d2 in -1..=1 {
                    if let Some(&l) = cell_of.get(&[g[0] + d0, g[1] + d1, g[2] + d2]) {
                        return Some(l);
                    }
                }
            }
        }
        None
    };
    let is_material: Vec<bool> = {
        let mut v = vec![false; pal.clusters.len()];
        for m in pal.materials() {
            v[m] = true;
        }
        v
    };
    // per leaf the votes of its pixels off the rims
    let mut votes: Vec<HashMap<usize, usize>> = vec![HashMap::default(); nl];
    let mut off_rim = vec![0usize; nl];
    let mut sum = vec![DVec3::ZERO; nl];
    let mut cnt = vec![0usize; nl];
    for i in 0..w * h {
        let l = leaves[i] as usize;
        sum[l] += img.px[i];
        cnt[l] += 1;
        if rim[i] {
            continue;
        }
        off_rim[l] += 1;
        if let Some(c) = cluster_at(img.px[i]) {
            *votes[l].entry(c).or_default() += 1;
        }
    }
    let leaf_mean: Vec<DVec3> = (0..nl).map(|l| sum[l] / cnt[l].max(1) as f64).collect();
    let leaf_mat: Vec<Option<usize>> = (0..nl)
        .map(|l| {
            let (&c, &v) = votes[l]
                .iter()
                .max_by_key(|(c, v)| (**v, usize::MAX - **c))?;
            (is_material[c] && 2 * v >= off_rim[l] && off_rim[l] > 0).then_some(c)
        })
        .collect();
    // leaf adjacency, with the crack cost summed per pair
    let u = unit_of(w);
    let mut crack: HashMap<(u32, u32), (f64, usize)> = HashMap::default();
    for i in 0..w * h {
        for j in forward(i, w, h) {
            let (a, b) = (leaves[i], leaves[j]);
            if a != b {
                let e = crack.entry((a.min(b), a.max(b))).or_default();
                e.0 += crack_cost(img, i, j) / u;
                e.1 += 1;
            }
        }
    }
    let mut parent: Vec<usize> = (0..nl).collect();
    for &(a, b) in crack.keys() {
        if let (Some(ma), Some(mb)) = (leaf_mat[a as usize], leaf_mat[b as usize])
            && ma == mb
        {
            let (ra, rb) = (find(&mut parent, a as usize), find(&mut parent, b as usize));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    // no contour: two leaves whose common boundary steps by less than the show-through on
    // average (the region cut's crack cost is the unit there) are one surface the
    // segmentation split
    for (&(a, b), &(c, n)) in &crack {
        if c / n as f64 >= (-0.5f64).exp() {
            let (ra, rb) = (find(&mut parent, a as usize), find(&mut parent, b as usize));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    // blends: a leaf of no material on the segment between two neighbours of a material
    let mut nbrs: Vec<Vec<u32>> = vec![Vec::new(); nl];
    for &(a, b) in crack.keys() {
        nbrs[a as usize].push(b);
        nbrs[b as usize].push(a);
    }
    for l in 0..nl {
        if leaf_mat[l].is_some() {
            continue;
        }
        let c = leaf_mean[l];
        let mut best: Option<(f64, usize)> = None;
        let ns: Vec<usize> = nbrs[l]
            .iter()
            .map(|&q| q as usize)
            .filter(|&q| leaf_mat[q].is_some())
            .collect();
        for (ia, &a) in ns.iter().enumerate() {
            for &b in &ns[ia + 1..] {
                let (ca, cb) = (leaf_mean[a], leaf_mean[b]);
                let ab = cb - ca;
                let len2 = ab.length_squared();
                if len2 < SHOW_THROUGH * SHOW_THROUGH {
                    continue;
                }
                let t = (c - ca).dot(ab) / len2;
                if !(0.0..=1.0).contains(&t) {
                    continue;
                }
                let off = (c - (ca + t * ab)).abs().max_element();
                if off <= SHOW_THROUGH && best.is_none_or(|(o, _)| off < o) {
                    best = Some((off, if t < 0.5 { a } else { b }));
                }
            }
        }
        if let Some((_, q)) = best {
            let (ra, rb) = (find(&mut parent, l), find(&mut parent, q));
            if ra != rb {
                parent[ra] = rb;
            }
        }
    }
    let mut id = vec![u32::MAX; nl];
    let mut n = 0u32;
    for l in 0..nl {
        let r = find(&mut parent, l);
        if id[r] == u32::MAX {
            id[r] = n;
            n += 1;
        }
        id[l] = id[r];
    }
    let label: Vec<u32> = leaves.iter().map(|&l| id[l as usize]).collect();
    let n = n as usize;
    let mut area = vec![0usize; n];
    let mut psum = vec![DVec3::ZERO; n];
    for (i, &p) in label.iter().enumerate() {
        area[p as usize] += 1;
        psum[p as usize] += img.px[i];
    }
    let mean = (0..n).map(|p| psum[p] / area[p].max(1) as f64).collect();
    Pieces {
        w,
        h,
        label,
        n,
        area,
        mean,
    }
}

/// Crack lengths between pieces, per unordered pair.
fn boundaries(label: &[u32], w: usize, h: usize) -> HashMap<(u32, u32), usize> {
    let mut out: HashMap<(u32, u32), usize> = HashMap::default();
    for i in 0..w * h {
        for j in forward(i, w, h) {
            let (a, b) = (label[i], label[j]);
            if a != b {
                *out.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
    }
    out
}

/// One junction of three objects on the crack contours.
#[derive(Clone, Debug)]
struct Junction {
    p: DVec2,
    /// the object whose sector is straight (in front), and the two of the stem
    front: u32,
    stem: [u32; 2],
    /// the stem's direction, away from the junction
    s: DVec2,
    /// the stem pieces' local colours and centroids
    col: [DVec3; 2],
    cen: [DVec2; 2],
    /// paired with another T across a gap (index), and how
    pair: Option<(usize, Pair)>,
}

/// How two facing Ts read.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Pair {
    /// the same contour, the same colours: an opaque thing between
    Between,
    /// one relation, k < 1: the lower contrast side is under a layer
    Veil { k: f64, m: DVec3, lo_is_self: bool },
    /// collinear but no one relation
    Broken,
}

/// The labels of the four pixels around corner (x, y), the frame outside.
fn corner(label: &[u32], w: usize, h: usize, x: usize, y: usize) -> [u32; 4] {
    let at = |xx: isize, yy: isize| {
        if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
            FRAME
        } else {
            label[yy as usize * w + xx as usize]
        }
    };
    let (x, y) = (x as isize, y as isize);
    [at(x - 1, y - 1), at(x, y - 1), at(x - 1, y), at(x, y)]
}

/// The two labels across the crack leaving corner c in direction d (0 up, 1 right, 2 down,
/// 3 left), from the corner's pixels [nw, ne, sw, se].
fn across(c: [u32; 4], d: usize) -> (u32, u32) {
    match d {
        0 => (c[0], c[1]),
        1 => (c[1], c[3]),
        2 => (c[2], c[3]),
        _ => (c[0], c[2]),
    }
}

const STEP: [(isize, isize); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

fn distinct(c: [u32; 4]) -> usize {
    let mut v = c.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// Follows the crack between `pair` from corner (x, y) leaving in direction d for at most
/// `reach` steps or to the next junction; the end corner.
fn follow(
    label: &[u32],
    w: usize,
    h: usize,
    (mut x, mut y): (usize, usize),
    mut d: usize,
    pair: (u32, u32),
    reach: usize,
) -> (usize, usize) {
    let same =
        |a: (u32, u32), b: (u32, u32)| (a.0 == b.0 && a.1 == b.1) || (a.0 == b.1 && a.1 == b.0);
    for _ in 0..reach {
        let (nx, ny) = (x as isize + STEP[d].0, y as isize + STEP[d].1);
        if nx < 0 || ny < 0 || nx > w as isize || ny > h as isize {
            break;
        }
        (x, y) = (nx as usize, ny as usize);
        let c = corner(label, w, h, x, y);
        if distinct(c) >= 3 {
            break;
        }
        let back = (d + 2) % 4;
        let next = (0..4).find(|&e| e != back && same(across(c, e), pair));
        match next {
            Some(e) => d = e,
            None => break,
        }
    }
    (x, y)
}

/// The pieces as the junctions read them: every object under [`min_material_px`] taken into
/// its neighbour of the nearest colour, so slivers and specks do not cut a contour into arms
/// of a pixel or two.
fn coarse(img: &Image, pieces: &Pieces) -> Vec<u32> {
    let (w, h) = (pieces.w, pieces.h);
    let min_px = min_material_px(img);
    let bounds = boundaries(&pieces.label, w, h);
    let mut nbrs: Vec<Vec<u32>> = vec![Vec::new(); pieces.n];
    for &(a, b) in bounds.keys() {
        nbrs[a as usize].push(b);
        nbrs[b as usize].push(a);
    }
    let mut to: Vec<usize> = (0..pieces.n).collect();
    let mut order: Vec<usize> = (0..pieces.n).collect();
    order.sort_by_key(|&p| pieces.area[p]);
    for p in order {
        if (pieces.area[p] as f64) >= min_px {
            continue;
        }
        let c = pieces.mean[p];
        let best = nbrs[p]
            .iter()
            .map(|&q| q as usize)
            .filter(|&q| {
                pieces.area[q] > pieces.area[p] || (pieces.area[q] == pieces.area[p] && q < p)
            })
            .min_by(|&a, &b| {
                (pieces.mean[a] - c)
                    .length_squared()
                    .total_cmp(&(pieces.mean[b] - c).length_squared())
            });
        if let Some(q) = best {
            to[p] = q;
        }
    }
    let resolve = |mut p: usize| {
        for _ in 0..64 {
            if to[p] == p {
                break;
            }
            p = to[p];
        }
        p as u32
    };
    pieces.label.iter().map(|&p| resolve(p as usize)).collect()
}

/// Every three-object junction of the coarsened pieces (none on the frame), read as a T
/// where one sector is straight within 22.5 degrees and the others turn by 30 at least; arms
/// followed up to 6 units, tangent the chord.
fn junctions(img: &Image, pieces: &Pieces) -> Vec<Junction> {
    let (w, h) = (pieces.w, pieces.h);
    let coarse = coarse(img, pieces);
    let label = &coarse;
    let reach = length_px(w, 6.0);
    let least = 2.0;
    let r_col = length_px(w, 3.0) as isize;
    let tol_t = std::f64::consts::PI / 8.0;
    let tol_stem = std::f64::consts::PI / 6.0;
    let mut out = Vec::new();
    for y in 1..h {
        for x in 1..w {
            let c = corner(label, w, h, x, y);
            if distinct(c) != 3 {
                continue;
            }
            let p = DVec2::new(x as f64, y as f64);
            // the three arms
            let mut arms: Vec<((u32, u32), DVec2, f64)> = Vec::new();
            for d in 0..4 {
                let pr = across(c, d);
                if pr.0 == pr.1 {
                    continue;
                }
                let (ex, ey) = follow(label, w, h, (x, y), d, pr, reach);
                let v = DVec2::new(ex as f64, ey as f64) - p;
                arms.push((pr, v.normalize_or_zero(), v.length()));
            }
            if arms.len() != 3 || arms.iter().any(|a| a.2 < least) {
                continue;
            }
            let ang = |v: DVec2| v.y.atan2(v.x);
            arms.sort_by(|a, b| ang(a.1).total_cmp(&ang(b.1)));
            let common = |a: (u32, u32), b: (u32, u32)| {
                if a.0 == b.0 || a.0 == b.1 { a.0 } else { a.1 }
            };
            let mut sectors: Vec<(u32, f64, usize)> = Vec::new();
            for i in 0..3 {
                let (a, b) = (&arms[i], &arms[(i + 1) % 3]);
                let mut t = ang(b.1) - ang(a.1);
                if t <= 0.0 {
                    t += 2.0 * std::f64::consts::PI;
                }
                // the arm opposite this sector is the stem
                sectors.push((common(a.0, b.0), t, (i + 2) % 3));
            }
            let Some((fi, f)) = sectors.iter().enumerate().min_by(|a, b| {
                (a.1.1 - std::f64::consts::PI)
                    .abs()
                    .total_cmp(&(b.1.1 - std::f64::consts::PI).abs())
            }) else {
                continue;
            };
            let dev = (f.1 - std::f64::consts::PI).abs();
            let others_ok = sectors
                .iter()
                .enumerate()
                .all(|(i, s)| i == fi || s.1 >= tol_stem);
            if dev > tol_t || !others_ok {
                continue;
            }
            let stem_arm = &arms[f.2];
            let stem = [stem_arm.0.0, stem_arm.0.1];
            if stem.contains(&f.0) {
                continue;
            }
            // local colours and centroids of the stem's pieces and the front, off their rims
            let mut acc = [(DVec3::ZERO, DVec2::ZERO, 0.0f64); 3];
            let mut all = [(DVec3::ZERO, DVec2::ZERO, 0.0f64); 3];
            let which = |l: u32| [stem[0], stem[1], f.0].iter().position(|&q| q == l);
            for yy in (y as isize - r_col).max(0)..(y as isize + r_col).min(h as isize) {
                for xx in (x as isize - r_col).max(0)..(x as isize + r_col).min(w as isize) {
                    let i = yy as usize * w + xx as usize;
                    let Some(k) = which(label[i]) else { continue };
                    let q = DVec2::new(xx as f64 + 0.5, yy as f64 + 0.5);
                    all[k].0 += img.px[i];
                    all[k].1 += q;
                    all[k].2 += 1.0;
                    let inner =
                        [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)]
                            .iter()
                            .all(|&(dx, dy)| {
                                let (ax, ay) = (xx + dx, yy + dy);
                                ax < 0
                                    || ay < 0
                                    || ax >= w as isize
                                    || ay >= h as isize
                                    || label[ay as usize * w + ax as usize] == label[i]
                            });
                    if inner {
                        acc[k].0 += img.px[i];
                        acc[k].1 += q;
                        acc[k].2 += 1.0;
                    }
                }
            }
            let mean = |k: usize| {
                let a = if acc[k].2 > 0.0 { acc[k] } else { all[k] };
                (a.0 / a.2.max(1.0), a.1 / a.2.max(1.0))
            };
            let (c0, p0) = mean(0);
            let (c1, p1) = mean(1);
            out.push(Junction {
                p,
                front: f.0,
                stem,
                s: stem_arm.1,
                col: [c0, c1],
                cen: [p0, p1],
                pair: None,
            });
        }
    }
    out
}

fn cross(a: DVec2, b: DVec2) -> f64 {
    a.x * b.y - a.y * b.x
}

/// The stem's step left minus right of direction `s`.
fn step_lr(j: &Junction, s: DVec2) -> (DVec3, DVec3) {
    let l0 = cross(s, j.cen[0] - j.p) > 0.0;
    if l0 {
        (j.col[0], j.col[1])
    } else {
        (j.col[1], j.col[0])
    }
}

/// Pairs Ts whose stems face each other across up to 8 units, collinear within 25 degrees,
/// offset at most 1.5 px plus a fifth of their distance, read as between (same colours),
/// veil (one relation, k < 1) or broken.
fn pair_junctions(js: &mut [Junction], w: usize) {
    let gap = length_px(w, 8.0) as f64;
    let cell = gap.max(1.0);
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::default();
    for (i, j) in js.iter().enumerate() {
        grid.entry(((j.p.x / cell) as i64, (j.p.y / cell) as i64))
            .or_default()
            .push(i);
    }
    let cos_tol = (25f64).to_radians().cos();
    let mut taken = vec![false; js.len()];
    for i in 0..js.len() {
        if taken[i] {
            continue;
        }
        let (p, s) = (js[i].p, js[i].s);
        let (gx, gy) = ((p.x / cell) as i64, (p.y / cell) as i64);
        let mut best: Option<(f64, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &k in grid.get(&(gx + dx, gy + dy)).map_or(&[][..], |v| &v[..]) {
                    if k == i || taken[k] {
                        continue;
                    }
                    let q = &js[k];
                    let d = q.p - p;
                    let along = d.dot(-s);
                    let perp = cross(s, d).abs();
                    if along < -1.0 || d.length() > gap || q.s.dot(s) > -cos_tol {
                        continue;
                    }
                    if perp > 1.5 + 0.2 * along.max(0.0) {
                        continue;
                    }
                    let score = perp + 0.1 * d.length();
                    if best.is_none_or(|(b, _)| score < b) {
                        best = Some((score, k));
                    }
                }
            }
        }
        let Some((_, k)) = best else { continue };
        let (a1, b1) = step_lr(&js[i], s);
        let (a2, b2) = step_lr(&js[k], s);
        let (d1, d2) = (a1 - b1, a2 - b2);
        let (n1, n2) = (d1.abs().max_element(), d2.abs().max_element());
        let read = if n1.max(n2) < SHOW_THROUGH {
            None
        } else if (a1 - a2).abs().max_element() <= SHOW_THROUGH
            && (b1 - b2).abs().max_element() <= SHOW_THROUGH
        {
            Some(Pair::Between)
        } else {
            // the lower contrast side as the image of the higher
            let (lo_is_self, hi, lo, hi_a, lo_a, hi_b, lo_b) = if n1 < n2 {
                (true, d2, d1, a2, a1, b2, b1)
            } else {
                (false, d1, d2, a1, a2, b1, b2)
            };
            let k = lo.dot(hi) / hi.length_squared().max(1e-12);
            let res = (lo - k * hi).abs().max_element();
            let hi_n = hi.abs().max_element();
            if res <= SHOW_THROUGH && k > 0.0 && k <= 1.0 - SHOW_THROUGH / hi_n {
                let m = 0.5 * ((lo_a - k * hi_a) + (lo_b - k * hi_b));
                Some(Pair::Veil { k, m, lo_is_self })
            } else {
                Some(Pair::Broken)
            }
        };
        let Some(read) = read else { continue };
        taken[i] = true;
        taken[k] = true;
        let flip = match read {
            Pair::Veil {
                k: kk,
                m,
                lo_is_self,
            } => Pair::Veil {
                k: kk,
                m,
                lo_is_self: !lo_is_self,
            },
            other => other,
        };
        js[i].pair = Some((k, read));
        js[k].pair = Some((i, flip));
    }
}

/// Each veil pair once: its low-contrast T, its high-contrast T and the pair's relation.
fn veil_pairs(js: &[Junction]) -> impl Iterator<Item = (&Junction, &Junction, f64, DVec3)> {
    js.iter().enumerate().filter_map(|(i, j)| {
        let Some((k, Pair::Veil { k: kk, m, lo_is_self })) = j.pair else {
            return None;
        };
        if k < i {
            return None;
        }
        let o = &js[k];
        let (lo, hi) = if lo_is_self { (j, o) } else { (o, j) };
        Some((lo, hi, kk, m))
    })
}

/// The veil pairs' votes per object: for each veil pair, each object of the low-contrast
/// stem that is not of the high-contrast stem is seen through the layer, with the pair's
/// relation.
fn veil_votes(js: &[Junction]) -> HashMap<u32, Vec<(f64, DVec3)>> {
    let mut veil: HashMap<u32, Vec<(f64, DVec3)>> = HashMap::default();
    for (lo, hi, kk, m) in veil_pairs(js) {
        for &l in &lo.stem {
            if hi.stem.contains(&l) {
                continue;
            }
            veil.entry(l).or_default().push((kk, m));
        }
    }
    veil
}

/// The veil pairs' fronts per veiled object: the front of either T that is of neither stem,
/// where the contour runs on under it, so in front of both sides and the layer. Keyed by the
/// low-contrast stem's objects, as [`veil_votes`] is.
fn veil_fronts(js: &[Junction]) -> HashMap<u32, Vec<u32>> {
    let mut fronts: HashMap<u32, Vec<u32>> = HashMap::default();
    for (lo, hi, ..) in veil_pairs(js) {
        for f in [lo.front, hi.front] {
            if lo.stem.contains(&f) || hi.stem.contains(&f) {
                continue;
            }
            for &l in &lo.stem {
                if !hi.stem.contains(&l) {
                    fronts.entry(l).or_default().push(f);
                }
            }
        }
    }
    fronts
}

/// The veil pairs' bare objects per veiled object: the high-contrast stem's objects of
/// neither the low-contrast stem nor a front, the veiled objects seen bare. Keyed as
/// [`veil_votes`] is.
fn veil_bare(js: &[Junction]) -> HashMap<u32, Vec<u32>> {
    let mut bare: HashMap<u32, Vec<u32>> = HashMap::default();
    for (lo, hi, ..) in veil_pairs(js) {
        for &b in &hi.stem {
            if lo.stem.contains(&b) || b == lo.front || b == hi.front {
                continue;
            }
            for &l in &lo.stem {
                if !hi.stem.contains(&l) {
                    bare.entry(l).or_default().push(b);
                }
            }
        }
    }
    bare
}

/// A layer the veil pairs propose: its seeds (the objects seen through it, grouped where
/// they touch) and the pairs' relation.
#[derive(Clone, Debug)]
pub struct Proposal {
    pub k: f64,
    pub m: DVec3,
    pub mask: Vec<bool>,
    pub px: usize,
    /// the veil pairs that read the group
    pub votes: usize,
    /// the objects the pairs' contours run on under, between their two Ts: in front of the
    /// layer and the beneath both ([`veil_fronts`])
    pub fronts: Vec<bool>,
    /// the objects of the pairs' bare stems: the things the group's objects are the veiled
    /// image of, seen bare across the front ([`veil_bare`])
    pub bare: Vec<bool>,
}

impl Proposal {
    /// The layer's colour, `m` over `1 - k`.
    pub fn c(&self) -> DVec3 {
        (self.m / (1.0 - self.k)).clamp(DVec3::ZERO, DVec3::ONE)
    }
}

/// The layers the veil pairs propose on the objects ([`pieces`]): objects with veil votes
/// grouped where they touch, relation the median of the votes, largest first. A group needs
/// a second agreeing contour, as one reading is any two steps in that ratio.
pub fn proposals(img: &Image, pieces: &Pieces) -> Vec<Proposal> {
    let (w, h) = (pieces.w, pieces.h);
    let mut js = junctions(img, pieces);
    pair_junctions(&mut js, w);
    let veil = veil_votes(&js);
    let front_of = veil_fronts(&js);
    let bare_of = veil_bare(&js);
    let bounds = boundaries(&pieces.label, w, h);
    let mut parent: Vec<usize> = (0..pieces.n).collect();
    for &(a, b) in bounds.keys() {
        if veil.contains_key(&a) && veil.contains_key(&b) {
            let (ra, rb) = (find(&mut parent, a as usize), find(&mut parent, b as usize));
            if ra != rb {
                parent[ra] = rb;
            }
        }
    }
    let mut groups: HashMap<usize, Vec<(f64, DVec3)>> = HashMap::default();
    for (&p, votes) in &veil {
        groups
            .entry(find(&mut parent, p as usize))
            .or_default()
            .extend(votes.iter().copied());
    }
    let mut out: Vec<Proposal> = groups
        .into_iter()
        .filter(|(_, votes)| votes.len() >= 2)
        .map(|(root, votes)| {
            let median = |mut v: Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let k = median(votes.iter().map(|v| v.0).collect());
            let m = DVec3::new(
                median(votes.iter().map(|v| v.1.x).collect()),
                median(votes.iter().map(|v| v.1.y).collect()),
                median(votes.iter().map(|v| v.1.z).collect()),
            );
            let members: Vec<bool> = (0..pieces.n as u32)
                .map(|p| veil.contains_key(&p) && find(&mut parent, p as usize) == root)
                .collect();
            // the objects the members' lists name, per pixel
            let named = |of: &HashMap<u32, Vec<u32>>| -> Vec<bool> {
                let mut named = vec![false; pieces.n];
                for (p, list) in of {
                    if members[*p as usize] {
                        for &q in list {
                            named[q as usize] = true;
                        }
                    }
                }
                pieces.label.iter().map(|&p| named[p as usize]).collect()
            };
            let mask: Vec<bool> = pieces.label.iter().map(|&p| members[p as usize]).collect();
            let px = mask.iter().filter(|v| **v).count();
            Proposal {
                k,
                m,
                mask,
                px,
                votes: votes.len(),
                fronts: named(&front_of),
                bare: named(&bare_of),
            }
        })
        .collect();
    out.sort_by_key(|p| std::cmp::Reverse(p.px));
    out
}
