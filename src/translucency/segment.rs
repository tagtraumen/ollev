//! Two segmentations on scikit-image's Felzenszwalb: the atoms a layer's region is voted
//! over, and the planar regions of the region proposal (a finer segmentation merged greedily
//! while one plane fits the union, so a gradient stays one region).

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap};

use super::raster::{Image, NOISE, area_px, gaussian_filter, np_sum, percentile};

/// scikit-image 0.26's `felzenszwalb(img, scale, sigma, min_size)` label for label (internal
/// costs in single precision, as the Cython's `float`): a label per pixel, 0-based, numbered
/// in raster order of each segment's first pixel.
pub fn felzenszwalb(img: &Image, scale: f64, sigma: f64, min_size: usize) -> Vec<u32> {
    let (w, h) = (img.width, img.height);
    let n = w * h;
    let scale = scale / 255.0;
    let sm: Vec<Vec<f64>> = (0..3)
        .map(|c| gaussian_filter(&img.channel(c), w, h, sigma, 0, 0))
        .collect();
    let cost = |p: usize, q: usize| {
        let d0 = sm[0][p] - sm[0][q];
        let d1 = sm[1][p] - sm[1][q];
        let d2 = sm[2][p] - sm[2][q];
        (d0 * d0 + d1 * d1 + d2 * d2).sqrt()
    };
    // the edges in scikit-image's order: right, down, down-right, up-right, each (later
    // pixel, earlier pixel) as its index arrays pair them
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(4 * n);
    for y in 0..h {
        for x in 0..w.saturating_sub(1) {
            let p = y * w + x;
            edges.push(((p + 1) as u32, p as u32));
        }
    }
    for y in 0..h.saturating_sub(1) {
        for x in 0..w {
            let p = y * w + x;
            edges.push(((p + w) as u32, p as u32));
        }
    }
    for y in 0..h.saturating_sub(1) {
        for x in 0..w.saturating_sub(1) {
            let p = y * w + x;
            edges.push(((p + w + 1) as u32, p as u32));
        }
    }
    for y in 0..h.saturating_sub(1) {
        for x in 0..w.saturating_sub(1) {
            let p = y * w + x;
            edges.push(((p + 1) as u32, (p + w) as u32));
        }
    }
    let costs: Vec<f64> = edges
        .iter()
        .map(|&(a, b)| cost(a as usize, b as usize))
        .collect();
    // by cost, ties by index (a stable sort): a cost is a norm, never negative, so its bits
    // order as it does
    let mut keyed: Vec<(u64, u32)> = (0..edges.len() as u32)
        .map(|e| (costs[e as usize].to_bits(), e))
        .collect();
    keyed.sort_unstable();
    let order: Vec<u32> = keyed.into_iter().map(|(_, e)| e).collect();

    // a union-find forest whose root is always the smallest pixel of its tree
    let mut forest: Vec<u32> = (0..n as u32).collect();
    fn find(forest: &mut [u32], n: u32) -> u32 {
        let mut root = n;
        while forest[root as usize] < root {
            root = forest[root as usize];
        }
        let mut m = n;
        while forest[m as usize] < m {
            let j = forest[m as usize];
            forest[m as usize] = root;
            m = j;
        }
        root
    }
    let mut size = vec![1usize; n];
    let mut cint = vec![0.0f64; n];
    for &e in &order {
        let (a, b) = edges[e as usize];
        let (s0, s1) = (find(&mut forest, a), find(&mut forest, b));
        if s0 == s1 {
            continue;
        }
        let c = costs[e as usize];
        let i0 = (cint[s0 as usize] + scale / size[s0 as usize] as f64) as f32;
        let i1 = (cint[s1 as usize] + scale / size[s1 as usize] as f64) as f32;
        if c < i0.min(i1) as f64 {
            let root = s0.min(s1);
            forest[s0.max(s1) as usize] = root;
            size[root as usize] = size[s0 as usize] + size[s1 as usize];
            cint[root as usize] = c;
        }
    }
    for &e in &order {
        let (a, b) = edges[e as usize];
        let (s0, s1) = (find(&mut forest, a), find(&mut forest, b));
        if s0 == s1 {
            continue;
        }
        if size[s0 as usize] < min_size || size[s1 as usize] < min_size {
            let root = s0.min(s1);
            forest[s0.max(s1) as usize] = root;
            size[root as usize] = size[s0 as usize] + size[s1 as usize];
        }
    }
    // labels by root, the roots in increasing order (numpy's unique with its inverse)
    let mut label = vec![u32::MAX; n];
    let mut next = 0u32;
    (0..n)
        .map(|p| {
            let r = find(&mut forest, p as u32) as usize;
            if label[r] == u32::MAX {
                label[r] = next;
                next += 1;
            }
            label[r]
        })
        .collect()
}

/// A color plane `a + G (x, y)`: per channel the constant and its slopes in x and y.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Plane {
    a: [f64; 3],
    g: [[f64; 2]; 3],
}

impl Plane {
    /// The largest channel difference of `col` from the plane at (x, y).
    fn off(&self, x: f64, y: f64, col: [f64; 3]) -> f64 {
        let mut d: f64 = 0.0;
        for ((a, g), o) in self.a.iter().zip(&self.g).zip(col) {
            let pred = a + (x * g[0] + y * g[1]);
            let v = (o - pred).abs();
            if v > d || v.is_nan() {
                d = v;
            }
        }
        d
    }
}

/// numpy's `det` of a 2 x 2 matrix: LAPACK's LU with partial pivoting, the determinant
/// from the factored diagonal as `sign * exp(sum of log |u|)` (numpy's slogdet route).
/// Also returns the factors for [`solve2`].
fn lu2(m: [[f64; 2]; 2]) -> ([[f64; 2]; 2], bool, f64, f64) {
    let swap = m[1][0].abs() > m[0][0].abs();
    let (r0, r1) = if swap { (m[1], m[0]) } else { (m[0], m[1]) };
    let l = if r0[0] != 0.0 {
        r1[0] * (1.0 / r0[0])
    } else {
        0.0
    };
    let u11 = r1[1] - l * r0[1];
    let lu = [[r0[0], r0[1]], [l, u11]];
    let det = if r0[0] == 0.0 || u11 == 0.0 {
        0.0
    } else {
        let mut sign = if swap { -1.0 } else { 1.0 };
        let mut logdet = 0.0;
        for u in [r0[0], u11] {
            if u < 0.0 {
                sign = -sign;
            }
            logdet += u.abs().ln();
        }
        sign * logdet.exp()
    };
    (lu, swap, det, l)
}

/// numpy's `solve` of a 2 x 2 system for one right-hand side, from [`lu2`]'s factors.
fn solve2(lu: &[[f64; 2]; 2], swap: bool, b: [f64; 2]) -> [f64; 2] {
    let (b0, b1) = if swap { (b[1], b[0]) } else { (b[0], b[1]) };
    let y1 = b1 - lu[1][0] * b0;
    let x1 = y1 / lu[1][1];
    let x0 = (b0 - lu[0][1] * x1) / lu[0][0];
    [x0, x1]
}

/// The weighted least squares plane over pixels by the normal
/// equations on centered coordinates (no slope across collinear pixels), and the 95th
/// percentile of the largest channel residual.
fn plane_fit(pts: &[(f64, f64)], cols: &[[f64; 3]], w: Option<&[f64]>) -> (Plane, f64) {
    let ones;
    let w = match w {
        Some(w) => w,
        None => {
            ones = vec![1.0; pts.len()];
            &ones
        }
    };
    let wsum = np_sum(w);
    let (mut mx, mut my) = (0.0, 0.0);
    for (p, wi) in pts.iter().zip(w) {
        mx += wi * p.0;
        my += wi * p.1;
    }
    let (mx, my) = (mx / wsum, my / wsum);
    let mut cm = [0.0; 3];
    for (c, wi) in cols.iter().zip(w) {
        for ch in 0..3 {
            cm[ch] += wi * c[ch];
        }
    }
    for v in cm.iter_mut() {
        *v /= wsum;
    }
    let mut s = [[0.0f64; 2]; 2];
    let mut rhs = [[0.0f64; 3]; 2];
    for ((p, c), wi) in pts.iter().zip(cols).zip(w) {
        let d = [p.0 - mx, p.1 - my];
        let dw = [d[0] * wi, d[1] * wi];
        for i in 0..2 {
            for j in 0..2 {
                s[i][j] += dw[i] * d[j];
            }
            for ch in 0..3 {
                rhs[i][ch] += dw[i] * (c[ch] - cm[ch]);
            }
        }
    }
    let (lu, swap, det, _) = lu2(s);
    let tr = s[0][0] + s[1][1];
    let mut g = [[0.0f64; 2]; 3];
    if det.abs() >= 1e-9 * tr.max(1.0).powi(2) {
        for ch in 0..3 {
            let x = solve2(&lu, swap, [rhs[0][ch], rhs[1][ch]]);
            g[ch] = x;
        }
    }
    let mut a = [0.0; 3];
    for ch in 0..3 {
        a[ch] = cm[ch] - (g[ch][0] * mx + g[ch][1] * my);
    }
    let plane = Plane { a, g };
    let mut res: Vec<f64> = pts
        .iter()
        .zip(cols)
        .map(|(p, c)| plane.off(p.0, p.1, *c))
        .collect();
    (plane, percentile(&mut res, 95.0))
}

/// A region while merging: its pixels in merge order, its exact moment sums, its plane and
/// where the plane comes from.
struct Region {
    pixels: Vec<u32>,
    /// n, x, y, xx, xy, yy, then per channel c, xc, yc, in levels: exact integers.
    mom: [i128; 15],
    plane: Plane,
    from_sums: bool,
    /// the fit over the pixels, while the region has not grown since
    fitted: Option<Plane>,
    grown: u32,
    sample: Option<(u32, Vec<u32>, usize)>,
    alive: bool,
    neighbors: BTreeSet<u32>,
}

/// The tolerance the equation regions merge to: 3 levels.
pub const EQ_TOL: f64 = 3.0 * NOISE;
/// The tests are percentiles, so a strided sample of a large region decides them as well
/// (so a large sky is not tested whole against each neighbor).
const EQ_SAMPLE: usize = 4000;

#[derive(Clone, Copy, PartialEq)]
struct Ord64(f64);
impl Eq for Ord64 {}
impl PartialOrd for Ord64 {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Ord64 {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Planar regions: Felzenszwalb, then neighbors merged cheapest first while each plane
/// predicts the other's sample within `tol` at the 90th percentile and the joint plane's
/// 95th percentile residual is within it. Returns the surviving Felzenszwalb label plus one.
pub fn equation_regions(img: &Image, scale: f64, min_size: usize, tol: f64) -> Vec<u32> {
    let (w, h) = (img.width, img.height);
    let lab = felzenszwalb(img, scale, 0.5, min_size);
    let n = lab.iter().max().map_or(0, |&l| l as usize + 1);
    let lv: Vec<[i128; 3]> = img
        .px
        .iter()
        .map(|p| {
            [
                (p.x * 255.0).round_ties_even() as i128,
                (p.y * 255.0).round_ties_even() as i128,
                (p.z * 255.0).round_ties_even() as i128,
            ]
        })
        .collect();
    let exact = img.px.iter().zip(&lv).all(|(p, l)| {
        l[0] as f64 / 255.0 == p.x && l[1] as f64 / 255.0 == p.y && l[2] as f64 / 255.0 == p.z
    });
    assert!(
        exact,
        "the lab's equation regions are ported for 8-bit images"
    );
    let at = |i: u32| ((i as usize % w) as f64, (i as usize / w) as f64);
    let col = |i: u32| {
        let p = img.px[i as usize];
        [p.x, p.y, p.z]
    };
    let mut regions: Vec<Region> = (0..n)
        .map(|_| Region {
            pixels: Vec::new(),
            mom: [0; 15],
            plane: Plane::default(),
            from_sums: false,
            fitted: None,
            grown: 0,
            sample: None,
            alive: true,
            neighbors: BTreeSet::new(),
        })
        .collect();
    for (i, &l) in lab.iter().enumerate() {
        let r = &mut regions[l as usize];
        r.pixels.push(i as u32);
        let (x, y) = ((i % w) as i128, (i / w) as i128);
        let c = lv[i];
        let m = &mut r.mom;
        m[0] += 1;
        m[1] += x;
        m[2] += y;
        m[3] += x * x;
        m[4] += x * y;
        m[5] += y * y;
        for ch in 0..3 {
            m[6 + ch] += c[ch];
            m[9 + ch] += x * c[ch];
            m[12 + ch] += y * c[ch];
        }
    }
    // a plane fitted over all the pixels (the mean below 12 pixels)
    let fit_all = |pixels: &[u32]| -> Plane {
        let pts: Vec<(f64, f64)> = pixels.iter().map(|&i| at(i)).collect();
        let cols: Vec<[f64; 3]> = pixels.iter().map(|&i| col(i)).collect();
        if pixels.len() >= 12 {
            plane_fit(&pts, &cols, None).0
        } else {
            // cols.mean(0): the rows summed in order over their count
            let mut a = [0.0; 3];
            for c in &cols {
                for ch in 0..3 {
                    a[ch] += c[ch];
                }
            }
            for v in a.iter_mut() {
                *v /= cols.len() as f64;
            }
            Plane {
                a,
                g: [[0.0; 2]; 3],
            }
        }
    };
    for r in regions.iter_mut() {
        r.plane = fit_all(&r.pixels);
    }
    // the pairs across each horizontal, then each vertical, pixel edge, as (low, high) in
    // the order of their first edge
    let mut adj: Vec<(u32, u32)> = Vec::new();
    {
        let mut seen = crate::HashSet::default();
        for (dy, dx) in [(0usize, 1usize), (1, 0)] {
            for y in 0..h - dy {
                for x in 0..w - dx {
                    let (a, b) = (lab[y * w + x], lab[(y + dy) * w + x + dx]);
                    if a != b {
                        let key = (a.min(b), a.max(b));
                        if seen.insert(key) {
                            adj.push(key);
                        }
                    }
                }
            }
        }
    }
    for &(a, b) in &adj {
        regions[a as usize].neighbors.insert(b);
        regions[b as usize].neighbors.insert(a);
    }

    // the plane from the moment sums, and whether it is safe from rounding (else the fit
    // over the pixels decides, where it is badly conditioned)
    let plane_of = |s: &[i128; 15]| -> (Plane, bool) {
        let n_ = s[0];
        let nf = n_ as f64;
        let (sx, sy, sxx, sxy, syy) = (s[1], s[2], s[3], s[4], s[5]);
        let mut mc = [0.0; 3];
        for ch in 0..3 {
            mc[ch] = (s[6 + ch] as f64 / nf) / 255.0;
        }
        if n_ < 12 {
            return (
                Plane {
                    a: mc,
                    g: [[0.0; 2]; 3],
                },
                false,
            );
        }
        let cxx = (n_ * sxx - sx * sx) as f64;
        let cxy = (n_ * sxy - sx * sy) as f64;
        let cyy = (n_ * syy - sy * sy) as f64;
        let sm = [[cxx / nf, cxy / nf], [cxy / nf, cyy / nf]];
        let (mean_x, mean_y) = (sx as f64 / nf, sy as f64 / nf);
        let (lu, swap, det, _) = lu2(sm);
        let tr = sm[0][0] + sm[1][1];
        let ratio = det.abs() / (1e-9 * tr.max(1.0).powi(2));
        if ratio < 1.0 {
            return (
                Plane {
                    a: mc,
                    g: [[0.0; 2]; 3],
                },
                ratio < 0.99,
            );
        }
        let mut g = [[0.0; 2]; 3];
        for ch in 0..3 {
            let rx = ((n_ * s[9 + ch] - sx * s[6 + ch]) as f64 / nf) / 255.0;
            let ry = ((n_ * s[12 + ch] - sy * s[6 + ch]) as f64 / nf) / 255.0;
            g[ch] = solve2(&lu, swap, [rx, ry]);
        }
        let mut a = [0.0; 3];
        for ch in 0..3 {
            a[ch] = mc[ch] - (g[ch][0] * mean_x + g[ch][1] * mean_y);
        }
        (Plane { a, g }, ratio > 1.01 && tr * tr <= 1e6 * det.abs())
    };

    fn sample_of(r: &mut Region) -> (Vec<u32>, usize) {
        if let Some((g, s, step)) = &r.sample
            && *g == r.grown
        {
            return (s.clone(), *step);
        }
        // every step-th pixel of the pieces as if joined
        let step = (r.pixels.len() / EQ_SAMPLE).max(1);
        let s: Vec<u32> = r.pixels.iter().step_by(step).copied().collect();
        r.sample = Some((r.grown, s.clone(), step));
        (s, step)
    }

    // whether region i's plane predicts the sample beyond the tolerance at the 90th
    // percentile, asking the fit over its pixels when the sums' plane is within 1e-7 of it
    let outside_tol = |regions: &mut [Region], i: usize, sample: &[u32]| -> bool {
        let plane = regions[i].plane;
        let mut d: Vec<f64> = sample
            .iter()
            .map(|&p| {
                let (x, y) = at(p);
                plane.off(x, y, col(p))
            })
            .collect();
        let mut pct = percentile(&mut d, 90.0);
        if regions[i].from_sums && (pct - tol).abs() <= 1e-7 {
            let fit = match regions[i].fitted {
                Some(f) => f,
                None => {
                    let f = fit_all(&regions[i].pixels);
                    regions[i].fitted = Some(f);
                    f
                }
            };
            let mut d: Vec<f64> = sample
                .iter()
                .map(|&p| {
                    let (x, y) = at(p);
                    fit.off(x, y, col(p))
                })
                .collect();
            pct = percentile(&mut d, 90.0);
        }
        pct > tol
    };

    let mergeable = |regions: &mut [Region], a: usize, b: usize| -> Option<f64> {
        let (sa, stepa) = sample_of(&mut regions[a]);
        let (sb, stepb) = sample_of(&mut regions[b]);
        if outside_tol(regions, a, &sb) || outside_tol(regions, b, &sa) {
            return None;
        }
        let pts: Vec<(f64, f64)> = sa.iter().chain(&sb).map(|&p| at(p)).collect();
        let cols: Vec<[f64; 3]> = sa.iter().chain(&sb).map(|&p| col(p)).collect();
        let wts: Vec<f64> = std::iter::repeat_n(stepa as f64, sa.len())
            .chain(std::iter::repeat_n(stepb as f64, sb.len()))
            .collect();
        let (_, res) = plane_fit(&pts, &cols, Some(&wts));
        (res <= tol).then_some(res)
    };

    // a merge test on the heap: (joint residual, the absorbing region, the
    // other, how many times each had grown when it was tested), the least first
    type Test = Reverse<(Ord64, u32, u32, u32, u32)>;
    let mut heap: BinaryHeap<Test> = BinaryHeap::new();
    for &(a, b) in &adj {
        if let Some(r) = mergeable(&mut regions, a as usize, b as usize) {
            heap.push(Reverse((Ord64(r), a, b, 0, 0)));
        }
    }
    while let Some(Reverse((_, a, b, va, vb))) = heap.pop() {
        let (a, b) = (a as usize, b as usize);
        if !(regions[a].alive && regions[b].alive) {
            continue;
        }
        if (va, vb) != (regions[a].grown, regions[b].grown)
            && mergeable(&mut regions, a, b).is_none()
        {
            continue;
        }
        regions[a].grown += 1;
        let taken = std::mem::take(&mut regions[b].pixels);
        regions[a].pixels.extend(taken);
        let mb = regions[b].mom;
        for (x, y) in regions[a].mom.iter_mut().zip(mb) {
            *x += y;
        }
        let (p, safe) = plane_of(&regions[a].mom);
        regions[a].fitted = None;
        if safe {
            regions[a].plane = p;
            regions[a].from_sums = true;
        } else {
            let f = fit_all(&regions[a].pixels);
            regions[a].fitted = Some(f);
            regions[a].plane = f;
            regions[a].from_sums = false;
        }
        regions[b].alive = false;
        let nbs: Vec<u32> = std::mem::take(&mut regions[b].neighbors)
            .into_iter()
            .collect();
        for nb in nbs {
            let nbu = nb as usize;
            if nbu != a && regions[nbu].alive {
                regions[a].neighbors.insert(nb);
                regions[nbu].neighbors.remove(&(b as u32));
                regions[nbu].neighbors.insert(a as u32);
            }
        }
        let nbs: Vec<u32> = regions[a].neighbors.iter().copied().collect();
        for nb in nbs {
            let nbu = nb as usize;
            if regions[nbu].alive
                && let Some(r2) = mergeable(&mut regions, a, nbu)
            {
                heap.push(Reverse((
                    Ord64(r2),
                    a as u32,
                    nb,
                    regions[a].grown,
                    regions[nbu].grown,
                )));
            }
        }
    }
    let mut out = vec![0u32; w * h];
    for (i, r) in regions.iter().enumerate() {
        if r.alive {
            for &p in &r.pixels {
                out[p as usize] = i as u32 + 1;
            }
        }
    }
    out
}

/// The equation regions at scale 40, 3 levels, and a least size that is an area of the
/// drawing (16 px at 330 by the unit squared, so the region count holds across resolutions).
/// The scale (in levels) and the smoothing stay fixed: the smoothing is against the noise.
pub fn equation_regions_default(img: &Image) -> Vec<u32> {
    equation_regions(img, 40.0, area_px(img.width, 16.0) as usize, EQ_TOL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translucency::raster::DVec3;

    fn image(w: usize, h: usize, f: impl Fn(usize, usize) -> DVec3) -> Image {
        let px = (0..w * h).map(|i| f(i % w, i / w)).collect();
        Image {
            width: w,
            height: h,
            px,
        }
    }

    #[test]
    fn two_flat_halves_as_scikit_image() {
        let img = image(20, 10, |x, _| {
            if x < 10 {
                DVec3::splat(0.2)
            } else {
                DVec3::new(0.9, 0.1, 0.1)
            }
        });
        let lab = felzenszwalb(&img, 80.0, 0.5, 10);
        // scikit-image 0.26 on the same image: the two blurred columns at the edge are
        // segments of their own (each of 10 px, the minimum size)
        let want = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3];
        for y in 0..10 {
            assert_eq!(&lab[y * 20..y * 20 + 20], &want);
        }
    }

    #[test]
    fn a_ramp_as_the_lab_merges_it() {
        // four levels per pixel beside a flat block: the ramp merges in runs, the steps
        // being beyond the tolerance of a plane per piece
        let img = image(40, 20, |x, _| {
            if x < 30 {
                DVec3::splat((40 + 4 * x) as f64 / 255.0)
            } else {
                DVec3::new(0.0, 0.0, 1.0)
            }
        });
        let eq = equation_regions_default(&img);
        let want = [
            1, 1, 3, 8, 8, 8, 8, 8, 8, 10, 13, 13, 13, 13, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        ];
        assert_eq!(&eq[..40], &want);
    }

    #[test]
    fn determinant_by_the_log_route() {
        let (_, swap, det, _) = lu2([[1.0, 2.0], [3.0, 4.0]]);
        assert!(swap);
        assert!((det + 2.0).abs() < 1e-14);
    }
}
