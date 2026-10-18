//! The region stage: a relation grown into regions by a frontier from the seed pixels, and
//! each verified on its own boundary.

use glam::{IVec2, Vec3};

use super::leaves::{Leaves, components};
use super::raster::neighbours4;
use super::{NOISE, SHOW_THROUGH, in_gamut, percentile};

/// A box of pixels, `min..max`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub min: IVec2,
    pub max: IVec2,
}

impl Bounds {
    fn width(&self) -> usize {
        (self.max.x - self.min.x) as usize
    }

    fn height(&self) -> usize {
        (self.max.y - self.min.y) as usize
    }

    /// This box grown by `pad` and clipped to the frame.
    fn grown(&self, pad: i32, width: usize, height: usize) -> Self {
        Bounds {
            min: (self.min - pad).max(IVec2::ZERO),
            max: (self.max + pad).min(IVec2::new(width as i32, height as i32)),
        }
    }

    /// The frame index of a box index.
    fn frame(&self, j: usize, width: usize) -> usize {
        let w = self.width();
        (self.min.y as usize + j / w) * width + self.min.x as usize + j % w
    }
}

/// The box of the set pixels of a frame mask.
fn bounds_of(mask: &[bool], width: usize) -> Option<Bounds> {
    let (min, max) = super::mask_bounds(mask, width);
    (min.x <= max.x).then_some(Bounds { min, max })
}

/// Squared Euclidean distance from every pixel of a `width` x `height` grid to the nearest
/// pixel where `feature` holds, infinite where there is none (Felzenszwalb and Huttenlocher's
/// lower envelope of parabolas, a row pass and a column pass).
pub fn distance_squared(width: usize, height: usize, feature: impl Fn(usize) -> bool) -> Vec<f32> {
    const FAR: f32 = 1e20;
    let mut d: Vec<f32> = (0..width * height)
        .map(|i| if feature(i) { 0. } else { FAR })
        .collect();
    let mut f = vec![0f32; width.max(height)];
    let mut out = vec![0f32; width.max(height)];
    let mut v = vec![0usize; width.max(height)];
    let mut z = vec![0f32; width.max(height) + 1];
    // The lower envelope of the parabolas rooted at each sample: `v` their roots, `z` where
    // each takes over
    let mut pass = |f: &[f32], n: usize, out: &mut [f32]| {
        let cross = |q: usize, p: usize| {
            ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2. * (q as f32 - p as f32))
        };
        let mut k = 0usize;
        v[0] = 0;
        z[0] = f32::NEG_INFINITY;
        z[1] = f32::INFINITY;
        for q in 1..n {
            let mut s = cross(q, v[k]);
            while s <= z[k] {
                k -= 1;
                s = cross(q, v[k]);
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f32::INFINITY;
        }
        k = 0;
        for (q, o) in out.iter_mut().enumerate().take(n) {
            while z[k + 1] < q as f32 {
                k += 1;
            }
            let dq = q as f32 - v[k] as f32;
            *o = dq * dq + f[v[k]];
        }
    };
    for y in 0..height {
        f[..width].copy_from_slice(&d[y * width..(y + 1) * width]);
        pass(&f, width, &mut out);
        d[y * width..(y + 1) * width].copy_from_slice(&out[..width]);
    }
    for x in 0..width {
        for y in 0..height {
            f[y] = d[y * width + x];
        }
        pass(&f, height, &mut out);
        for y in 0..height {
            d[y * width + x] = out[y].min(FAR);
        }
    }
    d
}

/// The gap a line drawn under or over a shape leaves in its mask, closed: the width the
/// leaves bridge, so strokes under a veil do not cut it into strips.
const CLOSE: f32 = 3.;

/// The shape's pixels for `o = k b + m` grown from its seed leaves where a pixel is nearer its
/// leaf's plane through the relation than the plane itself; gaps to [`CLOSE`] px closed, what it
/// encloses joined unless it shows a colour of a leaf reaching the open image (a hole).
pub fn grow_pixels(lv: &Leaves, target: &[Vec3], seeds: &[bool], k: f32, m: Vec3) -> Vec<bool> {
    let (width, height) = (lv.width, lv.height);
    let mut mask = vec![false; width * height];
    let mut frontier: Vec<usize> = Vec::new();
    for l in (0..lv.n).filter(|&l| seeds[l]) {
        for &i in lv.filled_of(l) {
            mask[i as usize] = true;
            frontier.push(i as usize);
        }
    }
    let candidate = |i: usize| -> bool {
        let f = lv.filled[i];
        if f >= 0 && seeds[f as usize] {
            return false;
        }
        let plain = lv.plain[i];
        let through = k * plain + m;
        let d_through = (target[i] - through).abs().max_element();
        d_through < (target[i] - plain).abs().max_element()
            && d_through <= 2. * NOISE
            && (through - plain).abs().max_element() > 2. * NOISE
    };
    while let Some(i) = frontier.pop() {
        for j in neighbours4(i, width, height) {
            if !mask[j] && candidate(j) {
                mask[j] = true;
                frontier.push(j);
            }
        }
    }
    let Some(bx) = bounds_of(&mask, width) else {
        return mask;
    };
    // The closing, a dilation and an erosion by a disc, in the box grown by one more than
    // its radius so the erosion sees nothing beyond (a closing never leaves the box of what it
    // closes)
    let pad = CLOSE as i32 + 1;
    let wide = Bounds {
        min: bx.min - pad,
        max: bx.max + pad,
    };
    let (ww, wh) = (wide.width(), wide.height());
    let inside = |j: usize| -> bool {
        let (u, v) = (wide.min.x + (j % ww) as i32, wide.min.y + (j / ww) as i32);
        u >= 0
            && v >= 0
            && (u as usize) < width
            && (v as usize) < height
            && mask[v as usize * width + u as usize]
    };
    let to_mask = distance_squared(ww, wh, inside);
    let to_open = distance_squared(ww, wh, |j| to_mask[j] > CLOSE * CLOSE);
    for v in bx.min.y..bx.max.y {
        for u in bx.min.x..bx.max.x {
            let j = (v - wide.min.y) as usize * ww + (u - wide.min.x) as usize;
            if to_open[j] > CLOSE * CLOSE {
                mask[v as usize * width + u as usize] = true;
            }
        }
    }
    // The holes: the parts of the box's complement that do not reach its edge
    let (bw, bh) = (bx.width(), bx.height());
    let frame = |j: usize| bx.frame(j, width);
    let (part, count) = components(bw, bh, |a, b| !mask[frame(a)] && !mask[frame(b)]);
    let mut open = vec![false; count];
    let mut sum = vec![Vec3::ZERO; count];
    let mut pixels = vec![0u32; count];
    let mut is_gap = vec![false; count];
    for j in 0..bw * bh {
        let p = part[j] as usize;
        if mask[frame(j)] {
            continue;
        }
        is_gap[p] = true;
        let (u, v) = (j % bw, j / bw);
        if u == 0 || v == 0 || u + 1 == bw || v + 1 == bh {
            open[p] = true;
        }
        sum[p] += target[frame(j)];
        pixels[p] += 1;
    }
    let holes: Vec<usize> = (0..count).filter(|&p| is_gap[p] && !open[p]).collect();
    if holes.is_empty() {
        return mask;
    }
    // The open leaves: those with a pixel neither grown nor in a hole
    let mut taken = vec![0u32; lv.n];
    let hole = {
        let mut h = vec![false; count];
        for &p in &holes {
            h[p] = true;
        }
        h
    };
    for j in 0..bw * bh {
        let i = frame(j);
        if (mask[i] || hole[part[j] as usize]) && lv.filled[i] >= 0 {
            taken[lv.filled[i] as usize] += 1;
        }
    }
    let open_colors: Vec<Vec3> = (0..lv.n)
        .filter(|&l| lv.filled_n[l] > taken[l])
        .map(|l| lv.color[l])
        .collect();
    if open_colors.is_empty() {
        return mask;
    }
    let mut fill = vec![false; count];
    for &p in &holes {
        let mean = sum[p] / pixels[p] as f32;
        fill[p] = !open_colors
            .iter()
            .any(|c| (*c - mean).abs().max_element() <= 2. * NOISE);
    }
    for j in 0..bw * bh {
        if fill[part[j] as usize] {
            mask[frame(j)] = true;
        }
    }
    mask
}

/// How far either side of a mask's boundary the pairs are taken, along its normal.
const PAIR_DISTANCE: f32 = 3.;
/// Fewest boundary pairs for a region to be judged, and fewest within the noise for its
/// relation to be refitted.
const MIN_PAIRS: usize = 16;
const MIN_REFIT: usize = 8;

/// A region a relation grew, judged on its own boundary: its relation refitted there, the
/// pairs that fit and all of them, the spread of the backgrounds that fit, its pixels, and the
/// three rules.
#[derive(Clone, Debug)]
pub struct Region {
    pub k: f32,
    pub m: Vec3,
    pub fitting: usize,
    pub pairs: usize,
    pub spread: f32,
    pub area: usize,
    pub share: bool,
    pub show: bool,
    pub interior: bool,
    /// The region's pixels in its box.
    pub bounds: Bounds,
    pub bits: Vec<bool>,
    /// Its boundary pairs, `(b, o)`, and the squared distance of its deepest pixel from its
    /// outside, for the merge phase.
    pub own: Vec<(Vec3, Vec3)>,
    pub depth: f32,
}

impl Region {
    pub fn kept(&self) -> bool {
        self.share && self.show && self.interior
    }
}

/// The connected parts a relation grows (`pair_fits`), each refitted over its boundary pairs and
/// kept where most pairs fit, their backgrounds spread past the noise and it has an interior;
/// with `shade` only the scale of a black or white relation is refitted.
pub fn regions_of(
    lv: &Leaves,
    target: &[Vec3],
    k: f32,
    m: Vec3,
    shade: Option<super::discovery::Shade>,
) -> Vec<Region> {
    let fits = lv.pair_fits(k, m);
    if !fits.contains(&true) {
        return Vec::new();
    }
    let mut seeds = vec![false; lv.n];
    for (p, &f) in lv.pairs.iter().zip(&fits) {
        if f {
            seeds[p.x as usize] = true;
        }
    }
    let mask = grow_pixels(lv, target, &seeds, k, m);
    judge(lv, target, &mask, k, m, shade, false)
}

/// The parts of a mask judged on their own boundaries, as [`regions_of`] does, or with
/// `whole` the mask as one region.
pub fn judge(
    lv: &Leaves,
    target: &[Vec3],
    mask: &[bool],
    k: f32,
    m: Vec3,
    shade: Option<super::discovery::Shade>,
    whole: bool,
) -> Vec<Region> {
    let (width, height) = (lv.width, lv.height);
    let Some(bx) = bounds_of(mask, width) else {
        return Vec::new();
    };
    // The parts, and the pairs, in the box grown by the pair distance plus one: every sample
    // lies within it, and each pixel's nearest pixel off the mask in the box grown by one
    let bx = bx.grown(PAIR_DISTANCE as i32 + 1, width, height);
    let (bw, bh) = (bx.width(), bx.height());
    let frame = |j: usize| bx.frame(j, width);
    let mk: Vec<bool> = (0..bw * bh).map(|j| mask[frame(j)]).collect();
    let (part, _) = components(bw, bh, |a, b| mk[a] && mk[b] || whole);
    // Parts of the mask numbered from one in the order of their first pixel, 0 off it
    let mut ids = vec![0u32; bw * bh];
    let mut remap: Vec<u32> = Vec::new();
    let mut map = crate::HashMap::default();
    for j in 0..bw * bh {
        if mk[j] {
            let next = remap.len() as u32 + 1;
            let id = *map.entry(part[j]).or_insert_with(|| {
                remap.push(part[j]);
                next
            });
            ids[j] = id;
        }
    }
    let count = remap.len();
    if count == 0 {
        return Vec::new();
    }
    let dist_in = distance_squared(bw, bh, |j| !mk[j]);
    let dist_out = distance_squared(bw, bh, |j| mk[j]);
    let sdf: Vec<f32> = (0..bw * bh)
        .map(|j| {
            if mk[j] {
                dist_in[j].sqrt()
            } else {
                -dist_out[j].sqrt()
            }
        })
        .collect();
    // The gradient by central differences, one-sided at the box's edge
    let grad = |u: usize, v: usize| -> (f32, f32) {
        let at = |u: usize, v: usize| sdf[v * bw + u];
        let gx = if bw < 2 {
            0.
        } else if u == 0 {
            at(1, v) - at(0, v)
        } else if u + 1 == bw {
            at(u, v) - at(u - 1, v)
        } else {
            (at(u + 1, v) - at(u - 1, v)) / 2.
        };
        let gy = if bh < 2 {
            0.
        } else if v == 0 {
            at(u, 1) - at(u, 0)
        } else if v + 1 == bh {
            at(u, v) - at(u, v - 1)
        } else {
            (at(u, v + 1) - at(u, v - 1)) / 2.
        };
        (gx, gy)
    };
    let mut pairs: Vec<(u32, Vec3, Vec3)> = Vec::new();
    for v in 0..bh {
        for u in 0..bw {
            let j = v * bw + u;
            if !mk[j] {
                continue;
            }
            let boundary = u == 0
                || v == 0
                || u + 1 == bw
                || v + 1 == bh
                || !mk[j - 1]
                || !mk[j + 1]
                || !mk[j - bw]
                || !mk[j + bw];
            if !boundary {
                continue;
            }
            let (gx, gy) = grad(u, v);
            let norm = (gx * gx + gy * gy).sqrt();
            if norm <= 1e-3 {
                continue;
            }
            let (nx, ny) = (gx / norm, gy / norm);
            let (x, y) = (
                (bx.min.x as usize + u) as f32,
                (bx.min.y as usize + v) as f32,
            );
            let clip = |t: f32, n: usize| (t.round_ties_even().max(0.) as usize).min(n - 1);
            let (ix, iy) = (
                clip(x + PAIR_DISTANCE * nx, width),
                clip(y + PAIR_DISTANCE * ny, height),
            );
            let (ox, oy) = (
                clip(x - PAIR_DISTANCE * nx, width),
                clip(y - PAIR_DISTANCE * ny, height),
            );
            let local =
                |px: usize, py: usize| (py - bx.min.y as usize) * bw + (px - bx.min.x as usize);
            if ids[local(ix, iy)] != ids[j] || mk[local(ox, oy)] {
                continue;
            }
            let o = target[iy * width + ix];
            let out = oy * width + ox;
            // The outside sample carried to the inside one by its leaf's slope: on a gradient
            // the background 2d px away is not the one beneath
            let mut b = target[out];
            if lv.filled[out] >= 0 {
                b += lv.plane[lv.filled[out] as usize].step(glam::Vec2::new(
                    ix as f32 - ox as f32,
                    iy as f32 - oy as f32,
                ));
            }
            if (o - b).abs().max_element() > 2. * NOISE {
                pairs.push((ids[j], b, o));
            }
        }
    }
    let mut deepest = vec![0f32; count + 1];
    let mut area = vec![0usize; count + 1];
    for j in 0..bw * bh {
        let id = ids[j] as usize;
        if id > 0 {
            deepest[id] = deepest[id].max(dist_in[j]);
            area[id] += 1;
        }
    }
    let tol = 3. * NOISE;
    let mut out = Vec::new();
    for id in 1..=count {
        let own: Vec<(Vec3, Vec3)> = pairs
            .iter()
            .filter(|p| p.0 as usize == id)
            .map(|p| (p.1, p.2))
            .collect();
        if own.len() < MIN_PAIRS {
            continue;
        }
        let (mut kk, mut mm) = (k, m);
        let sel: Vec<&(Vec3, Vec3)> = own
            .iter()
            .filter(|(b, o)| (*o - (k * *b + m)).abs().max_element() <= tol)
            .collect();
        if sel.len() >= MIN_REFIT {
            let n = sel.len() as f32;
            if let Some(kind) = shade {
                // A shading layer's scale through the origin in its own space
                let (num, den) = sel.iter().fold((0f32, 0f32), |(a, d), (b, o)| {
                    let (tb, to) = (kind.space(*b), kind.space(*o));
                    (a + tb.dot(to), d + tb.length_squared())
                });
                if den > 1e-9 {
                    let k2 = num / den;
                    if (0.02..=0.98).contains(&k2) {
                        kk = k2;
                        mm = (1. - k2) * kind.color();
                    }
                }
            } else {
                let mb = sel.iter().map(|p| p.0).sum::<Vec3>() / n;
                let mo = sel.iter().map(|p| p.1).sum::<Vec3>() / n;
                let den: f32 = sel.iter().map(|p| (p.0 - mb).length_squared()).sum();
                if den > 1e-9 {
                    let k2 = sel.iter().map(|p| (p.0 - mb).dot(p.1 - mo)).sum::<f32>() / den;
                    if (0.02..=0.98).contains(&k2) && in_gamut((mo - k2 * mb) / (1. - k2)) {
                        kk = k2;
                        mm = mo - k2 * mb;
                    }
                }
            }
        }
        let fit: Vec<Vec3> = own
            .iter()
            .filter(|(b, o)| (*o - (kk * *b + mm)).abs().max_element() <= tol)
            .map(|p| p.0)
            .collect();
        let spread = if fit.len() >= 2 {
            (0..3)
                .map(|c| {
                    let mut v: Vec<f32> = fit.iter().map(|b| b[c]).collect();
                    percentile(&mut v, 95.) - percentile(&mut v, 5.)
                })
                .fold(f32::NEG_INFINITY, f32::max)
        } else {
            0.
        };
        let bits: Vec<bool> = ids.iter().map(|&i| i as usize == id).collect();
        out.push(Region {
            k: kk,
            m: mm,
            fitting: fit.len(),
            pairs: own.len(),
            spread,
            area: area[id],
            share: fit.len() * 2 > own.len(),
            show: kk * spread > SHOW_THROUGH,
            interior: deepest[id] >= 4.,
            bounds: bx,
            bits,
            own,
            depth: deepest[id],
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances_match_brute_force() {
        let (width, height) = (23, 17);
        let feature = |i: usize| (i * 7919).is_multiple_of(31) || i == 100;
        let d = distance_squared(width, height, feature);
        for (i, &di) in d.iter().enumerate() {
            let (x, y) = ((i % width) as i64, (i / width) as i64);
            let best = (0..width * height)
                .filter(|&j| feature(j))
                .map(|j| {
                    let (u, v) = ((j % width) as i64, (j / width) as i64);
                    ((u - x).pow(2) + (v - y).pow(2)) as f32
                })
                .fold(f32::INFINITY, f32::min);
            assert_eq!(di, best, "pixel {x}, {y}");
        }
    }
}
