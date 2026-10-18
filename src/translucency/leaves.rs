//! The leaves the translucency pipeline works on, each with a planar color equation, and the
//! contact map between them: per directed pair of leaves that touch, points along the contact
//! where the two sides' colors are compared.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

use glam::{DVec2, DVec3, IVec3, Vec2, Vec3};

use super::{NOISE, SLOPE_MIN, percentile};

/// A color that varies linearly over the image: `a + gx x + gy y` at pixel `(x, y)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Plane {
    pub a: Vec3,
    pub gx: Vec3,
    pub gy: Vec3,
}

impl Plane {
    pub fn at(&self, p: Vec2) -> Vec3 {
        self.a + self.gx * p.x + self.gy * p.y
    }

    /// How steep the plane is, over its three channels and two directions.
    pub fn slope(&self) -> f32 {
        (self.gx.length_squared() + self.gy.length_squared()).sqrt()
    }

    /// How far it changes color between two points, per channel.
    pub fn step(&self, d: Vec2) -> Vec3 {
        self.gx * d.x + self.gy * d.y
    }
}

/// Whether the inside's color slope is the outside's times `k`, in the same direction:
/// beneath a translucent shape a gradient stays the gradient scaled by `k`, and a flat region
/// cannot be a gradient seen through one (the equation graph's constraint).
pub fn slope_fits(inside: &Plane, outside: &Plane, k: f32) -> bool {
    let diff = ((inside.gx - k * outside.gx).length_squared()
        + (inside.gy - k * outside.gy).length_squared())
    .sqrt();
    diff <= 0.3 * inside.slope().max(k * outside.slope()) + SLOPE_MIN
}

/// Sums over pixels from which their least squares plane follows.
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    n: f64,
    x: f64,
    y: f64,
    xx: f64,
    xy: f64,
    yy: f64,
    c: DVec3,
    xc: DVec3,
    yc: DVec3,
}

impl Moments {
    fn add(&mut self, x: f64, y: f64, c: DVec3, w: f64) {
        self.n += w;
        self.x += w * x;
        self.y += w * y;
        self.xx += w * x * x;
        self.xy += w * x * y;
        self.yy += w * y * y;
        self.c += w * c;
        self.xc += w * x * c;
        self.yc += w * y * c;
    }

    fn merge(&mut self, o: &Self) {
        self.n += o.n;
        self.x += o.x;
        self.y += o.y;
        self.xx += o.xx;
        self.xy += o.xy;
        self.yy += o.yy;
        self.c += o.c;
        self.xc += o.xc;
        self.yc += o.yc;
    }

    fn mean(&self) -> DVec3 {
        self.c / self.n.max(f64::MIN_POSITIVE)
    }

    /// The scatter of the positions about their mean, `[sxx, sxy, syy]`, and the mean.
    fn scatter(&self) -> ([f64; 3], DVec2) {
        let n = self.n.max(f64::MIN_POSITIVE);
        let (mx, my) = (self.x / n, self.y / n);
        (
            [
                self.xx - n * mx * mx,
                self.xy - n * mx * my,
                self.yy - n * my * my,
            ],
            DVec2::new(mx, my),
        )
    }

    /// The least squares plane, or the mean where `planar` says the scatter of the positions
    /// does not span two dimensions.
    fn plane(&self, planar: impl Fn(&[f64; 3], f64) -> bool) -> Plane {
        let mean = self.mean();
        let flat = Plane {
            a: mean.as_vec3(),
            gx: Vec3::ZERO,
            gy: Vec3::ZERO,
        };
        let ([sxx, sxy, syy], mp) = self.scatter();
        if !planar(&[sxx, sxy, syy], self.n) {
            return flat;
        }
        let det = sxx * syy - sxy * sxy;
        // The color's scatter against the positions, per channel
        let cx = self.xc - mp.x * self.c;
        let cy = self.yc - mp.y * self.c;
        let gx = (syy * cx - sxy * cy) / det;
        let gy = (sxx * cy - sxy * cx) / det;
        let a = mean - gx * mp.x - gy * mp.y;
        Plane {
            a: a.as_vec3(),
            gx: gx.as_vec3(),
            gy: gy.as_vec3(),
        }
    }
}

/// The equation regions' test for a plane: pixels on a line, a region one pixel wide, have
/// no slope across it.
fn region_planar(s: &[f64; 3], _n: f64) -> bool {
    let det = s[0] * s[2] - s[1] * s[1];
    det.abs() >= 1e-9 * (s[0] + s[2]).max(1.).powi(2)
}

/// The leaves' test: a dozen pixels spanning two dimensions.
fn leaf_planar(s: &[f64; 3], n: f64) -> bool {
    let det = s[0] * s[2] - s[1] * s[1];
    n >= 12. && det.abs() > 1e-6 * n * n
}

/// Most a channel may be off a plane for two regions to be one surface: 3 levels.
const PLANE_TOLERANCE: f32 = 3. * NOISE;
/// Pixels of a region sampled for the merge tests, which are percentiles: a strided sample
/// of a large region decides them as well, and a large sky against each neighbor is costly.
const MERGE_SAMPLE: usize = 4000;

/// The segmentation's regions merged greedily while one plane fits the union, so a gradient
/// stays one region: a pair merges, cheapest first, when each plane predicts the other's
/// pixels to the 90th percentile within the tolerance and so does the joint plane to the 95th.
pub fn equation_regions(target: &[Vec3], labels: &[u32], width: usize) -> Vec<u32> {
    let count = labels.iter().max().map_or(0, |&l| l as usize + 1);
    let mut pixels: Vec<Vec<u32>> = vec![Vec::new(); count];
    let mut moments = vec![Moments::default(); count];
    for (i, &l) in labels.iter().enumerate() {
        pixels[l as usize].push(i as u32);
        let (x, y) = ((i % width) as f64, (i / width) as f64);
        moments[l as usize].add(x, y, target[i].as_dvec3(), 1.);
    }
    let region_plane = |m: &Moments, area: usize| {
        if area >= 12 {
            m.plane(region_planar)
        } else {
            m.plane(|_, _| false)
        }
    };
    let mut planes: Vec<Plane> = (0..count)
        .map(|r| region_plane(&moments[r], pixels[r].len()))
        .collect();
    let mut neighbors: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); count];
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    let height = labels.len() / width.max(1);
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            for j in [
                (x + 1 < width).then(|| i + 1),
                (y + 1 < height).then(|| i + width),
            ]
            .into_iter()
            .flatten()
            {
                let (a, b) = (labels[i], labels[j]);
                if a != b && neighbors[a as usize].insert(b) {
                    neighbors[b as usize].insert(a);
                    pairs.push((a.min(b), a.max(b)));
                }
            }
        }
    }
    let at = |i: u32| Vec2::new((i as usize % width) as f32, (i as usize / width) as f32);
    let mergeable = |pa: &[u32], plane_a: &Plane, pb: &[u32], plane_b: &Plane| -> Option<f32> {
        let step_a = (pa.len() / MERGE_SAMPLE).max(1);
        let step_b = (pb.len() / MERGE_SAMPLE).max(1);
        let off =
            |plane: &Plane, i: u32| (target[i as usize] - plane.at(at(i))).abs().max_element();
        let mut by_a: Vec<f32> = pb
            .iter()
            .step_by(step_b)
            .map(|&i| off(plane_a, i))
            .collect();
        if percentile(&mut by_a, 90.) > PLANE_TOLERANCE {
            return None;
        }
        let mut by_b: Vec<f32> = pa
            .iter()
            .step_by(step_a)
            .map(|&i| off(plane_b, i))
            .collect();
        if percentile(&mut by_b, 90.) > PLANE_TOLERANCE {
            return None;
        }
        // The joint plane over the two samples, each pixel weighing its stride
        let mut joint = Moments::default();
        for (list, step) in [(pa, step_a), (pb, step_b)] {
            for &i in list.iter().step_by(step) {
                let p = at(i);
                joint.add(
                    p.x as f64,
                    p.y as f64,
                    target[i as usize].as_dvec3(),
                    step as f64,
                );
            }
        }
        let plane = joint.plane(region_planar);
        let mut res: Vec<f32> = pa
            .iter()
            .step_by(step_a)
            .chain(pb.iter().step_by(step_b))
            .map(|&i| off(&plane, i))
            .collect();
        let r = percentile(&mut res, 95.);
        (r <= PLANE_TOLERANCE).then_some(r)
    };
    let mut heap: BinaryHeap<Reverse<(Ordered, u32, u32)>> = BinaryHeap::new();
    for &(a, b) in &pairs {
        let (a, b) = (a as usize, b as usize);
        if let Some(r) = mergeable(&pixels[a], &planes[a], &pixels[b], &planes[b]) {
            heap.push(Reverse((Ordered(r), a as u32, b as u32)));
        }
    }
    let mut alive = vec![true; count];
    while let Some(Reverse((_, a, b))) = heap.pop() {
        let (a, b) = (a as usize, b as usize);
        if !(alive[a] && alive[b]) {
            continue;
        }
        if mergeable(&pixels[a], &planes[a], &pixels[b], &planes[b]).is_none() {
            continue;
        }
        // The larger list takes the smaller: the order of the pixels only decides the
        // strided samples, and appending a large list to each piece it absorbs is quadratic
        let taken = std::mem::take(&mut pixels[b]);
        if taken.len() > pixels[a].len() {
            let mine = std::mem::replace(&mut pixels[a], taken);
            pixels[a].extend(mine);
        } else {
            pixels[a].extend(taken);
        }
        let mb = moments[b];
        moments[a].merge(&mb);
        planes[a] = region_plane(&moments[a], pixels[a].len());
        alive[b] = false;
        let theirs = std::mem::take(&mut neighbors[b]);
        neighbors[a].remove(&(b as u32));
        for nb in theirs {
            if nb as usize != a && alive[nb as usize] {
                neighbors[a].insert(nb);
                neighbors[nb as usize].remove(&(b as u32));
                neighbors[nb as usize].insert(a as u32);
            }
        }
        for &nb in &neighbors[a] {
            let nb = nb as usize;
            if alive[nb]
                && let Some(r) = mergeable(&pixels[a], &planes[a], &pixels[nb], &planes[nb])
            {
                heap.push(Reverse((Ordered(r), a as u32, nb as u32)));
            }
        }
    }
    let mut out = vec![0u32; labels.len()];
    let mut next = 0;
    for (r, list) in pixels.iter().enumerate() {
        if !alive[r] {
            continue;
        }
        for &i in list {
            out[i as usize] = next;
        }
        next += 1;
    }
    out
}

/// An `f32` ordered by `total_cmp`, for the merge heap.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ordered(f32);
impl Eq for Ordered {}
impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The 4-connected parts of an image where `together(i, j)` says two neighbors belong to one,
/// numbered in the order of their first pixel. Returns the part per pixel and their count.
pub fn components(
    width: usize,
    height: usize,
    together: impl Fn(usize, usize) -> bool,
) -> (Vec<u32>, usize) {
    let mut part = vec![u32::MAX; width * height];
    let mut count = 0u32;
    let mut stack = Vec::new();
    for start in 0..width * height {
        if part[start] != u32::MAX {
            continue;
        }
        part[start] = count;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % width, i / width);
            for j in [
                (x > 0).then(|| i - 1),
                (x + 1 < width).then(|| i + 1),
                (y > 0).then(|| i - width),
                (y + 1 < height).then(|| i + width),
            ]
            .into_iter()
            .flatten()
            {
                if part[j] == u32::MAX && together(i, j) {
                    part[j] = count;
                    stack.push(j);
                }
            }
        }
        count += 1;
    }
    (part, count as usize)
}

/// Fewest pixels of a flat component.
const FLAT_MIN: u32 = 24;

/// Connected runs of one 8-bit color with at least [`FLAT_MIN`] pixels and an interior pixel
/// (all four neighbors in the run, within the frame): flat art is exact there. Returns a run
/// id plus one per pixel, nothing (0) elsewhere.
fn flat_components(target: &[Vec3], width: usize, height: usize) -> Vec<u32> {
    let key: Vec<u32> = target
        .iter()
        .map(|c| {
            let q = (*c * 255.).round();
            ((q.x as u32) << 16) | ((q.y as u32) << 8) | q.z as u32
        })
        .collect();
    let (part, count) = components(width, height, |i, j| key[i] == key[j]);
    let mut size = vec![0u32; count];
    let mut interior = vec![false; count];
    for (i, &p) in part.iter().enumerate() {
        size[p as usize] += 1;
        let (x, y) = (i % width, i / width);
        if x > 0
            && y > 0
            && x + 1 < width
            && y + 1 < height
            && [i - 1, i + 1, i - width, i + width]
                .iter()
                .all(|&j| part[j] == p)
        {
            interior[p as usize] = true;
        }
    }
    // Flat runs that touch and differ by a level at most in every channel are one color, as
    // 8 bits quantize it: a peeled color continuing out from under a shape comes back as two
    // neighbors a level apart (triple-02's blue rectangle)
    let flat = |p: u32| size[p as usize] >= FLAT_MIN && interior[p as usize];
    let mut first = vec![usize::MAX; count];
    for (i, &p) in part.iter().enumerate() {
        if first[p as usize] == usize::MAX {
            first[p as usize] = i;
        }
    }
    let color = |p: u32| {
        let k = key[first[p as usize]];
        IVec3::new((k >> 16) as i32, (k >> 8 & 255) as i32, (k & 255) as i32)
    };
    let mut group: Vec<u32> = (0..count as u32).collect();
    fn root(group: &mut [u32], mut p: u32) -> u32 {
        while group[p as usize] != p {
            group[p as usize] = group[group[p as usize] as usize];
            p = group[p as usize];
        }
        p
    }
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            for j in [
                (x + 1 < width).then(|| i + 1),
                (y + 1 < height).then(|| i + width),
            ]
            .into_iter()
            .flatten()
            {
                let (a, b) = (part[i], part[j]);
                if a != b && flat(a) && flat(b) && (color(a) - color(b)).abs().max_element() <= 1 {
                    let (ra, rb) = (root(&mut group, a), root(&mut group, b));
                    if ra != rb {
                        group[ra.max(rb) as usize] = ra.min(rb);
                    }
                }
            }
        }
    }
    let mut id = vec![0u32; count];
    let mut next = 1;
    for p in 0..count as u32 {
        let r = root(&mut group, p);
        if flat(p) && id[r as usize] == 0 {
            id[r as usize] = next;
            next += 1;
        }
    }
    part.iter()
        .map(|&p| {
            if flat(p) {
                id[root(&mut group, p) as usize]
            } else {
                0
            }
        })
        .collect()
}

/// A directed pair of leaves that touch: `x` inside, `y` outside.
#[derive(Clone, Debug)]
pub struct Pair {
    pub x: u32,
    pub y: u32,
    /// Pixel edges between the two on the bridged map.
    pub len: u32,
    /// Its contact points in [`Leaves::points`].
    pub points: std::ops::Range<usize>,
    /// The mean position of the contact.
    pub mid: Vec2,
}

/// A point along a contact, where the two sides are compared: `o` the inside's color there
/// and `b` the outside's, each from its own leaf's core pixels near the point carried to it
/// by the leaf's slope; `w` the contact pixels the point stands for.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub w: f32,
    pub o: Vec3,
    pub b: Vec3,
}

/// Fewest contact pixels per point.
const PIXELS_PER_POINT: usize = 8;
const MAX_POINTS: usize = 32;
/// How far a leaf's core pixels are gathered around a contact point for its color there.
const LOCAL_RADIUS: i32 = 5;
/// How far a thin leaf's pixels go to the nearest wide leaf: a stroke of three pixels with
/// its rims (the glassware's glass pieces have a 3 px stroke each).
const BRIDGE: i32 = 3;

/// The leaves of the region stage and the plane pass: the flat components where there are
/// any, the equation regions elsewhere, each connected part a leaf with its plane, texture,
/// thinness and interior. Thin leaves are bridged to a wide one, then contacts are sampled.
pub struct Leaves {
    pub width: usize,
    pub height: usize,
    pub n: usize,
    /// Leaf per pixel.
    pub label: Vec<u32>,
    /// Whether all four neighbors of a pixel are in its leaf (the frame counts as another).
    pub core: Vec<bool>,
    pub size: Vec<u32>,
    pub interior: Vec<bool>,
    pub thin: Vec<bool>,
    pub plane: Vec<Plane>,
    pub texture: Vec<f32>,
    pub uncertainty: Vec<f32>,
    /// The plane at the centroid of the pixels it was fitted on.
    pub color: Vec<Vec3>,
    /// The mean position of all its pixels.
    pub centroid: Vec<Vec2>,
    /// Each pixel's leaf's plane at it.
    pub plain: Vec<Vec3>,
    /// The bridged leaf per pixel, -1 where no wide leaf lies within [`BRIDGE`] px.
    pub filled: Vec<i32>,
    /// Pixels per bridged leaf.
    pub filled_n: Vec<u32>,
    /// Pixel indices by bridged leaf: those of leaf `l` at `filled_start[l]..filled_start[l + 1]`.
    pub filled_pixels: Vec<u32>,
    pub filled_start: Vec<usize>,
    /// Pixel indices by leaf, as `filled_pixels`.
    pub pixels: Vec<u32>,
    pub start: Vec<usize>,
    pub pairs: Vec<Pair>,
    pub points: Vec<Point>,
    /// The pairs each leaf is the inside of, as indices into `pairs`, in their order.
    pub reach: Vec<Vec<u32>>,
}

impl Leaves {
    pub fn new(target: &[Vec3], segments: &[u32], width: usize, height: usize) -> Self {
        let eq = equation_regions(target, segments, width);
        let flat = flat_components(target, width, height);
        // A flat component takes precedence: on flat art it is exact, where the plane merge mixes
        // a stroke with the dotted border it crosses (the airmail's stamp ring)
        let base: Vec<(bool, u32)> = (0..width * height)
            .map(|i| {
                if flat[i] > 0 {
                    (true, flat[i])
                } else {
                    (false, eq[i])
                }
            })
            .collect();
        let (label, n) = components(width, height, |i, j| base[i] == base[j]);
        let at = |i: usize| Vec2::new((i % width) as f32, (i / width) as f32);

        // Neighbors in the same leaf per pixel, the frame counting as another
        let same: Vec<u8> = (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                [
                    (x > 0).then(|| i - 1),
                    (x + 1 < width).then(|| i + 1),
                    (y > 0).then(|| i - width),
                    (y + 1 < height).then(|| i + width),
                ]
                .into_iter()
                .flatten()
                .filter(|&j| label[j] == label[i])
                .count() as u8
            })
            .collect();
        let core: Vec<bool> = same.iter().map(|&s| s == 4).collect();
        let mut size = vec![0u32; n];
        let mut ncore = vec![0u32; n];
        let mut interior = vec![false; n];
        for i in 0..width * height {
            let l = label[i] as usize;
            size[l] += 1;
            ncore[l] += u32::from(core[i]);
            interior[l] |= same[i] >= 3;
        }
        // Each leaf's plane by least squares over its core pixels where it has a dozen: the
        // pixels at its edge are the anti-aliased blend with the neighbor, which tilts a small
        // leaf's plane (saturn's ring over the globe)
        let used = |i: usize| core[i] || ncore[label[i] as usize] < 12;
        let mut moments = vec![Moments::default(); n];
        let mut all = vec![DVec2::ZERO; n];
        for i in 0..width * height {
            let l = label[i] as usize;
            let p = at(i).as_dvec2();
            all[l] += p;
            if used(i) {
                moments[l].add(p.x, p.y, target[i].as_dvec3(), 1.);
            }
        }
        let plane: Vec<Plane> = moments.iter().map(|m| m.plane(leaf_planar)).collect();
        let color: Vec<Vec3> = moments
            .iter()
            .map(|m| m.mean().as_vec3().clamp(Vec3::ZERO, Vec3::ONE))
            .collect();
        let centroid: Vec<Vec2> = (0..n)
            .map(|l| (all[l] / size[l].max(1) as f64).as_vec2())
            .collect();
        let plain: Vec<Vec3> = (0..width * height)
            .map(|i| plane[label[i] as usize].at(at(i)))
            .collect();
        let (pixels, start) = by_label(&label, n);
        // The texture: the median deviation of the leaf's fitted pixels off its plane
        let mut texture = vec![0f32; n];
        let mut uncertainty = vec![0f32; n];
        for l in 0..n {
            let mut dev: Vec<f32> = pixels[start[l]..start[l + 1]]
                .iter()
                .map(|&i| i as usize)
                .filter(|&i| used(i))
                .map(|i| (target[i] - plain[i]).abs().max_element())
                .collect();
            texture[l] = median(&mut dev);
            uncertainty[l] = texture[l] / (moments[l].n.max(1.) as f32).sqrt();
        }
        // Thin: no pixel whose eight neighbors are all core pixels, so at most four pixels across
        // (an anti-aliased rim, or a stroke of up to three pixels with its rims). Counted this way
        // rather than by area over perimeter, which miscounts diagonals (the geode's veil strip).
        let mut thin = vec![true; n];
        for y in 1..height.saturating_sub(1) {
            for x in 1..width - 1 {
                let i = y * width + x;
                if core[i]
                    && (-1..=1).all(|dy: i32| {
                        (-1..=1).all(|dx: i32| core[(i as i32 + dy * width as i32 + dx) as usize])
                    })
                {
                    thin[label[i] as usize] = false;
                }
            }
        }
        // The bridged map: each pixel of a thin leaf to the nearest pixel of a wide one, ties to
        // the leaf whose plane is nearer the pixel's color, as the hierarchy dissolves its blends
        // (a scan-order tie shifts a shape's mask by half a pixel)
        let mut filled = vec![-1i32; width * height];
        for i in 0..width * height {
            if !thin[label[i] as usize] {
                filled[i] = label[i] as i32;
                continue;
            }
            let (x, y) = ((i % width) as i32, (i / width) as i32);
            let off = |j: usize| {
                (target[i] - plane[label[j] as usize].at(at(i)))
                    .abs()
                    .max_element()
            };
            let mut best: Option<(i32, usize)> = None;
            for dy in -BRIDGE..=BRIDGE {
                for dx in -BRIDGE..=BRIDGE {
                    let d = dx * dx + dy * dy;
                    let (u, v) = (x + dx, y + dy);
                    if d > BRIDGE * BRIDGE
                        || u < 0
                        || v < 0
                        || u >= width as i32
                        || v >= height as i32
                    {
                        continue;
                    }
                    let j = v as usize * width + u as usize;
                    if !thin[label[j] as usize]
                        && best.is_none_or(|(e, b)| d < e || (d == e && off(j) < off(b)))
                    {
                        best = Some((d, j));
                    }
                }
            }
            if let Some((_, j)) = best {
                filled[i] = label[j] as i32;
            }
        }
        let filled_u: Vec<u32> = filled
            .iter()
            .map(|&f| if f < 0 { n as u32 } else { f as u32 })
            .collect();
        let (filled_pixels, mut filled_start) = by_label(&filled_u, n + 1);
        filled_start.truncate(n + 1);
        let filled_n: Vec<u32> = (0..n)
            .map(|l| (filled_start[l + 1] - filled_start[l]) as u32)
            .collect();

        let mut leaves = Leaves {
            width,
            height,
            n,
            label,
            core,
            size,
            interior,
            thin,
            plane,
            texture,
            uncertainty,
            color,
            centroid,
            plain,
            filled,
            filled_n,
            filled_pixels,
            filled_start,
            pixels,
            start,
            pairs: Vec::new(),
            points: Vec::new(),
            reach: vec![Vec::new(); n],
        };
        leaves.contacts(target);
        leaves
    }

    /// The pixels of leaf `l`.
    pub fn pixels_of(&self, l: usize) -> &[u32] {
        &self.pixels[self.start[l]..self.start[l + 1]]
    }

    /// The pixels the bridged map gives leaf `l`.
    pub fn filled_of(&self, l: usize) -> &[u32] {
        &self.filled_pixels[self.filled_start[l]..self.filled_start[l + 1]]
    }

    /// The contact map on the bridged map: per pair of leaves that touch, both directions,
    /// in the order of the unordered pair, with points along the contact's principal axis.
    fn contacts(&mut self, target: &[Vec3]) {
        let (width, height, n) = (self.width, self.height, self.n);
        let mut edges: Vec<(u64, Vec2)> = Vec::new();
        for (dx, dy) in [(1usize, 0usize), (0, 1)] {
            for y in 0..height - dy {
                for x in 0..width - dx {
                    let i = y * width + x;
                    let (a, b) = (self.filled[i], self.filled[i + dy * width + dx]);
                    if a != b && a >= 0 && b >= 0 {
                        let (lo, hi) = (a.min(b) as u64, a.max(b) as u64);
                        edges.push((
                            lo * n as u64 + hi,
                            Vec2::new(x as f32 + dx as f32 / 2., y as f32 + dy as f32 / 2.),
                        ));
                    }
                }
            }
        }
        edges.sort_by_key(|e| e.0);
        let mut s = 0;
        while s < edges.len() {
            let key = edges[s].0;
            let mut e = s;
            while e < edges.len() && edges[e].0 == key {
                e += 1;
            }
            let pts: Vec<Vec2> = edges[s..e].iter().map(|e| e.1).collect();
            s = e;
            let c = pts.len();
            let mid = pts.iter().copied().sum::<Vec2>() / c as f32;
            // Chunks of the contact along its principal axis
            let chunks: Vec<Vec<usize>> = if c >= PIXELS_PER_POINT {
                let (mut sxx, mut sxy, mut syy) = (0f32, 0f32, 0f32);
                for p in &pts {
                    let d = *p - mid;
                    sxx += d.x * d.x;
                    sxy += d.x * d.y;
                    syy += d.y * d.y;
                }
                // The eigenvector of the larger eigenvalue of the scatter
                let t = (sxx - syy) / 2.;
                let l = (sxx + syy) / 2. + (t * t + sxy * sxy).sqrt();
                let axis = if sxy.abs() > 1e-12 {
                    Vec2::new(l - syy, sxy).normalize()
                } else if sxx >= syy {
                    Vec2::X
                } else {
                    Vec2::Y
                };
                let mut order: Vec<usize> = (0..c).collect();
                order.sort_by(|&a, &b| {
                    (pts[a] - mid)
                        .dot(axis)
                        .total_cmp(&(pts[b] - mid).dot(axis))
                });
                let parts = (c / PIXELS_PER_POINT).min(MAX_POINTS);
                let (base, extra) = (c / parts, c % parts);
                let mut out = Vec::with_capacity(parts);
                let mut at = 0;
                for k in 0..parts {
                    let len = base + usize::from(k < extra);
                    out.push(order[at..at + len].to_vec());
                    at += len;
                }
                out
            } else {
                vec![(0..c).collect()]
            };
            let (lo, hi) = ((key / n as u64) as u32, (key % n as u64) as u32);
            for (x, y) in [(lo, hi), (hi, lo)] {
                let first = self.points.len();
                for chunk in &chunks {
                    let at = chunk.iter().map(|&k| pts[k]).sum::<Vec2>() / chunk.len() as f32;
                    let o = self.local(target, x, at);
                    let b = self.local(target, y, at);
                    self.points.push(Point {
                        w: chunk.len() as f32,
                        o,
                        b,
                    });
                }
                self.reach[x as usize].push(self.pairs.len() as u32);
                self.pairs.push(Pair {
                    x,
                    y,
                    len: c as u32,
                    points: first..self.points.len(),
                    mid,
                });
            }
        }
    }

    /// A leaf's color at a contact point: the mean of its own core pixels within
    /// [`LOCAL_RADIUS`], carried to the point by its plane's slope (its plane where it has
    /// none). The plane alone is too loose: it fits to 3 noise levels and drifts over a span.
    fn local(&self, target: &[Vec3], leaf: u32, p: Vec2) -> Vec3 {
        let (cx, cy) = (p.x.floor() as i32, p.y.floor() as i32);
        let (mut n, mut sum, mut pos) = (0f32, Vec3::ZERO, Vec2::ZERO);
        for v in cy - LOCAL_RADIUS..=cy + LOCAL_RADIUS {
            for u in cx - LOCAL_RADIUS..=cx + LOCAL_RADIUS {
                if u < 0 || v < 0 || u >= self.width as i32 || v >= self.height as i32 {
                    continue;
                }
                let j = v as usize * self.width + u as usize;
                if self.label[j] == leaf && self.core[j] {
                    n += 1.;
                    sum += target[j];
                    pos += Vec2::new(u as f32, v as f32);
                }
            }
        }
        let plane = &self.plane[leaf as usize];
        if n > 0. {
            sum / n + plane.step(p - pos / n)
        } else {
            plane.at(p)
        }
    }

    /// Per directed pair (x inside, y outside): x is y seen through `o = k b + m` at every
    /// contact point to the noise and the planes' uncertainty, the two differ somewhere, x's
    /// texture is k times y's, the slopes agree, and x has an interior.
    pub fn pair_fits(&self, k: f32, m: Vec3) -> Vec<bool> {
        let unc = &self.uncertainty;
        self.pairs
            .iter()
            .map(|p| {
                let (x, y) = (p.x as usize, p.y as usize);
                let slack = 2. * NOISE + unc[x] + k * unc[y];
                let points = &self.points[p.points.clone()];
                self.interior[x]
                    && points
                        .iter()
                        .all(|q| (q.o - (k * q.b + m)).abs().max_element() <= slack)
                    && points
                        .iter()
                        .any(|q| (q.o - q.b).abs().max_element() > 2. * NOISE)
                    && (self.texture[x] - k * self.texture[y]).abs() <= slack
                    && slope_fits(&self.plane[x], &self.plane[y], k)
            })
            .collect()
    }
}

/// Indices `0..labels.len()` grouped by label: the indices and where each label's start.
fn by_label(labels: &[u32], n: usize) -> (Vec<u32>, Vec<usize>) {
    let mut start = vec![0usize; n + 1];
    for &l in labels {
        start[l as usize + 1] += 1;
    }
    for l in 0..n {
        start[l + 1] += start[l];
    }
    let mut at = start.clone();
    let mut out = vec![0u32; labels.len()];
    for (i, &l) in labels.iter().enumerate() {
        out[at[l as usize]] = i as u32;
        at[l as usize] += 1;
    }
    (out, start)
}

/// The median, the mean of the two middle values for an even count; nothing for no values.
pub fn median(values: &mut [f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.;
    }
    values.sort_unstable_by(f32::total_cmp);
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    }
}
