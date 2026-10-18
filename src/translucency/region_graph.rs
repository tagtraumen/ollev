//! The layer's region as the minimum cut over the segmentation's leaves: a leaf pays for its
//! pixels read the other way (layer, inner layer above it, or an outside material seen bare),
//! a boundary per pixel of length by how little contour it lies on, times the unit.

use crate::HashMap;
use std::collections::VecDeque;

use glam::DVec3;

use super::materials::{MATCH_TOL, Palette};
use super::raster::{Conn, Image, SHOW_THROUGH, label, unit_of};
use super::region::{evidence, fill_holes};

/// What the relation says of a pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Neither or both of the layer's and the bare reading.
    Undecided,
    /// Within the matcher's tolerance of a material colour's image, moved by more than the
    /// show-through.
    Core,
    /// On an inner layer above the image, or on the image of a material's shading.
    Inner,
    /// An outside material seen bare.
    Bare,
}

/// The point of the triangle `a`, `b`, `c` nearest `p` (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn nearest_on_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + d1 / (d1 - d3) * ab;
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + d2 / (d2 - d6) * ac;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b);
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

/// What the relation says of each pixel: its [`Class`], and the relation's core.
pub struct Reading {
    pub class: Vec<Class>,
    pub core: Vec<bool>,
}

/// The [`Reading`] of the relation `k`, `c` over the materials of `outside`, with `tints` the
/// colours of the tints the pin found above the layer (each a further corner of the
/// triangles); none when `outside` has no material.
pub fn classes(
    img: &Image,
    outside: &Palette,
    k: f64,
    c: DVec3,
    tints: &[DVec3],
    min_px: f64,
) -> Option<Reading> {
    let ev = evidence(img, outside, k, c, min_px)?;
    let m = (1.0 - k) * c;
    // A material's reading under the layer (dichromatic): triangles with apex its base image I,
    // a far end of its polyline's image (highlight to k + m, shade to m) and an end above (white,
    // black, c, or a pin tint: niabot's dark folds)
    let above: Vec<DVec3> = [DVec3::ONE, DVec3::ZERO, c]
        .into_iter()
        .chain(tints.iter().copied())
        .collect();
    let triangles: Vec<[DVec3; 3]> = ev
        .bases
        .iter()
        .flat_map(|&b| {
            let i = k * b + m;
            let above = &above;
            [k * DVec3::ONE + m, m]
                .into_iter()
                .flat_map(move |e| above.iter().map(move |&a| [i, e, a]))
        })
        .collect();
    // Off the core a pixel is the layer's where only a layer reading explains it (matcher's
    // tolerance), bare where only a bare one does (show-through), else undecided; asked once
    // per distinct colour
    let mut memo: HashMap<[u64; 3], bool> = HashMap::default();
    let class = img
        .px
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            if ev.core[i] {
                return Class::Core;
            }
            let layer = *memo
                .entry(p.to_array().map(f64::to_bits))
                .or_insert_with(|| {
                    triangles.iter().any(|&[a, b, c]| {
                        (p - nearest_on_triangle(p, a, b, c)).abs().max_element() <= MATCH_TOL
                    })
                });
            match (layer, ev.bare[i]) {
                (true, false) => Class::Inner,
                (false, true) => Class::Bare,
                _ => Class::Undecided,
            }
        })
        .collect();
    Some(Reading {
        class,
        core: ev.core,
    })
}

/// The cost per pixel of length of a boundary on the crack between `i` and `j`: the unit, falling
/// with the step in show-throughs (0.61 at one, 0.14 at two), the step read one pixel further out
/// on each side, so an anti-aliased step counts fully and a crack along an edge's band does not.
pub fn crack_cost(img: &Image, i: usize, j: usize) -> f64 {
    let (w, h) = (img.width, img.height);
    let u = unit_of(w);
    // the pixels one further out across the crack, where the image has them
    let (o_i, o_j) = if j == i + 1 {
        (
            if !i.is_multiple_of(w) { i - 1 } else { i },
            if j % w + 1 < w { j + 1 } else { j },
        )
    } else {
        (
            if i >= w { i - w } else { i },
            if j + w < w * h { j + w } else { j },
        )
    };
    // and no more than twice the step between the crack's own two pixels: an anti-aliased
    // pixel holds at least half of the step it mixes, while a crack beside a corner of a
    // contour sees the contour in its outer pixels but no step of its own
    let own = (img.px[i] - img.px[j]).abs().max_element();
    let step = (img.px[o_i] - img.px[o_j])
        .abs()
        .max_element()
        .min(2.0 * own);
    u * (-step * step / (2.0 * SHOW_THROUGH * SHOW_THROUGH)).exp()
}

/// The layer's region under `k`, `c` over the materials of `outside` with the pin's `tints`,
/// the minimum cut over `leaves` ([`classes`], [`from_graph`]); `min_px` is a material's least
/// pixels at this width. Empty where `outside` has no material.
pub fn region(
    img: &Image,
    outside: &Palette,
    k: f64,
    c: DVec3,
    tints: &[DVec3],
    leaves: &[u32],
    min_px: f64,
) -> Vec<bool> {
    let Some(reading) = classes(img, outside, k, c, tints, min_px) else {
        return vec![false; img.px.len()];
    };
    let region = from_graph(&reading, leaves, img, min_px);
    if crate::diagnostics::debug() {
        let count = |want: Class| reading.class.iter().filter(|&&v| v == want).count();
        eprintln!(
            "region: relation k {k:.3} c {:?} with {} tints: core {} px, inner {} px, bare {} px, region {} px",
            (c * 255.).round().to_array(),
            tints.len(),
            count(Class::Core),
            count(Class::Inner),
            count(Class::Bare),
            region.iter().filter(|v| **v).count()
        );
    }
    region
}

/// A flow network with real capacities: Dinic's algorithm.
struct Flow {
    head: Vec<usize>,
    to: Vec<usize>,
    cap: Vec<f64>,
    next: Vec<usize>,
}

const NONE: usize = usize::MAX;

impl Flow {
    fn new(n: usize) -> Self {
        Flow {
            head: vec![NONE; n],
            to: Vec::new(),
            cap: Vec::new(),
            next: Vec::new(),
        }
    }

    /// An edge `a` to `b` of capacity `ab` and its reverse of capacity `ba`.
    fn add(&mut self, a: usize, b: usize, ab: f64, ba: f64) {
        for (from, to, cap) in [(a, b, ab), (b, a, ba)] {
            self.to.push(to);
            self.cap.push(cap);
            self.next.push(self.head[from]);
            self.head[from] = self.to.len() - 1;
        }
    }

    /// The maximum flow from `s` to `t`; afterwards the nodes reachable from `s` in the
    /// residual graph are the source's side of a minimum cut.
    fn run(&mut self, s: usize, t: usize) -> Vec<bool> {
        let n = self.head.len();
        // a capacity below this is spent (the costs are pixel counts, at least a hundredth)
        let eps = 1e-9;
        loop {
            let mut level = vec![u32::MAX; n];
            level[s] = 0;
            let mut queue = VecDeque::from([s]);
            while let Some(v) = queue.pop_front() {
                let mut e = self.head[v];
                while e != NONE {
                    if self.cap[e] > eps && level[self.to[e]] == u32::MAX {
                        level[self.to[e]] = level[v] + 1;
                        queue.push_back(self.to[e]);
                    }
                    e = self.next[e];
                }
            }
            if level[t] == u32::MAX {
                return level.iter().map(|&l| l != u32::MAX).collect();
            }
            let mut it = self.head.clone();
            // blocking flow by iterative depth-first search
            loop {
                let mut path: Vec<usize> = Vec::new();
                let mut v = s;
                let found = loop {
                    if v == t {
                        break true;
                    }
                    let mut advanced = false;
                    while it[v] != NONE {
                        let e = it[v];
                        let w = self.to[e];
                        if self.cap[e] > eps && level[w] == level[v] + 1 {
                            path.push(e);
                            v = w;
                            advanced = true;
                            break;
                        }
                        it[v] = self.next[e];
                    }
                    if !advanced {
                        if v == s {
                            break false;
                        }
                        // a dead end: never visited again in this phase
                        level[v] = u32::MAX;
                        let e = path.pop().expect("a path to retreat along");
                        v = self.to[e ^ 1];
                        it[v] = self.next[it[v]];
                    }
                };
                if !found {
                    break;
                }
                let push = path.iter().map(|&e| self.cap[e]).fold(f64::MAX, f64::min);
                for &e in &path {
                    self.cap[e] -= push;
                    self.cap[e ^ 1] += push;
                }
            }
        }
    }
}

/// The layer's region from the pixels' [`Class`]es over the graph of `nodes` (a leaf per
/// pixel): the minimum cut, then its pieces holding a core piece of `min_px`, holes filled.
/// The frame counts as seen bare: the layer is a thing in the picture.
pub fn from_graph(reading: &Reading, nodes: &[u32], img: &Image, min_px: f64) -> Vec<bool> {
    let (w, h) = (img.width, img.height);
    let n = w * h;
    let count = nodes.iter().max().map_or(0, |&l| l as usize + 1);
    let (mut inside, mut bare) = (vec![0.0f64; count], vec![0.0f64; count]);
    for (i, &node) in nodes.iter().enumerate() {
        let v = node as usize;
        let (x, y) = (i % w, i / w);
        if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
            bare[v] += 1.0;
            continue;
        }
        match reading.class[i] {
            Class::Core | Class::Inner => inside[v] += 1.0,
            Class::Bare => bare[v] += 1.0,
            Class::Undecided => {}
        }
    }
    let mut boundary: HashMap<(u32, u32), f64> = HashMap::default();
    for i in 0..n {
        let (x, y) = (i % w, i / w);
        for j in [(x + 1 < w).then(|| i + 1), (y + 1 < h).then(|| i + w)]
            .into_iter()
            .flatten()
        {
            let (a, b) = (nodes[i], nodes[j]);
            if a != b {
                *boundary.entry((a.min(b), a.max(b))).or_insert(0.0) += crack_cost(img, i, j);
            }
        }
    }
    let (s, t) = (count, count + 1);
    let mut flow = Flow::new(count + 2);
    for v in 0..count {
        // the shared part of both sides' evidence passes either way: only the difference cuts
        let d = inside[v] - bare[v];
        if d > 0.0 {
            flow.add(s, v, d, 0.0);
        } else if d < 0.0 {
            flow.add(v, t, -d, 0.0);
        }
    }
    let mut pairs: Vec<_> = boundary.into_iter().collect();
    pairs.sort_by_key(|p| p.0);
    for ((a, b), cost) in pairs {
        flow.add(a as usize, b as usize, cost, cost);
    }
    let side = flow.run(s, t);
    let cut: Vec<bool> = nodes.iter().map(|&v| side[v as usize]).collect();
    // the cut's pieces that hold a core piece of a material's size
    let (lab, pieces) = label(&cut, w, h, Conn::Four);
    let mut held = vec![0usize; pieces + 1];
    for i in 0..n {
        if cut[i] && reading.core[i] {
            held[lab[i] as usize] += 1;
        }
    }
    let kept: Vec<bool> = (0..n)
        .map(|i| cut[i] && held[lab[i] as usize] as f64 >= min_px)
        .collect();
    fill_holes(&kept, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square of one colour on another, its left half read as the layer, its right half
    /// undecided, the rest bare: the cut follows the square's contour, not the evidence's
    /// edge through its middle, which no contour supports.
    #[test]
    fn the_cut_follows_the_contour_past_the_evidence() {
        let (w, h) = (40usize, 40usize);
        let inside = |x: usize, y: usize| (10..30).contains(&x) && (10..30).contains(&y);
        let mut rgb = Vec::with_capacity(3 * w * h);
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(if inside(x, y) {
                    &[200, 240, 200]
                } else {
                    &[120, 160, 220]
                });
            }
        }
        let img = Image::from_rgb8(&rgb, w, h);
        let (mut class, mut core) = (Vec::new(), Vec::new());
        for y in 0..h {
            for x in 0..w {
                let c = match (inside(x, y), x < 20) {
                    (true, true) => Class::Core,
                    (true, false) => Class::Undecided,
                    _ => Class::Bare,
                };
                core.push(c == Class::Core);
                class.push(c);
            }
        }
        let nodes: Vec<u32> = (0..(w * h) as u32).collect();
        let region = from_graph(&Reading { class, core }, &nodes, &img, 16.0);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(region[y * w + x], inside(x, y), "at {x}, {y}");
            }
        }
    }
}
