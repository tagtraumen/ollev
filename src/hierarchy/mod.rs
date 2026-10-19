//! A tree of regions from a fine segmentation up to the whole image. Neighbors merge in order of
//! the color error added per boundary pixel removed (the Mumford-Shah boundary cost accepting it),
//! so cutting the tree at a cost gives the segmentation at that cost.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::render::{Gradient, Stops};
use glam::{DVec2, DVec3, IVec2, Vec2, Vec3};

use crate::HashMap;

pub mod analyze;
mod blends;
mod fills;
mod stats;

pub use analyze::analyze;
pub use blends::*;
pub use fills::*;

/// Sums over a set of pixels, enough for its mean color and squared error, and for the linear
/// gradient that fits it best.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub n: f64,
    pub sum: DVec3,
    pub squares: f64,
    /// Sums of the products of channel pairs, `xx, xy, xz, yy, yz, zz`, for the covariance of the
    /// colors.
    pub products: [f64; 6],
    /// Sums of the pixel centers, of their products `xx, xy, yy`, and of each color times the
    /// position (`x` then `y`), for the linear fill of the pixels (see [`Stats::linear_fill`]).
    /// Zero in statistics built without positions.
    pub positions: DVec2,
    pub position_products: [f64; 3],
    pub color_positions: [DVec3; 2],
    /// Sums of the squared distance `r^2 = x^2 + y^2` of the pixel centers from the origin
    /// times `x`, times `y`, and squared, and of each color times it, for the radial fill of
    /// the pixels (see [`Stats::radial_fill`]).
    pub radial_moments: [f64; 3],
    pub color_radial: DVec3,
}

pub struct Node {
    pub stats: Stats,
    pub children: Option<[u32; 2]>,
    pub parent: Option<u32>,
    /// The boundary cost above which the node's regions would merge into it: the cost per
    /// boundary pixel of its merge, and never less than its children's.
    pub scale: f64,
    /// Pixel bounding box, `min` inclusive and `max` exclusive.
    pub min: IVec2,
    pub max: IVec2,
    /// Pixel edges between the node and other regions, and along the image border.
    pub perimeter: u32,
    pub frame: u32,
    /// Mean width in pixels of the color transition across the node's boundary: one for a clean
    /// edge, more for a blurred one, at most eight. Where the outline is only defined to within
    /// this width, its exact course matters less.
    pub blur: f32,
    /// The node's leaves are those with a depth-first rank in this range.
    leaves: (u32, u32),
}

/// Least contrast between a node and its neighbor for a rim pixel's mixture to be measurable.
const MIXED_CONTRAST: f64 = 0.1;
/// Scales at which the rim is looked at, in pixels: anti-aliasing is one pixel, pixel art is
/// drawn in blocks of a few.
const MIXED_STEPS: [i32; 4] = [1, 2, 4, 8];
/// Where between the two sides a rim pixel counts as a mixture: the pair across an anti-aliased
/// edge sums to one, so one lies a tenth from pure unless the edge hugs the grid.
const MIXED_LOW: f64 = 0.1;
const MIXED_HIGH: f64 = 0.9;
/// How far off the line between the two sides' colors a mixture may lie.
const MIXED_OFF_LINE: f64 = 0.08;
/// Pixels of a pass over the picture between two checkpoints.
const POLL: usize = 1 << 16;

pub struct Hierarchy {
    /// The leaves (regions of the segmentation) first, then each merge.
    pub nodes: Vec<Node>,
    pub root: u32,
    width: usize,
    /// The colors the statistics were taken from, per pixel.
    pub colors: Vec<DVec3>,
    /// Leaf per pixel.
    labels: Vec<u32>,
    /// Depth-first rank per leaf.
    rank: Vec<u32>,
    /// Leaves by depth-first rank.
    order: Vec<u32>,
    /// The pixels by the rank of their leaf, so a node's pixels are one run, and where each
    /// rank's pixels start (one more entry than ranks).
    pixels: Vec<u32>,
    pixel_start: Vec<u32>,
}

struct Merge {
    cost: f64,
    a: u32,
    b: u32,
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}
impl Eq for Merge {}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    // Reversed, so the max-heap pops the cheapest merge first
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.total_cmp(&self.cost)
    }
}

impl Hierarchy {
    /// Builds the tree over the regions of `labels` (`count` regions), with region statistics
    /// from `colors`.
    pub async fn build(
        labels: Vec<u32>,
        count: usize,
        colors: &[DVec3],
        width: usize,
        height: usize,
    ) -> Self {
        let mut nodes: Vec<Node> = (0..count)
            .map(|_| Node {
                stats: Stats::default(),
                children: None,
                parent: None,
                scale: 0.,
                min: IVec2::MAX,
                max: IVec2::MIN,
                perimeter: 0,
                frame: 0,
                blur: 1.,
                leaves: (0, 0),
            })
            .collect();
        for (i, &label) in labels.iter().enumerate() {
            if i % POLL == 0 {
                crate::progress::poll().await;
            }
            let (x, y) = (i % width, i / width);
            let p = IVec2::new(x as i32, y as i32);
            let node = &mut nodes[label as usize];
            node.stats = node
                .stats
                .add(&Stats::pixel_at(colors[i], p.as_dvec2() + 0.5));
            node.min = node.min.min(p);
            node.max = node.max.max(p + 1);
            node.frame += u32::from(x == 0)
                + u32::from(y == 0)
                + u32::from(x + 1 == width)
                + u32::from(y + 1 == height);
        }

        // Shared boundary length between neighboring regions, and the summed width of the color
        // transition across their boundary edges (see `transition_width`)
        let mut neighbors: Vec<HashMap<u32, u32>> = vec![HashMap::default(); count];
        let mut widths: HashMap<(u32, u32), f64> = HashMap::default();
        let mut link = |i: usize, j: usize, step: usize| {
            let (a, b) = (labels[i], labels[j]);
            if a != b {
                *neighbors[a as usize].entry(b).or_default() += 1;
                *neighbors[b as usize].entry(a).or_default() += 1;
                *widths.entry((a.min(b), a.max(b))).or_default() +=
                    transition_width(&labels, colors, i, j, step);
            }
        };
        for y in 0..height {
            if y % 16 == 0 {
                crate::progress::poll().await;
            }
            for x in 0..width {
                let i = y * width + x;
                if x + 1 < width {
                    link(i, i + 1, 1);
                }
                if y + 1 < height {
                    link(i, i + width, width);
                }
            }
        }

        // The boundaries between regions, for the perimeters and transition widths of the
        // merged nodes below
        let widths = &widths;
        let mut boundaries: Vec<(u32, u32, u32, f64)> = neighbors
            .iter()
            .enumerate()
            .flat_map(|(a, map)| {
                let a = a as u32;
                map.iter()
                    .filter(move |(b, _)| a < **b)
                    .map(move |(&b, &length)| (a, b, length, widths[&(a, b)]))
            })
            .collect();
        boundaries.sort_unstable_by_key(|b| (b.0, b.1));

        // Regions on the image border are neighbors through the outside too, since what lies
        // behind everything continues beyond the frame: a background split by an object reaching
        // the border is still one background. They share the shorter of their border lengths.
        let mut border: HashMap<u32, u32> = HashMap::default();
        for x in 0..width {
            *border.entry(labels[x]).or_default() += 1;
            *border.entry(labels[(height - 1) * width + x]).or_default() += 1;
        }
        for y in 0..height {
            *border.entry(labels[y * width]).or_default() += 1;
            *border.entry(labels[y * width + width - 1]).or_default() += 1;
        }
        let mut border: Vec<(u32, u32)> = border.into_iter().collect();
        border.sort_unstable();
        for (k, &(a, length_a)) in border.iter().enumerate() {
            for &(b, length_b) in &border[k + 1..] {
                neighbors[a as usize]
                    .entry(b)
                    .or_insert(length_a.min(length_b));
                neighbors[b as usize]
                    .entry(a)
                    .or_insert(length_a.min(length_b));
            }
        }

        let cost = |nodes: &[Node], a: u32, b: u32, length: u32| {
            nodes[a as usize].stats.merge_cost(&nodes[b as usize].stats) / length as f64
        };
        let mut heap = BinaryHeap::new();
        for (a, map) in neighbors.iter().enumerate() {
            if a % (POLL / 4) == 0 {
                crate::progress::poll().await;
            }
            for (&b, &length) in map {
                if (a as u32) < b {
                    heap.push(Merge {
                        cost: cost(&nodes, a as u32, b, length),
                        a: a as u32,
                        b,
                    });
                }
            }
        }

        // Merged nodes are replaced by a new one, so a queued merge is current while both live
        let mut alive = vec![true; count];
        let merge = |nodes: &mut Vec<Node>,
                     neighbors: &mut Vec<HashMap<u32, u32>>,
                     alive: &mut Vec<bool>,
                     a: u32,
                     b: u32,
                     scale: f64| {
            let c = nodes.len() as u32;
            let (na, nb) = (&nodes[a as usize], &nodes[b as usize]);
            let scale = scale.max(na.scale).max(nb.scale);
            nodes.push(Node {
                stats: na.stats.add(&nb.stats),
                children: Some([a, b]),
                parent: None,
                scale,
                min: na.min.min(nb.min),
                max: na.max.max(nb.max),
                perimeter: 0,
                frame: na.frame + nb.frame,
                blur: 1.,
                leaves: (0, 0),
            });
            alive[a as usize] = false;
            alive[b as usize] = false;
            alive.push(true);
            nodes[a as usize].parent = Some(c);
            nodes[b as usize].parent = Some(c);

            let mut merged = std::mem::take(&mut neighbors[a as usize]);
            let mut other = std::mem::take(&mut neighbors[b as usize]);
            if merged.len() < other.len() {
                std::mem::swap(&mut merged, &mut other);
            }
            for (n, length) in other {
                *merged.entry(n).or_default() += length;
            }
            merged.remove(&a);
            merged.remove(&b);
            for (&n, &length) in &merged {
                let map = &mut neighbors[n as usize];
                map.remove(&a);
                map.remove(&b);
                map.insert(c, length);
            }
            neighbors.push(merged);
            c
        };
        let mut popped = 0usize;
        while let Some(Merge { cost: scale, a, b }) = heap.pop() {
            popped += 1;
            if popped.is_multiple_of(POLL / 16) {
                crate::progress::poll().await;
            }
            if !alive[a as usize] || !alive[b as usize] {
                continue;
            }
            let c = merge(&mut nodes, &mut neighbors, &mut alive, a, b, scale);
            for (&n, &length) in &neighbors[c as usize] {
                heap.push(Merge {
                    cost: cost(&nodes, c, n, length),
                    a: c,
                    b: n,
                });
            }
        }
        // The pixel grid is connected, so one root remains; join any others to be safe
        let mut roots = (0..nodes.len() as u32).filter(|&n| alive[n as usize]);
        let mut root = roots.next().expect("an image has pixels");
        let others: Vec<u32> = roots.collect();
        for other in others {
            root = merge(
                &mut nodes,
                &mut neighbors,
                &mut alive,
                root,
                other,
                f64::INFINITY,
            );
        }

        // Number the leaves depth first, so each node's leaves form a range
        let mut rank = vec![0; count];
        let mut order = Vec::with_capacity(count);
        let mut next = 0;
        let mut stack = vec![(root, false)];
        while let Some((node, done)) = stack.pop() {
            match (nodes[node as usize].children, done) {
                (None, _) => {
                    rank[node as usize] = next;
                    order.push(node);
                    nodes[node as usize].leaves = (next, next + 1);
                    next += 1;
                }
                (Some([a, b]), false) => {
                    nodes[node as usize].leaves.0 = next;
                    stack.extend([(node, true), (b, false), (a, false)]);
                }
                (Some(_), true) => nodes[node as usize].leaves.1 = next,
            }
        }

        // The pixels sorted by the rank of their leaf, by counting
        let mut pixel_start = vec![0u32; count + 1];
        for &label in &labels {
            pixel_start[rank[label as usize] as usize + 1] += 1;
        }
        for k in 0..count {
            pixel_start[k + 1] += pixel_start[k];
        }
        let mut pixels = vec![0u32; labels.len()];
        let mut fill = pixel_start.clone();
        for (i, &label) in labels.iter().enumerate() {
            let r = rank[label as usize] as usize;
            pixels[fill[r] as usize] = i as u32;
            fill[r] += 1;
        }

        let mut hierarchy = Self {
            nodes,
            root,
            width,
            colors: colors.to_vec(),
            labels,
            rank,
            order,
            pixels,
            pixel_start,
        };
        crate::progress::poll().await;
        hierarchy.measure_perimeters(&boundaries).await;
        hierarchy
    }

    /// Fills in every node's perimeter from the `boundaries` between leaves `(a, b, length)`: a
    /// boundary lies on the perimeter of every node containing one of its sides but not the
    /// other, those below the lowest node containing both.
    async fn measure_perimeters(&mut self, boundaries: &[(u32, u32, u32, f64)]) {
        let mut contribution = vec![0i64; self.nodes.len()];
        let mut width_contribution = vec![0f64; self.nodes.len()];
        for (k, &(a, b, length, width)) in boundaries.iter().enumerate() {
            if k % (POLL / 16) == 0 {
                crate::progress::poll().await;
            }
            contribution[a as usize] += length as i64;
            contribution[b as usize] += length as i64;
            width_contribution[a as usize] += width;
            width_contribution[b as usize] += width;
            // The lowest common ancestor: up from `a` until the node holds `b`
            let mut at = a;
            while !self.within(b, at) {
                at = self.nodes[at as usize]
                    .parent
                    .expect("a has an ancestor holding b");
            }
            contribution[at as usize] -= 2 * length as i64;
            width_contribution[at as usize] -= 2. * width;
        }
        // Children come before their parents, so one pass sums each subtree
        for k in 0..self.nodes.len() {
            let total = contribution[k];
            let width = width_contribution[k];
            self.nodes[k].perimeter = total.max(0) as u32;
            self.nodes[k].blur = if total > 0 {
                (width / total as f64).clamp(1., 8.) as f32
            } else {
                1.
            };
            if let Some(parent) = self.nodes[k].parent {
                contribution[parent as usize] += total;
                width_contribution[parent as usize] += width;
            }
        }
    }

    /// What lies behind everything: of the two parts the image first splits into, the one along
    /// more of the frame.
    pub fn background(&self, height: usize) -> u32 {
        let width = self.width;
        let frame_length = |node: u32| {
            (0..width)
                .flat_map(|x| [x, (height - 1) * width + x])
                .chain((0..height).flat_map(|y| [y * width, y * width + width - 1]))
                .filter(|&i| self.contains(node, i))
                .count()
        };
        self.nodes[self.root as usize]
            .children
            .and_then(|parts| parts.into_iter().max_by_key(|&p| frame_length(p)))
            .unwrap_or(self.root)
    }

    /// Whether `node` is `ancestor` or lies within it.
    pub fn within(&self, node: u32, ancestor: u32) -> bool {
        let (lo, hi) = self.nodes[node as usize].leaves;
        let (a_lo, a_hi) = self.nodes[ancestor as usize].leaves;
        a_lo <= lo && hi <= a_hi
    }

    /// The leaf (region of the segmentation) of pixel `i`.
    pub fn leaf(&self, i: usize) -> u32 {
        self.labels[i]
    }

    /// The leaf of every pixel.
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// The depth-first ranks of the leaves of `node`, a range.
    pub fn leaf_ranks(&self, node: u32) -> (u32, u32) {
        self.nodes[node as usize].leaves
    }

    /// The pixels (row-major indices) of `node`, leaf by leaf in the order of their ranks.
    pub fn pixels(&self, node: u32) -> &[u32] {
        let (lo, hi) = self.nodes[node as usize].leaves;
        &self.pixels[self.pixel_start[lo as usize] as usize..self.pixel_start[hi as usize] as usize]
    }

    /// The leaves of `node`.
    pub fn leaves(&self, node: u32) -> &[u32] {
        let (lo, hi) = self.nodes[node as usize].leaves;
        &self.order[lo as usize..hi as usize]
    }

    /// The smallest scale in [`MIXED_STEPS`] at which at least `share` of the node's rim is
    /// mixed (see `mixed_rim_at`): one for an anti-aliased edge, the block size for
    /// pixel art that renders a gradient or a blur in blocks, `None` for a crisp edge.
    pub fn antialias_scale(
        &self,
        node: u32,
        rgb: &[glam::Vec3],
        own: DVec3,
        share: f64,
    ) -> Option<i32> {
        MIXED_STEPS
            .iter()
            .copied()
            .find(|&step| self.mixed_rim_at(node, rgb, own, step) >= share)
    }

    fn mixed_rim_at(&self, node: u32, rgb: &[glam::Vec3], own: DVec3, step: i32) -> f64 {
        let n = &self.nodes[node as usize];
        let (w, h) = (self.width as i32, (self.labels.len() / self.width) as i32);
        let color = |k: usize| rgb[k].as_dvec3();
        let at = |x: i32, y: i32| -> Option<usize> {
            (x >= 0 && y >= 0 && x < w && y < h).then(|| y as usize * self.width + x as usize)
        };
        let (mut mixed, mut counted) = (0usize, 0usize);
        for y in n.min.y..n.max.y {
            for x in n.min.x..n.max.x {
                let Some(i) = at(x, y) else { continue };
                if !self.contains(node, i) {
                    continue;
                }
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    // An edge of the node here
                    let Some(next) = at(x + dx, y + dy) else {
                        continue;
                    };
                    if self.contains(node, next) {
                        continue;
                    }
                    // Inside a straight run of the edge along the pixel grid, which is crisp
                    // whether drawn as pixel art or as vector art on the grid, and so says
                    // nothing
                    let along = |k: i32| -> bool {
                        let (u, v) = (x + k * dy, y + k * dx);
                        at(u, v).is_some_and(|p| self.contains(node, p))
                            && at(u + dx, v + dy).is_some_and(|q| !self.contains(node, q))
                    };
                    if along(1) && along(-1) {
                        continue;
                    }
                    let (Some(j), Some(far)) = (
                        at(x + step * dx, y + step * dy),
                        at(x + 2 * step * dx, y + 2 * step * dy),
                    ) else {
                        continue;
                    };
                    if self.contains(node, j) || self.contains(node, far) {
                        continue;
                    }
                    let span = color(far) - own;
                    let length = span.length_squared();
                    if length < MIXED_CONTRAST * MIXED_CONTRAST {
                        continue;
                    }
                    counted += 1;
                    // A mixture lies between the two sides and on the line between them in
                    // color space; a third color in a row does not
                    let between = |k: usize| {
                        let d = color(k) - own;
                        let t = d.dot(span) / length;
                        (MIXED_LOW..MIXED_HIGH).contains(&t)
                            && (d - span * t).length() <= MIXED_OFF_LINE
                    };
                    if between(i) || between(j) {
                        mixed += 1;
                    }
                }
            }
        }
        if counted == 0 {
            0.
        } else {
            mixed as f64 / counted as f64
        }
    }

    /// Color statistics of the node's own pixels (outside `covered`, more than `radius` from
    /// anything else) within `band` of its outline, the color a primitive has to match. The
    /// node's stats when under a quarter of the band survives the erosion.
    pub fn own_rim_stats(&self, node: u32, covered: &[u32], radius: i32, band: i32) -> Stats {
        let n = &self.nodes[node as usize];
        let (w, h) = (self.width as i32, (self.labels.len() / self.width) as i32);
        let is_own =
            |i: usize| self.contains(node, i) && !covered.iter().any(|&c| self.contains(c, i));
        // Over the node's box grown by the larger reach: what is not the node (the frame's
        // outside counts), and what is the node's own; each window then read off prefix counts
        let r = band.max(radius).max(0);
        let (x0, y0) = (n.min.x - r, n.min.y - r);
        let (bw, bh) = (
            (n.max.x - n.min.x + 2 * r) as usize,
            (n.max.y - n.min.y + 2 * r) as usize,
        );
        let mut not_node = vec![false; bw * bh];
        let mut own = vec![false; bw * bh];
        for by in 0..bh {
            for bx in 0..bw {
                let (x, y) = (x0 + bx as i32, y0 + by as i32);
                let j = by * bw + bx;
                if x < 0 || y < 0 || x >= w || y >= h {
                    not_node[j] = true;
                    continue;
                }
                let i = y as usize * self.width + x as usize;
                not_node[j] = !self.contains(node, i);
                own[j] = !not_node[j] && is_own(i);
            }
        }
        // Whether any of `mask` lies in the square of radius `q` about each pixel of the box,
        // rows then columns, by prefix counts
        let any_within = |mask: &[bool], q: i32| -> Vec<bool> {
            let q = q.max(0) as usize;
            let mut rows = vec![false; bw * bh];
            let mut prefix = vec![0u32; bw.max(bh) + 1];
            for by in 0..bh {
                for bx in 0..bw {
                    prefix[bx + 1] = prefix[bx] + u32::from(mask[by * bw + bx]);
                }
                for bx in 0..bw {
                    let (lo, hi) = (bx.saturating_sub(q), (bx + q + 1).min(bw));
                    rows[by * bw + bx] = prefix[hi] > prefix[lo];
                }
            }
            let mut out = vec![false; bw * bh];
            for bx in 0..bw {
                for by in 0..bh {
                    prefix[by + 1] = prefix[by] + u32::from(rows[by * bw + bx]);
                }
                for by in 0..bh {
                    let (lo, hi) = (by.saturating_sub(q), (by + q + 1).min(bh));
                    out[by * bw + bx] = prefix[hi] > prefix[lo];
                }
            }
            out
        };
        let near_outline = any_within(&not_node, band);
        let not_own: Vec<bool> = own.iter().map(|o| !o).collect();
        let crowded = any_within(&not_own, radius);
        let mut stats = Stats::default();
        let mut band_pixels = 0f64;
        for y in n.min.y..n.max.y {
            for x in n.min.x..n.max.x {
                let j = (y - y0) as usize * bw + (x - x0) as usize;
                // Along the node's outline: something not the node within the band (a covered
                // child inside is not the outline); counted where nothing else is within radius
                if !own[j] || !near_outline[j] {
                    continue;
                }
                band_pixels += 1.;
                if !crowded[j] {
                    let i = y as usize * self.width + x as usize;
                    stats = stats.add(&Stats::pixel(self.colors[i]));
                }
            }
        }
        if stats.n < 0.25 * band_pixels {
            n.stats
        } else {
            stats
        }
    }

    pub fn contains(&self, node: u32, i: usize) -> bool {
        let (lo, hi) = self.nodes[node as usize].leaves;
        (lo..hi).contains(&self.rank[self.labels[i] as usize])
    }

    pub fn width(&self) -> usize {
        self.width
    }
}
