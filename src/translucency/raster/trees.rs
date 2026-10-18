//! Nearest neighbours in the Chebyshev metric: a k-d tree and a grid.

use super::{DVec3, cheb};

/// Nearest neighbors among colors in the Chebyshev metric (as scipy's `cKDTree` with
/// `p=inf`): a balanced tree of boxes, leaves of up to 16 points.
pub struct ChebTree {
    pts: Vec<DVec3>,
    idx: Vec<usize>,
    nodes: Vec<Node>,
}

struct Node {
    lo: DVec3,
    hi: DVec3,
    start: usize,
    end: usize,
    kids: Option<(usize, usize)>,
}

const LEAF: usize = 16;

impl ChebTree {
    pub fn new(pts: &[DVec3]) -> Self {
        let mut t = ChebTree {
            pts: pts.to_vec(),
            idx: (0..pts.len()).collect(),
            nodes: Vec::new(),
        };
        if !pts.is_empty() {
            t.build(0, pts.len());
        }
        t
    }

    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    fn build(&mut self, start: usize, end: usize) -> usize {
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for &i in &self.idx[start..end] {
            lo = lo.min(self.pts[i]);
            hi = hi.max(self.pts[i]);
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            lo,
            hi,
            start,
            end,
            kids: None,
        });
        let spread = hi - lo;
        if end - start <= LEAF || spread.max_element() == 0.0 {
            return id;
        }
        let d = if spread.x >= spread.y && spread.x >= spread.z {
            0
        } else if spread.y >= spread.z {
            1
        } else {
            2
        };
        let mid = start + (end - start) / 2;
        let pts = &self.pts;
        self.idx[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            pts[a][d].total_cmp(&pts[b][d]).then(a.cmp(&b))
        });
        let a = self.build(start, mid);
        let b = self.build(mid, end);
        self.nodes[id].kids = Some((a, b));
        id
    }

    fn box_dist(n: &Node, q: DVec3) -> f64 {
        let d = (n.lo - q).max(q - n.hi).max(DVec3::ZERO);
        d.max_element()
    }

    /// The nearest point within `bound` (exclusive), its distance and index; the lowest
    /// index among points at the same distance.
    pub fn nearest(&self, q: DVec3, bound: f64) -> Option<(f64, usize)> {
        if self.pts.is_empty() {
            return None;
        }
        let mut best: Option<(f64, usize)> = None;
        self.search(0, q, bound, &mut best);
        best
    }

    fn search(
        &self,
        node: usize,
        q: DVec3,
        bound: f64,
        best: &mut Option<(f64, usize)>,
    ) {
        let n = &self.nodes[node];
        let limit = best.map_or(bound, |b| b.0);
        let bd = Self::box_dist(n, q);
        // a point at the best distance with a lower index still counts
        if bd > limit || (best.is_none() && bd >= bound) {
            return;
        }
        match n.kids {
            None => {
                for &i in &self.idx[n.start..n.end] {
                    let d = cheb(self.pts[i], q);
                    if d >= bound {
                        continue;
                    }
                    match *best {
                        Some((bd, bi)) if d > bd || (d == bd && i > bi) => {}
                        _ => *best = Some((d, i)),
                    }
                }
            }
            Some((a, b)) => {
                let (da, db) = (
                    Self::box_dist(&self.nodes[a], q),
                    Self::box_dist(&self.nodes[b], q),
                );
                let (first, second) = if da <= db { (a, b) } else { (b, a) };
                self.search(first, q, bound, best);
                self.search(second, q, bound, best);
            }
        }
    }

    /// The distance to the nearest point (infinity for none).
    pub fn nearest_dist(&self, q: DVec3) -> f64 {
        self.nearest(q, f64::INFINITY)
            .map_or(f64::INFINITY, |b| b.0)
    }
}

/// [`ChebTree::nearest`] on a grid of cubes, for bounds about the cube's side: only the
/// cubes the query's box reaches (widened past the rounding) are searched, ties to the lowest
/// index as the tree; a query reaching more cubes than there are points scans the points.
pub struct ChebGrid {
    pts: Vec<DVec3>,
    lo: DVec3,
    side: f64,
    dims: [usize; 3],
    /// per cube the range of `order` holding its points, the points by cube then index
    start: Vec<u32>,
    order: Vec<u32>,
}

impl ChebGrid {
    pub fn new(pts: &[DVec3], side: f64) -> Self {
        let mut grid = ChebGrid {
            pts: pts.to_vec(),
            lo: DVec3::ZERO,
            side,
            dims: [0; 3],
            start: Vec::new(),
            order: Vec::new(),
        };
        let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
        for p in pts {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let spans = (hi - lo) / side;
        // no grid (every query scans) without points, for points or a side not finite, or
        // for more cubes than a grid should hold
        if pts.is_empty()
            || !(side > 0.0 && lo.is_finite() && hi.is_finite() && spans.is_finite())
            || spans.max_element() > 1e6
        {
            return grid;
        }
        let dims = spans.to_array().map(|s| s.floor() as usize + 1);
        if dims.iter().product::<usize>() > 1 << 22 || pts.len() >= u32::MAX as usize {
            return grid;
        }
        grid.lo = lo;
        grid.dims = dims;
        let cells: Vec<usize> = pts.iter().map(|p| grid.cube(*p)).collect();
        let mut start = vec![0u32; dims.iter().product::<usize>() + 1];
        for &c in &cells {
            start[c + 1] += 1;
        }
        for c in 0..start.len() - 1 {
            start[c + 1] += start[c];
        }
        let mut next = start.clone();
        let mut order = vec![0u32; pts.len()];
        for (i, &c) in cells.iter().enumerate() {
            order[next[c] as usize] = i as u32;
            next[c] += 1;
        }
        grid.start = start;
        grid.order = order;
        grid
    }

    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    /// The cube along axis `a` of coordinate `v`, clamped to the grid.
    fn axis(&self, v: f64, a: usize) -> usize {
        let s = ((v - self.lo[a]) / self.side).floor();
        if s >= (self.dims[a] - 1) as f64 {
            self.dims[a] - 1
        } else if s >= 0.0 {
            s as usize
        } else {
            0
        }
    }

    fn cube(&self, p: DVec3) -> usize {
        let [x, y, z] = [0, 1, 2].map(|a| self.axis(p[a], a));
        (z * self.dims[1] + y) * self.dims[0] + x
    }

    /// The nearest point within `bound` (exclusive), its distance and index; the lowest
    /// index among points at the same distance.
    pub fn nearest(&self, q: DVec3, bound: f64) -> Option<(f64, usize)> {
        if self.pts.is_empty() {
            return None;
        }
        let mut best: Option<(f64, usize)> = None;
        let mut consider = |i: usize| {
            let d = cheb(self.pts[i], q);
            if d >= bound {
                return;
            }
            match best {
                Some((bd, bi)) if d > bd || (d == bd && i > bi) => {}
                _ => best = Some((d, i)),
            }
        };
        let reach = bound * (1.0 + 1e-9) + 1e-12;
        let ranges: Option<[(usize, usize); 3]> = (!self.start.is_empty()
            && q.is_finite()
            && reach.is_finite())
        .then(|| [0, 1, 2].map(|a| (self.axis(q[a] - reach, a), self.axis(q[a] + reach, a))));
        let cubes = ranges.map_or(usize::MAX, |r| {
            r.iter().map(|(a, b)| b - a + 1).product::<usize>()
        });
        match ranges {
            Some([(x0, x1), (y0, y1), (z0, z1)]) if cubes <= self.pts.len() => {
                for z in z0..=z1 {
                    for y in y0..=y1 {
                        let row = (z * self.dims[1] + y) * self.dims[0];
                        let (from, to) = (self.start[row + x0], self.start[row + x1 + 1]);
                        for &i in &self.order[from as usize..to as usize] {
                            consider(i as usize);
                        }
                    }
                }
            }
            _ => (0..self.pts.len()).for_each(consider),
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Conn, arange, area_px, deep_in, label, length_px, pairwise, skeletonize};
    use super::*;

    /// The grid's nearest point is the tree's, the distance and the index on ties, at bounds
    /// below, at and above the cube's side and for queries inside and outside the points.
    #[test]
    fn the_grid_finds_what_the_tree_finds() {
        let mut seed = 11u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for n in [0, 1, 5, 300, 3000] {
            // colours on the 8-bit levels, so that some lie at equal distances and repeat
            let pts: Vec<DVec3> = (0..n)
                .map(|_| {
                    DVec3::new(next(), next(), next() * 0.3).map(|v| (v * 255.0).round() / 255.0)
                })
                .collect();
            let tree = ChebTree::new(&pts);
            for side in [0.01, 0.024, 0.1] {
                let grid = ChebGrid::new(&pts, side);
                for _ in 0..2000 {
                    let q = DVec3::new(next(), next(), next()) * 1.4 - 0.2;
                    let q = if next() < 0.3 {
                        (q * 255.0).round() / 255.0
                    } else {
                        q
                    };
                    for bound in [side * 0.5, side, side * 2.5, 1.0, f64::INFINITY] {
                        assert_eq!(
                            grid.nearest(q, bound).map(|(d, i)| (d.to_bits(), i)),
                            tree.nearest(q, bound).map(|(d, i)| (d.to_bits(), i))
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_unit_is_one_at_330() {
        assert_eq!(length_px(330, 2.0), 2);
        assert_eq!(area_px(330, 16.0), 16.0);
        assert_eq!(length_px(660, 2.0), 4);
        assert_eq!(area_px(660, 16.0), 64.0);
        assert_eq!(length_px(100, 1.0), 1);
    }

    #[test]
    fn deep_in_is_the_square_erosion() {
        // a 7 x 5 block in a 12 x 9 mask: radius 1 keeps 5 x 3, radius 2 keeps 3 x 1
        let (w, h) = (12, 9);
        let m: Vec<bool> = (0..w * h)
            .map(|i| (2..9).contains(&(i % w)) && (2..7).contains(&(i / w)))
            .collect();
        let count = |r| deep_in(&m, w, h, r).iter().filter(|v| **v).count();
        assert_eq!(count(1), 15);
        assert_eq!(count(2), 3);
        // the frame counts as inside
        let full = vec![true; w * h];
        assert!(deep_in(&full, w, h, 3).iter().all(|v| *v));
    }

    #[test]
    fn pairwise_matches_the_blocks() {
        let a: Vec<f64> = (0..300).map(|i| (i as f64 * 0.37).sin()).collect();
        // the halves of 300 are 144 and 156, each summed in blocks of eight
        assert_eq!(pairwise(&a), pairwise(&a[..144]) + pairwise(&a[144..]));
        assert_eq!(pairwise(&a[..5]), (((a[0] + a[1]) + a[2]) + a[3]) + a[4]);
    }

    #[test]
    fn labels_in_raster_order() {
        let m = [true, false, true, false, false, false, true, false, false];
        let (lab, n) = label(&m, 3, 3, Conn::Four);
        assert_eq!(n, 3);
        assert_eq!(lab, vec![1, 0, 2, 0, 0, 0, 3, 0, 0]);
    }

    #[test]
    fn skeleton_of_a_bar_as_scikit_image() {
        let (w, h) = (12, 5);
        let mut m = vec![false; w * h];
        for y in 1..4 {
            for x in 1..11 {
                m[y * w + x] = true;
            }
        }
        let sk = skeletonize(&m, w, h);
        // scikit-image 0.26 on the same bar: the line along row 2 and a stub at its end
        let on: Vec<(usize, usize)> = (0..w * h)
            .filter(|&i| sk[i])
            .map(|i| (i / w, i % w))
            .collect();
        let mut want: Vec<(usize, usize)> = (1..9).map(|x| (2, x)).collect();
        want.insert(0, (1, 9));
        assert_eq!(on, want);
    }

    #[test]
    fn tree_finds_the_lowest_index_on_ties() {
        let pts: Vec<DVec3> = (0..100)
            .map(|i| DVec3::new((i % 7) as f64, (i % 3) as f64, 0.0))
            .collect();
        let t = ChebTree::new(&pts);
        let q = DVec3::new(3.2, 1.0, 0.0);
        let brute = (0..100)
            .map(|i| (cheb(pts[i], q), i))
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
            .unwrap();
        assert_eq!(t.nearest(q, 10.0), Some(brute));
        assert_eq!(t.nearest(q, 0.2), None);
    }

    #[test]
    fn arange_as_numpy() {
        let k = arange(0.1, 0.951, 0.02);
        assert_eq!(k.len(), 43);
        assert_eq!(k[1], 0.1 + 0.02);
    }
}
