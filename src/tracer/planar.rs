use glam::IVec2;

use crate::HashMap;

/// Label of everything outside the image, so the image border is a boundary like any other.
pub const OUTSIDE: u32 = u32::MAX;

/// A boundary between two regions running along pixel cracks from one junction to another (or
/// around a closed loop, in which case the first and last points coincide).
#[derive(Clone, Debug)]
pub struct Chain {
    pub points: Vec<IVec2>,
    /// Region to the left of the walking direction (in y-down coordinates).
    pub left: u32,
    pub right: u32,
}

impl Chain {
    pub fn is_closed(&self) -> bool {
        self.points.first() == self.points.last()
    }
}

/// A chain used as part of a region's boundary, reversed when the region is on its right, so the
/// region is always on the left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainUse {
    pub chain: usize,
    pub reversed: bool,
}

pub struct PlanarMap {
    pub chains: Vec<Chain>,
    /// Boundary loops per region, the region always to the left.
    pub loops: Vec<Vec<Vec<ChainUse>>>,
}

const DIRECTIONS: [IVec2; 4] = [
    IVec2::new(1, 0),
    IVec2::new(0, 1),
    IVec2::new(-1, 0),
    IVec2::new(0, -1),
];

struct Grid<'a> {
    labels: &'a [u32],
    width: i32,
    height: i32,
}

impl Grid<'_> {
    fn label(&self, p: IVec2) -> u32 {
        if p.x < 0 || p.y < 0 || p.x >= self.width || p.y >= self.height {
            OUTSIDE
        } else {
            self.labels[(p.y * self.width + p.x) as usize]
        }
    }

    /// The pixels left and right of the crack from `corner` one step in `dir`.
    fn sides(&self, corner: IVec2, dir: IVec2) -> (u32, u32) {
        let (left, right) = match (dir.x, dir.y) {
            (1, 0) => (corner + IVec2::new(0, -1), corner),
            (0, 1) => (corner, corner + IVec2::new(-1, 0)),
            (-1, 0) => (corner + IVec2::new(-1, 0), corner + IVec2::new(-1, -1)),
            _ => (corner + IVec2::new(-1, -1), corner + IVec2::new(0, -1)),
        };
        (self.label(left), self.label(right))
    }

    fn is_crack(&self, corner: IVec2, dir: IVec2) -> bool {
        let (l, r) = self.sides(corner, dir);
        l != r
    }

    fn degree(&self, corner: IVec2) -> usize {
        DIRECTIONS
            .iter()
            .filter(|&&d| self.is_crack(corner, d))
            .count()
    }

    fn is_junction(&self, corner: IVec2) -> bool {
        let image_corner =
            (corner.x == 0 || corner.x == self.width) && (corner.y == 0 || corner.y == self.height);
        let degree = self.degree(corner);
        degree > 2 || (degree == 2 && image_corner)
    }

    /// Index of the undirected crack starting at `corner` in `dir`, for marking cracks as used.
    fn crack_index(&self, corner: IVec2, dir: IVec2) -> usize {
        let (p, horizontal) = match (dir.x, dir.y) {
            (1, 0) => (corner, true),
            (-1, 0) => (corner - IVec2::X, true),
            (0, 1) => (corner, false),
            _ => (corner - IVec2::Y, false),
        };
        let index = (p.y * (self.width + 1) + p.x) as usize * 2;
        index + usize::from(!horizontal)
    }
}

pub fn build(labels: &[u32], width: usize, height: usize, region_count: usize) -> PlanarMap {
    let grid = Grid {
        labels,
        width: width as i32,
        height: height as i32,
    };
    let mut used = vec![false; (width + 1) * (height + 1) * 2];
    let mut chains = Vec::new();

    let walk = |start: IVec2, dir: IVec2, used: &mut Vec<bool>| {
        let (left, right) = grid.sides(start, dir);
        let mut points = vec![start];
        let (mut corner, mut dir) = (start, dir);
        loop {
            used[grid.crack_index(corner, dir)] = true;
            corner += dir;
            points.push(corner);
            if corner == start || grid.is_junction(corner) {
                break;
            }
            // Regular corners have exactly one other crack, which separates the same two regions
            let back = -dir;
            dir = *DIRECTIONS
                .iter()
                .find(|&&d| d != back && grid.is_crack(corner, d))
                .expect("regular corner continues");
        }
        Chain {
            points,
            left,
            right,
        }
    };

    for y in 0..=grid.height {
        for x in 0..=grid.width {
            let corner = IVec2::new(x, y);
            if !grid.is_junction(corner) {
                continue;
            }
            for dir in DIRECTIONS {
                if grid.is_crack(corner, dir) && !used[grid.crack_index(corner, dir)] {
                    chains.push(walk(corner, dir, &mut used));
                }
            }
        }
    }
    // What remains are closed boundaries without junctions, like an island inside one region
    for y in 0..=grid.height {
        for x in 0..=grid.width {
            let corner = IVec2::new(x, y);
            for dir in [IVec2::X, IVec2::Y] {
                if grid.is_crack(corner, dir) && !used[grid.crack_index(corner, dir)] {
                    chains.push(walk(corner, dir, &mut used));
                }
            }
        }
    }

    let loops = region_loops(&chains, region_count);
    PlanarMap { chains, loops }
}

fn use_endpoints(chains: &[Chain], u: ChainUse) -> (IVec2, IVec2, IVec2, IVec2) {
    let p = &chains[u.chain].points;
    let n = p.len();
    if u.reversed {
        (p[n - 1], p[n - 2] - p[n - 1], p[0], p[0] - p[1])
    } else {
        (p[0], p[1] - p[0], p[n - 1], p[n - 1] - p[n - 2])
    }
}

/// Links each region's chain uses into closed loops. Where a region touches a corner twice (two
/// of its pixels meet diagonally), the walk turns left first, which keeps to the boundary of the
/// pixel it came from and so treats diagonal neighbors as separate.
fn region_loops(chains: &[Chain], region_count: usize) -> Vec<Vec<Vec<ChainUse>>> {
    let mut starts: Vec<HashMap<IVec2, Vec<ChainUse>>> = vec![HashMap::default(); region_count];
    for (i, chain) in chains.iter().enumerate() {
        for (region, reversed) in [(chain.left, false), (chain.right, true)] {
            if region != OUTSIDE {
                let u = ChainUse { chain: i, reversed };
                starts[region as usize]
                    .entry(use_endpoints(chains, u).0)
                    .or_default()
                    .push(u);
            }
        }
    }

    let turn_rank = |incoming: IVec2, outgoing: IVec2| {
        let left = IVec2::new(incoming.y, -incoming.x);
        if outgoing == left {
            0
        } else if outgoing == incoming {
            1
        } else if outgoing == -left {
            2
        } else {
            3
        }
    };

    starts
        .into_iter()
        .map(|mut starts| {
            let mut loops = Vec::new();
            while let Some(first) = starts.values_mut().find_map(Vec::pop) {
                let first_start = use_endpoints(chains, first).0;
                let mut current = vec![first];
                let mut last = first;
                loop {
                    let (_, _, end, incoming) = use_endpoints(chains, last);
                    let candidates = starts.entry(end).or_default();
                    let best = candidates
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, u)| turn_rank(incoming, use_endpoints(chains, **u).1));
                    let closes = end == first_start;
                    let close_rank = turn_rank(incoming, use_endpoints(chains, first).1);
                    match best {
                        Some((i, u))
                            if !closes
                                || turn_rank(incoming, use_endpoints(chains, *u).1)
                                    < close_rank =>
                        {
                            last = candidates.swap_remove(i);
                            current.push(last);
                        }
                        // Closed, or an open walk (an inconsistent map): keep what we have
                        _ => break,
                    }
                }
                loops.push(current);
            }
            loops
        })
        .collect()
}
