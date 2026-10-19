use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::{DVec2, DVec3};

use crate::HashMap;

/// Sufficient statistics of a set of pixels for fitting flat and affine color models in O(1).
#[derive(Clone, Copy, Debug, Default)]
pub struct Moments {
    n: f64,
    p: DVec2,
    xx: f64,
    xy: f64,
    yy: f64,
    c: DVec3,
    cx: DVec3,
    cy: DVec3,
    cc: DVec3,
}

impl Moments {
    pub fn pixel(x: f64, y: f64, c: DVec3) -> Self {
        Self {
            n: 1.,
            p: DVec2::new(x, y),
            xx: x * x,
            xy: x * y,
            yy: y * y,
            c,
            cx: c * x,
            cy: c * y,
            cc: c * c,
        }
    }

    pub(crate) fn add(&self, o: &Self) -> Self {
        Self {
            n: self.n + o.n,
            p: self.p + o.p,
            xx: self.xx + o.xx,
            xy: self.xy + o.xy,
            yy: self.yy + o.yy,
            c: self.c + o.c,
            cx: self.cx + o.cx,
            cy: self.cy + o.cy,
            cc: self.cc + o.cc,
        }
    }

    fn flat_sse(&self) -> f64 {
        (self.cc - self.c * self.c / self.n).element_sum().max(0.)
    }

    /// Centered second moments `(sxx, sxy, syy)` and the color/position covariances.
    fn centered(&self) -> (f64, f64, f64, DVec3, DVec3) {
        let m = self.p / self.n;
        let sxx = self.xx - self.p.x * m.x;
        let sxy = self.xy - self.p.x * m.y;
        let syy = self.yy - self.p.y * m.y;
        let scx = self.cx - self.c * m.x;
        let scy = self.cy - self.c * m.y;
        (sxx, sxy, syy, scx, scy)
    }

    /// The least squares color gradient, with a small ridge so thin or tiny regions do not get
    /// wild gradients from nearly collinear pixels.
    fn gradient(&self) -> (DVec3, DVec3) {
        let (sxx, sxy, syy, scx, scy) = self.centered();
        let ridge = 0.05 * self.n;
        let (sxx, syy) = (sxx + ridge, syy + ridge);
        let det = sxx * syy - sxy * sxy;
        ((scx * syy - scy * sxy) / det, (scy * sxx - scx * sxy) / det)
    }

    fn affine_sse(&self) -> f64 {
        let (_, _, _, scx, scy) = self.centered();
        let (gx, gy) = self.gradient();
        (self.flat_sse() - (gx * scx + gy * scy).element_sum()).max(0.)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SegmentParams {
    /// Cost per pixel edge of boundary, against squared color error summed over channels.
    pub boundary_cost: f64,
    /// Extra cost of giving a region a gradient instead of a flat color.
    pub gradient_cost: f64,
    /// Regions merge while the RMS color step across their shared boundary is below this, so
    /// smooth ramps join up while real edges stay.
    pub contrast: f64,
    /// A merge by contrast is refused when no color model fits the merged region within this RMS
    /// error, which stops soft edges from chaining two different colors together.
    pub max_fit_error: f64,
}

/// Regions merged between two checkpoints (see [`crate::progress`]), about a millisecond.
const MERGES_PER_POLL: usize = 1024;

/// Packed to four bytes, so a neighbour's entry takes 16 bytes instead of 24.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, packed(4))]
struct Boundary {
    len: u32,
    /// Summed squared color difference of the pixel pairs across the boundary.
    contrast: f64,
}

impl Boundary {
    fn add(&mut self, o: &Boundary) {
        self.len += o.len;
        self.contrast += o.contrast;
    }
}

#[derive(Clone, Copy)]
enum Criterion {
    /// Merge across low contrast boundaries, ignoring how the region's colors fit as a whole.
    Contrast,
    /// Merge while the fit error plus boundary cost (Mumford-Shah energy) goes down.
    Energy,
}

struct Region {
    moments: Moments,
    energy: f64,
    /// Neighbor region and the length of the shared boundary.
    neighbors: HashMap<u32, Boundary>,
    version: u32,
}

impl Region {
    fn new(moments: Moments, params: &SegmentParams) -> Self {
        Self {
            energy: energy(&moments, params),
            moments,
            neighbors: HashMap::default(),
            version: 0,
        }
    }
}

/// The fit error of the cheaper colour model, flat or affine (a gradient at its price).
fn energy(m: &Moments, params: &SegmentParams) -> f64 {
    let flat = m.flat_sse();
    // Fewer than a handful of pixels cannot support a meaningful gradient
    if m.n < 6. {
        return flat;
    }
    let affine = m.affine_sse() + params.gradient_cost;
    if affine < flat { affine } else { flat }
}

struct Candidate {
    cost: f64,
    a: u32,
    b: u32,
    version_a: u32,
    version_b: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    // Reversed, so the max-heap pops the cheapest merge first
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.total_cmp(&self.cost)
    }
}

/// The root of `i` in the union-find forest `parent`, halving the paths walked.
fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let grandparent = parent[parent[i as usize] as usize];
        parent[i as usize] = grandparent;
        i = grandparent;
    }
    i
}

pub struct Segmentation {
    /// Region index per pixel, row-major.
    pub labels: Vec<u32>,
}

/// Connected regions of pixels with exactly the same color, the cheapest start for
/// [`segment_seeded`] since no merge can split them. Returns the label per pixel and the number
/// of regions.
pub fn equal_color_regions(rgb: &[u8], width: usize, height: usize) -> (Vec<u32>, usize) {
    let color = |i: usize| &rgb[3 * i..3 * i + 3];
    let mut parent: Vec<u32> = (0..(width * height) as u32).collect();
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            for j in [
                (x + 1 < width).then_some(i + 1),
                (y + 1 < height).then_some(i + width),
            ]
            .into_iter()
            .flatten()
            {
                if color(i) == color(j) {
                    let (a, b) = (find(&mut parent, i as u32), find(&mut parent, j as u32));
                    if a != b {
                        parent[a.max(b) as usize] = a.min(b);
                    }
                }
            }
        }
    }
    let mut labels = vec![0; width * height];
    let mut index = HashMap::default();
    for (i, label) in labels.iter_mut().enumerate() {
        let root = find(&mut parent, i as u32);
        let next = index.len() as u32;
        *label = *index.entry(root).or_insert(next);
    }
    let count = index.len();
    (labels, count)
}

/// Greedy region merging that minimizes the fit error of every region plus the boundary cost (a
/// discrete Mumford-Shah energy), from the regions of `seeds` (a label per pixel, `count`
/// regions), which the merging would form anyway, like [`equal_color_regions`].
pub async fn segment_seeded(
    colors: &[DVec3],
    width: usize,
    height: usize,
    params: &SegmentParams,
    seeds: &[u32],
    count: usize,
) -> Segmentation {
    let mut moments = vec![Moments::default(); count];
    for (i, (&c, &seed)) in colors.iter().zip(seeds).enumerate() {
        if i % (MERGES_PER_POLL * 64) == 0 {
            crate::progress::poll().await;
        }
        let pixel = Moments::pixel((i % width) as f64 + 0.5, (i / width) as f64 + 0.5, c);
        moments[seed as usize] = moments[seed as usize].add(&pixel);
    }
    let mut regions: Vec<Region> = moments
        .into_iter()
        .map(|m| Region::new(m, params))
        .collect();
    for y in 0..height {
        if y % 16 == 0 {
            crate::progress::poll().await;
        }
        for x in 0..width {
            let i = y * width + x;
            let mut link = |j: usize| {
                let (a, b) = (seeds[i], seeds[j]);
                if a == b {
                    return;
                }
                let boundary = Boundary {
                    len: 1,
                    contrast: colors[i].distance_squared(colors[j]),
                };
                regions[a as usize]
                    .neighbors
                    .entry(b)
                    .or_default()
                    .add(&boundary);
                regions[b as usize]
                    .neighbors
                    .entry(a)
                    .or_default()
                    .add(&boundary);
            };
            if x + 1 < width {
                link(i + 1);
            }
            if y + 1 < height {
                link(i + width);
            }
        }
    }

    // Union-find parents, so pixels can be relabeled once at the end
    let mut parent: Vec<u32> = (0..regions.len() as u32).collect();

    // Geometry comes from local contrast alone, so a smooth ramp becomes one region however far
    // apart its end colors are; a final energy pass removes what is too small to be worth its
    // boundary
    merge(&mut regions, &mut parent, params, Criterion::Contrast).await;
    merge(&mut regions, &mut parent, params, Criterion::Energy).await;

    // Compact the surviving regions into consecutive labels
    let mut compact = HashMap::default();
    let labels = seeds
        .iter()
        .map(|&i| {
            let root = find(&mut parent, i);
            let next = compact.len() as u32;
            *compact.entry(root).or_insert(next)
        })
        .collect();
    Segmentation { labels }
}

async fn merge(
    regions: &mut [Region],
    parent: &mut [u32],
    params: &SegmentParams,
    criterion: Criterion,
) {
    // Candidates with a cost below zero are merged, cheapest first
    let merge_cost = |regions: &[Region], a: u32, b: u32| {
        let (ra, rb) = (&regions[a as usize], &regions[b as usize]);
        let boundary = ra.neighbors[&b];
        match criterion {
            Criterion::Contrast => {
                boundary.contrast / boundary.len as f64 - params.contrast * params.contrast
            }
            Criterion::Energy => {
                let merged = energy(&ra.moments.add(&rb.moments), params);
                merged - ra.energy - rb.energy - params.boundary_cost * boundary.len as f64
            }
        }
    };
    let fits = |regions: &[Region], a: u32, b: u32| {
        let merged = regions[a as usize]
            .moments
            .add(&regions[b as usize].moments);
        let sse = merged.flat_sse().min(merged.affine_sse());
        sse <= params.max_fit_error * params.max_fit_error * 3. * merged.n
    };
    let candidate = |regions: &[Region], a: u32, b: u32| Candidate {
        cost: merge_cost(regions, a, b),
        a,
        b,
        version_a: regions[a as usize].version,
        version_b: regions[b as usize].version,
    };

    // Only merges that lower the cost are ever taken, so the others need no place in the queue
    let mut heap = BinaryHeap::new();
    let push = |heap: &mut BinaryHeap<Candidate>, candidate: Candidate| {
        if candidate.cost < 0. {
            heap.push(candidate);
        }
    };
    for (a, region) in regions.iter().enumerate() {
        if a % (MERGES_PER_POLL * 16) == 0 {
            crate::progress::poll().await;
        }
        if parent[a] != a as u32 {
            continue;
        }
        for &b in region.neighbors.keys() {
            if (a as u32) < b {
                push(&mut heap, candidate(regions, a as u32, b));
            }
        }
    }

    // A contrast cost only changes with the shared boundary, so a candidate is current exactly when
    // its cost still matches. Energy costs depend on both regions as a whole.
    let is_current = |regions: &[Region], parent: &[u32], c: &Candidate| {
        let (a, b) = (c.a as usize, c.b as usize);
        parent[a] == c.a
            && parent[b] == c.b
            && match criterion {
                Criterion::Contrast => {
                    regions[a].neighbors.contains_key(&c.b)
                        && merge_cost(regions, c.a, c.b) == c.cost
                }
                Criterion::Energy => {
                    regions[a].version == c.version_a && regions[b].version == c.version_b
                }
            }
    };

    // Merging into a region with many neighbors requeues all of them, so outdated candidates can
    // outnumber current ones by far and are dropped whenever the queue has doubled
    let mut compacted = heap.len();
    let mut popped = 0usize;
    // Per region, the neighbours a contrast merge was refused with because the merged colors
    // did not fit, retried once either side grows
    let mut misfits: HashMap<u32, Vec<u32>> = HashMap::default();
    while let Some(c) = heap.pop() {
        popped += 1;
        if popped.is_multiple_of(MERGES_PER_POLL) {
            crate::progress::poll().await;
        }
        if heap.len() > 2 * compacted + 1024 {
            heap.retain(|c| is_current(regions, parent, c));
            compacted = heap.len();
        }
        if !is_current(regions, parent, &c) {
            continue;
        }
        let (a, b) = (c.a, c.b);
        if matches!(criterion, Criterion::Contrast) && !fits(regions, a, b) {
            misfits.entry(a).or_default().push(b);
            misfits.entry(b).or_default().push(a);
            continue;
        }

        // Merge the region with fewer neighbors into the one with more
        let (keep, gone) =
            if regions[a as usize].neighbors.len() >= regions[b as usize].neighbors.len() {
                (a, b)
            } else {
                (b, a)
            };
        let gone_neighbors = std::mem::take(&mut regions[gone as usize].neighbors);
        for (&n, boundary) in &gone_neighbors {
            let n_neighbors = &mut regions[n as usize].neighbors;
            n_neighbors.remove(&gone);
            if n != keep {
                n_neighbors.entry(keep).or_default().add(boundary);
                regions[keep as usize]
                    .neighbors
                    .entry(n)
                    .or_default()
                    .add(boundary);
            }
        }
        regions[keep as usize].neighbors.remove(&gone);
        let moments = regions[keep as usize]
            .moments
            .add(&regions[gone as usize].moments);
        let kept = &mut regions[keep as usize];
        kept.energy = energy(&moments, params);
        kept.moments = moments;
        kept.version += 1;
        parent[gone as usize] = keep;

        let neighbors: Vec<u32> = match criterion {
            Criterion::Contrast => {
                let mut refused = misfits.remove(&keep).unwrap_or_default();
                refused.extend(misfits.remove(&gone).unwrap_or_default());
                let kept = &regions[keep as usize].neighbors;
                let changed = gone_neighbors.keys().chain(&refused);
                let mut changed: Vec<u32> = changed
                    .copied()
                    .filter(|&n| parent[n as usize] == n && kept.contains_key(&n))
                    .collect();
                changed.sort_unstable();
                changed.dedup();
                changed
            }
            Criterion::Energy => regions[keep as usize].neighbors.keys().copied().collect(),
        };
        for n in neighbors {
            push(&mut heap, candidate(regions, keep, n));
        }
    }
}
