//! The edge term: squared error pays for an edge only by the pixels it covers, so errors near a
//! target edge (Di Zenzo steps and thin lines, grown by the anti-aliasing reach) weigh more.
//! Weights scale by the image's unit.

use glam::{DVec2, Vec3};

use crate::hierarchy::Hierarchy;
use crate::translucency::lines::suppressed_edges;
use crate::translucency::raster::{
    Conn, Image, NOISE, di_zenzo, digest, dilate, remembered, unit_of,
};
use crate::translucency::vocabulary::{colour_lines_at, thin_lines};

/// The anti-aliasing scale in pixels: the Gaussian the gradient is taken at. An edge is
/// anti-aliased over one pixel.
pub const SIGMA: f64 = 1.0;
/// The anti-aliasing reach in pixels: the pixels either side of an edge whose colour it mixes.
pub const REACH: usize = 1;

/// The edge term: an error on a pixel within the anti-aliasing reach of a target edge counts
/// `1 + lambda * unit` times. At `lambda` 1 the edge band weighs as much again as a band one
/// unit wide of its own error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeTerm {
    pub lambda: f64,
}

/// How much each pixel counts as an edge (1 within the anti-aliasing reach of one, else 0),
/// and the image's unit.
#[derive(Clone, Debug)]
pub struct EdgeField {
    pub width: usize,
    pub height: usize,
    /// how much the edge at a pixel counts, 0 to 1 (0 off the edges)
    pub strength: Vec<f64>,
    pub unit: f64,
    /// How far each pixel may deviate before it counts as error ([`crate::Context::tolerance`]),
    /// for the linework's candidate sums; none for the plain squared error.
    pub tolerance: Option<std::sync::Arc<Vec<f32>>>,
}

type Mask<K> = (K, std::rc::Rc<Vec<bool>>);
/// Remembered masks, newest last, keyed by what they were found from.
type Memo<K> = std::cell::RefCell<Vec<Mask<K>>>;

thread_local! {
    /// The edges, lines and both found last: the plan, the linework, the marks and the economy
    /// each read the same picture's.
    static EDGES: Memo<(u64, usize, usize, (u64, u64))> =
        const { std::cell::RefCell::new(Vec::new()) };
    static CENTRES: Memo<(u64, usize, usize, (u64, u64))> =
        const { std::cell::RefCell::new(Vec::new()) };
    static LINES: Memo<(usize, usize, (u64, u64))> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The suppressed edges of an image at `sigma`.
pub fn edges_of(img: &Image, sigma: f64) -> Vec<bool> {
    let key = (sigma.to_bits(), img.width, img.height, digest(&img.px));
    (*remembered(&EDGES, key, || {
        let (mag, theta) = di_zenzo(img, sigma);
        suppressed_edges(&mag, &theta, img.width, img.height, sigma)
    }))
    .clone()
}

/// The centres of an image's thin lines of any colour at the unit `u`.
pub fn line_centres(img: &Image, u: f64) -> Vec<bool> {
    let key = (u.to_bits(), img.width, img.height, digest(&img.px));
    (*remembered(&CENTRES, key, || {
        let (s, d) = colour_lines_at(img, u);
        thin_lines(&s, &d, img.width, img.height, 3.0 * NOISE, 6.0 * NOISE)
    }))
    .clone()
}

/// The step edges and the thin lines of an image.
pub fn edges_and_lines(img: &Image) -> Vec<bool> {
    let key = (img.width, img.height, digest(&img.px));
    (*remembered(&LINES, key, || edges_and_lines_of(img))).clone()
}

fn edges_and_lines_of(img: &Image) -> Vec<bool> {
    let mut edges = edges_of(img, SIGMA);
    let lines = line_centres(img, unit_of(img.width));
    for (e, l) in edges.iter_mut().zip(lines) {
        *e |= l;
    }
    edges
}

impl EdgeField {
    pub fn of_target(target: &[Vec3], width: usize, height: usize) -> Self {
        let img = Image::from_target(target, width);
        let edges = edges_and_lines(&img);
        let near = dilate(&edges, width, height, Conn::Eight, REACH);
        // An edge's count reaches the pixels its anti-aliasing mixes
        let r = REACH as i32;
        let mut strength = vec![0.; edges.len()];
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let i = y as usize * width + x as usize;
                if !near[i] {
                    continue;
                }
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx >= 0 && yy >= 0 && (xx as usize) < width && (yy as usize) < height {
                            let j = yy as usize * width + xx as usize;
                            strength[i] = f64::max(strength[i], f64::from(u8::from(edges[j])));
                        }
                    }
                }
            }
        }
        Self {
            width,
            height,
            strength,
            unit: unit_of(width),
            tolerance: None,
        }
    }

    /// The squared residual `d` at pixel `i` beyond the pixel's tolerance (all of it without
    /// one), see [`crate::uncertainty::excess`].
    pub fn residual(&self, i: usize, d: glam::DVec3) -> f64 {
        let t = self.tolerance.as_ref().map_or(0., |t| t[i] as f64);
        crate::uncertainty::excess64(d.length_squared(), t)
    }

    /// The fit's weight per pixel.
    pub fn weight(&self, i: usize, lambda: f64) -> f32 {
        (1.0 + lambda * self.unit * self.strength[i]) as f32
    }
}

/// The edge-weighted region sums: per node, the statistics of its pixels on the
/// edge band weighted by `lambda * unit`, so that adding them to the node's own gives the
/// edge-weighted error of any color or gradient fill about them.
pub fn weighted_stats(
    hierarchy: &Hierarchy,
    field: &EdgeField,
    lambda: f64,
) -> Vec<crate::hierarchy::Stats> {
    use crate::hierarchy::Stats;
    let w = lambda * field.unit;
    let mut stats = vec![Stats::default(); hierarchy.nodes.len()];
    let width = hierarchy.width();
    for (i, &s) in field.strength.iter().enumerate() {
        if s <= 0. {
            continue;
        }
        let leaf = hierarchy.leaf(i) as usize;
        let p = DVec2::new((i % width) as f64 + 0.5, (i / width) as f64 + 0.5);
        stats[leaf] = stats[leaf].add(&Stats::pixel_at(hierarchy.colors[i], p).scaled(w * s));
    }
    // Children come before their parents in the node order
    for node in 0..hierarchy.nodes.len() {
        if let Some([a, b]) = hierarchy.nodes[node].children {
            stats[node] = stats[a as usize].add(&stats[b as usize]);
        }
    }
    stats
        .iter()
        .zip(&hierarchy.nodes)
        .map(|(e, n)| e.add(&n.stats))
        .collect()
}
