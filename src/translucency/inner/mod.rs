//! A layer's inner layers: what a family's relation leaves unexplained in its region, read
//! as a white or black of per-pixel opacity or a tint over the layer's image. An inner layer
//! moves an image only toward its own color, so it pins k where one material would not.

use crate::HashMap;

use super::materials::{
    MATCH_TOL, Matcher, Palette, VEIL_MAX, VOTE_GRID, in_gamut_c, min_material_px, region_palettes,
    vote_cells,
};
use super::raster::{
    ChebGrid, ChebTree, Conn, DVec3, Image, NOISE, SHOW_THROUGH, arange, cheb, label,
};

mod fit;
mod open;
mod scan;
mod white;

pub use fit::*;
pub use open::*;
pub use scan::*;
pub use white::*;

/// What explains a pixel of the region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Class {
    /// nothing (or a rim, or a color shared with an open material)
    None,
    /// the layer alone over an open material
    Alone,
    /// a white above the layer's image at the pixel's opacity
    White,
    /// a black above the layer's image at the pixel's opacity
    Black,
    /// the tint of this index above the layer's image
    Tint(usize),
}

/// Per pixel what explains it: the class, the open material, the inner layer's opacity (0
/// for the layer alone), the residual in the image.
#[derive(Clone, Copy, Debug)]
pub struct PixelFit {
    pub class: Class,
    pub mat: usize,
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))] // read by the lab
    pub t: f64,
    pub d: f64,
}

const NO_FIT: PixelFit = PixelFit {
    class: Class::None,
    mat: usize::MAX,
    t: 0.0,
    d: f64::INFINITY,
};

/// A tint above the layer: `o = k d + m` over the layer's image d.
#[derive(Clone, Copy, Debug)]
pub struct Tint {
    pub k: f64,
    pub m: DVec3,
}

#[cfg(feature = "diagnostics")]
impl Tint {
    pub fn c(&self) -> DVec3 {
        self.m / (1.0 - self.k)
    }
}

/// The region's pixels and the outside materials, fixed for every relation.
pub struct InnerInputs {
    pub width: usize,
    pub height: usize,
    /// the region
    pub region: Vec<bool>,
    /// the image's colors
    pub px: Vec<DVec3>,
    /// the pixels scored: the region off the rims
    pub inside: Vec<bool>,
    /// the outside materials' colors, their material and pixels, and a tree of them
    pub b: Vec<DVec3>,
    pub bmat: Vec<usize>,
    pub bw: Vec<f64>,
    pub btree: ChebTree,
    /// per material its mean and pixels
    pub bmean: Vec<DVec3>,
    pub bpx: Vec<f64>,
    /// the inside's distinct colors, their pixels, per pixel its distinct color, and per
    /// color the outside colors within the show-through of it
    pub colors: Vec<DVec3>,
    pub weights: Vec<f64>,
    pub index: Vec<u32>,
    pub nearby: Vec<Vec<u32>>,
    pub inside_palette: Palette,
    /// the inside palette with nothing dropped (the drop takes the layer's own colors where
    /// an outside material is the layer elsewhere)
    pub inside_all: Palette,
    pub outside_palette: Palette,
    /// the region's pixels on rims
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))] // read by the lab
    pub rim_px: usize,
}

fn code_of(c: DVec3) -> u32 {
    let q = |v: f64| (v * 255.0).round_ties_even() as u32;
    (q(c.x) * 256 + q(c.y)) * 256 + q(c.z)
}

impl InnerInputs {
    /// The pin's palettes (shared colors dropped from the inside palette, which only
    /// proposes m) and the region's pixels off the rims. `layer`, when given, are pixels
    /// taken out of the outside.
    pub fn new(img: &Image, region: &[bool], rim: &[bool], layer: Option<&[bool]>) -> Self {
        let open: Option<Vec<bool>> = layer.map(|l| l.iter().map(|v| !v).collect());
        let (pi, po) = region_palettes(img, region, rim, true, open.as_deref());
        let inside: Vec<bool> = (0..img.len()).map(|i| region[i] && !rim[i]).collect();
        let mut mat_of = vec![usize::MAX; po.clusters.len()];
        for (r, ci) in po.materials().enumerate() {
            mat_of[ci] = r;
        }
        let nm = po.materials().len();
        let (mut b, mut bmat, mut bw) = (Vec::new(), Vec::new(), Vec::new());
        let mut bpx = vec![0.0f64; nm];
        let mut bcode: HashMap<u32, u32> = HashMap::default();
        for ((c, w), &l) in po.colors.iter().zip(&po.weights).zip(&po.label) {
            if mat_of[l] != usize::MAX {
                bcode.insert(code_of(*c), b.len() as u32);
                b.push(*c);
                bmat.push(mat_of[l]);
                bw.push(*w);
                bpx[mat_of[l]] += w;
            }
        }
        let bmean = po.materials().map(|ci| po.clusters[ci].mean).collect();
        let mut codes: HashMap<u32, u32> = HashMap::default();
        let (mut colors, mut weights) = (Vec::new(), Vec::new());
        let mut index = vec![u32::MAX; img.len()];
        for i in 0..img.len() {
            if !inside[i] {
                continue;
            }
            let e = *codes.entry(code_of(img.px[i])).or_insert_with(|| {
                colors.push(img.px[i]);
                weights.push(0.0);
                (colors.len() - 1) as u32
            });
            weights[e as usize] += 1.0;
            index[i] = e;
        }
        // the outside colors in the show-through's cube about each inside color, in code
        // order: per red and green of the cube, the run of blues found by a binary search in
        // the sorted codes (red, green, blue in that order)
        let r = (SHOW_THROUGH * 255.0).round() as i32;
        let mut sorted: Vec<(u32, u32)> = bcode.iter().map(|(&c, &j)| (c, j)).collect();
        sorted.sort_unstable();
        let nearby = colors
            .iter()
            .map(|&o| {
                let q = |v: f64| (v * 255.0).round_ties_even() as i32;
                let (r0, g0, b0) = (q(o.x), q(o.y), q(o.z));
                let (blo, bhi) = ((b0 - r).max(0), (b0 + r).min(255));
                let mut out = Vec::new();
                for rr in (r0 - r).max(0)..=(r0 + r).min(255) {
                    for gg in (g0 - r).max(0)..=(g0 + r).min(255) {
                        let base = ((rr * 256 + gg) * 256) as u32;
                        let (lo, hi) = (base + blo as u32, base + bhi as u32);
                        let start = sorted.partition_point(|e| e.0 < lo);
                        out.extend(
                            sorted[start..]
                                .iter()
                                .take_while(|e| e.0 <= hi)
                                .map(|e| e.1),
                        );
                    }
                }
                out
            })
            .collect();
        InnerInputs {
            width: img.width,
            height: img.height,
            region: region.to_vec(),
            px: img.px.clone(),
            inside,
            btree: ChebTree::new(&b),
            b,
            bmat,
            bw,
            bmean,
            bpx,
            colors,
            weights,
            index,
            nearby,
            rim_px: (0..img.len()).filter(|&i| region[i] && rim[i]).count(),
            inside_palette: pi,
            inside_all: Palette::new(img, region, rim, min_material_px(img)),
            outside_palette: po,
        }
    }
}
