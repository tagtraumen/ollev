//! The first stage: the picture segmented into regions and their boundaries fitted with cubics
//! (with the primitives, circles, ellipses and polygons, that a boundary may be).

pub mod angle;
mod antialias;
pub mod beautify;
pub mod fit;
pub mod jpeg;
pub mod planar;
pub mod primitives;
pub mod segment;

use glam::DVec3;

pub use antialias::AntialiasParams;
pub use fit::{Cubic, FitParams};
pub use segment::SegmentParams;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Bilateral filter strength in color distance, or `None` to skip denoising.
    pub denoise: Option<f64>,
    /// Sharpening of anti-aliased edges before segmentation, or `None` to segment the blends too.
    pub antialias: Option<AntialiasParams>,
    pub segment: SegmentParams,
    /// The detail level these parameters were scaled for, see [`Params::with_detail`].
    pub detail: f64,
}

/// A default detail level from the count of pixels with a clear color change to a neighbor, per
/// pixel of image width (which grows with the boundaries crossing the image, not with its size).
/// Flat art stays at 1, busy illustrations and photos go up to 3.
pub fn auto_detail(rgb: &[u8], width: usize, height: usize) -> f64 {
    let color = |i: usize| {
        DVec3::new(
            rgb[3 * i] as f64,
            rgb[3 * i + 1] as f64,
            rgb[3 * i + 2] as f64,
        ) / 255.
    };
    let mut edges = 0usize;
    for y in 0..height.saturating_sub(1) {
        for x in 0..width.saturating_sub(1) {
            let i = y * width + x;
            if color(i)
                .distance(color(i + 1))
                .max(color(i).distance(color(i + width)))
                > 0.1
            {
                edges += 1;
            }
        }
    }
    let per_width = edges as f64 / ((width * height) as f64).sqrt();
    (1. + 0.5 * (per_width / 16.).max(1.).log2()).clamp(1., 3.)
}

impl Params {
    /// Scales every threshold that controls how much detail survives. `1` is the default, larger
    /// values keep more regions and split more curves during refinement. The fit tolerance stays
    /// fixed, since tighter fits follow the pixel staircase and trap refinement in wobbly curves.
    pub fn with_detail(detail: f64) -> Self {
        let mut params = Self::default();
        params.segment.contrast /= detail;
        params.segment.max_fit_error /= detail;
        params.segment.boundary_cost /= detail * detail;
        params.detail = detail;
        params
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            denoise: Some(0.08),
            antialias: Some(AntialiasParams {
                edge_contrast: 0.15,
                residual: 0.05,
            }),
            segment: SegmentParams {
                boundary_cost: 0.5,
                gradient_cost: 2.,
                contrast: 0.04,
                max_fit_error: 0.06,
            },
            detail: 1.,
        }
    }
}

fn rgb_colors(rgb: &[u8]) -> Vec<DVec3> {
    rgb.as_chunks::<3>()
        .0
        .iter()
        .map(|c| DVec3::new(c[0] as f64, c[1] as f64, c[2] as f64) / 255.)
        .collect()
}

/// The picture's colors with anti-aliased pixels unmixed to the side they mostly belong to, and
/// no denoise: the translucency stage reads differences of a few levels, which the
/// segmentation's bilateral filter would smooth away.
pub async fn unmixed_colors(
    rgb: &[u8],
    width: usize,
    height: usize,
    antialias: &AntialiasParams,
) -> Vec<DVec3> {
    antialias::unmix(&rgb_colors(rgb), width, height, antialias).await
}

/// The colors segmentation works on: denoised and with anti-aliased pixels unmixed back to the
/// colors on either side, as `params` asks.
pub async fn segmentation_colors(
    rgb: &[u8],
    width: usize,
    height: usize,
    params: &Params,
) -> Vec<DVec3> {
    let mut colors = rgb_colors(rgb);
    if let Some(range) = params.denoise {
        colors = bilateral(&colors, width, height, range).await;
    }
    if let Some(antialias) = &params.antialias {
        colors = antialias::unmix(&colors, width, height, antialias).await;
    }
    colors
}

/// Rows of a pass over the picture between two checkpoints.
const ROWS: isize = 32;

pub fn signed_area(cubics: &[Cubic]) -> f32 {
    cubics
        .iter()
        .flat_map(|c| [(c[0], c[1]), (c[1], c[2]), (c[2], c[3])])
        .map(|(a, b)| a.perp_dot(b))
        .sum::<f32>()
        * 0.5
}

/// Edge-preserving smoothing, which removes compression noise and dithering before segmentation
/// so they do not turn into speckle regions.
async fn bilateral(colors: &[DVec3], width: usize, height: usize, range: f64) -> Vec<DVec3> {
    const RADIUS: isize = 2;
    const SPATIAL: f64 = 1.5;
    let (w, h) = (width as isize, height as isize);
    let mut out = vec![DVec3::ZERO; colors.len()];
    // a neighbour of the centre's own colour weighs its spatial weight alone, the same bits
    let side = (2 * RADIUS + 1) as usize;
    let spatial_of = |dx: isize, dy: isize| (dx * dx + dy * dy) as f64 / (2. * SPATIAL * SPATIAL);
    let alone: Vec<f64> = (0..side * side)
        .map(|k| {
            let (dx, dy) = ((k % side) as isize - RADIUS, (k / side) as isize - RADIUS);
            (-spatial_of(dx, dy) - 0.).exp()
        })
        .collect();
    for y in 0..h {
        if y % ROWS == 0 {
            crate::progress::poll().await;
        }
        for x in 0..w {
            let center = colors[(y * w + x) as usize];
            let (mut sum, mut weight) = (DVec3::ZERO, 0.);
            for dy in -RADIUS..=RADIUS {
                for dx in -RADIUS..=RADIUS {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let c = colors[(ny * w + nx) as usize];
                    let tonal = c.distance_squared(center) / (2. * range * range);
                    let wgt = if tonal == 0. {
                        alone[(dy + RADIUS) as usize * side + (dx + RADIUS) as usize]
                    } else {
                        (-spatial_of(dx, dy) - tonal).exp()
                    };
                    sum += c * wgt;
                    weight += wgt;
                }
            }
            out[(y * w + x) as usize] = sum / weight;
        }
    }
    out
}
