//! Raster primitives in double precision with the arithmetic of numpy, scipy and
//! scikit-image (sums in their order, filters, labels, skeletons): tests at a tolerance on
//! flat art compare exact sums of levels, and a sum taken in another order flips them.

use glam::Vec3;
pub use glam::{DVec2, DVec3};

mod filters;
mod labels;
mod pixels;
mod sums;
mod trees;

pub use filters::*;
pub use labels::*;
pub use pixels::*;
pub use sums::*;
pub use trees::*;

/// The 8-bit quantization plus a level of anti-aliasing, in doubles.
pub const NOISE: f64 = 2.0 / 255.0;
/// What the backgrounds have to show through a shape, k times their spread.
pub const SHOW_THROUGH: f64 = 4.0 * NOISE;

/// The width the constants in pixels were set at (niabot's scenes at 330 px).
pub const REFERENCE_WIDTH: f64 = 330.0;

/// The drawing's length unit in pixels, the width over 330. Constants that measure the
/// drawing (a reach, a smoothing, a least size) scale by it; constants that measure the
/// raster (a step's anti-aliasing, the 8-bit noise) stay in pixels and levels.
pub fn unit_of(width: usize) -> f64 {
    width as f64 / REFERENCE_WIDTH
}

/// A length of `px` pixels at 330 at this image's width, rounded, at least 1.
pub fn length_px(width: usize, px: f64) -> usize {
    ((px * unit_of(width)).round() as usize).max(1)
}

/// An area of `px2` pixels at 330 at this image's width (the unit squared).
pub fn area_px(width: usize, px2: f64) -> f64 {
    px2 * unit_of(width) * unit_of(width)
}

/// The pixels of a mask all of whose pixels within `r` (the square of side 2r + 1, cut by
/// the frame: the frame counts as inside) are in the mask: the erosion by the square,
/// separable.
pub fn deep_in(mask: &[bool], width: usize, height: usize, r: usize) -> Vec<bool> {
    let pass = |src: &[bool], along_x: bool| -> Vec<bool> {
        let (n, m) = if along_x {
            (height, width)
        } else {
            (width, height)
        };
        let at = |line: usize, i: usize| {
            if along_x {
                line * width + i
            } else {
                i * width + line
            }
        };
        let mut out = vec![false; src.len()];
        // the run of mask pixels ending at each i, one buffer for every line (each entry is
        // written before it is read)
        let mut prefix = vec![0usize; m + 1];
        for line in 0..n {
            for i in 0..m {
                prefix[i + 1] = prefix[i] + src[at(line, i)] as usize;
            }
            for i in 0..m {
                let (a, b) = (i.saturating_sub(r), (i + r).min(m - 1));
                out[at(line, i)] = prefix[b + 1] - prefix[a] == b + 1 - a;
            }
        }
        out
    };
    let h = pass(mask, true);
    pass(&h, false)
}

/// An 8-bit image as doubles in 0..1, row-major.
#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub px: Vec<DVec3>,
}

impl Image {
    /// From packed 8-bit RGB, each level over 255 (for the lab).
    #[cfg(any(test, feature = "diagnostics"))]
    pub fn from_rgb8(rgb: &[u8], width: usize, height: usize) -> Self {
        let px = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| {
                DVec3::new(
                    p[0] as f64 / 255.0,
                    p[1] as f64 / 255.0,
                    p[2] as f64 / 255.0,
                )
            })
            .collect();
        Image { width, height, px }
    }

    /// From the crate's target colors, which are 8-bit levels: each channel rounded back
    /// to its level first, so the doubles are exactly the 8-bit ones.
    pub fn from_target(target: &[Vec3], width: usize) -> Self {
        let level = |v: f32| (v as f64 * 255.0).round().clamp(0.0, 255.0) / 255.0;
        let px = target
            .iter()
            .map(|c| DVec3::new(level(c.x), level(c.y), level(c.z)))
            .collect();
        Image {
            width,
            height: target.len() / width.max(1),
            px,
        }
    }

    pub fn len(&self) -> usize {
        self.px.len()
    }

    #[cfg(feature = "diagnostics")]
    pub fn is_empty(&self) -> bool {
        self.px.is_empty()
    }

    pub fn channel(&self, c: usize) -> Vec<f64> {
        self.px.iter().map(|p| p[c]).collect()
    }

    /// The channel means, numpy's `img.mean(axis=2)`.
    pub fn luminance(&self) -> Vec<f64> {
        self.px.iter().map(|p| (p.x + p.y + p.z) / 3.0).collect()
    }

    /// The bilinear color at (x, y), each channel by `map1`.
    pub fn sample(&self, x: f64, y: f64) -> DVec3 {
        let (iy, wy) = axis_weights(y, self.height);
        let (ix, wx) = axis_weights(x, self.width);
        let mut out = DVec3::ZERO;
        for c in 0..3 {
            let mut t = 0.0;
            for a in 0..2 {
                for b in 0..2 {
                    t += self.px[iy[a] * self.width + ix[b]][c] * wy[a] * wx[b];
                }
            }
            out[c] = t;
        }
        out
    }
}
