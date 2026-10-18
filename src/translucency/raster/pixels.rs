//! Thin-line pixels and the Chebyshev distance.

use super::{DVec3, Image};

/// Thin dark ridges: per pixel the most it is darker (the channels' mean) than both pixels
/// r away along one of four directions, and the direction (dy, dx) across the line (outlines
/// a pixel wide).
pub struct LinePixels {
    pub strength: Vec<f64>,
    pub normal: Vec<(i32, i32)>,
}

/// [`LinePixels`] with the reaches `radii` of the ridge test.
pub fn line_pixels_at(img: &Image, radii: &[isize]) -> LinePixels {
    let (w, h) = (img.width as isize, img.height as isize);
    let l = img.luminance();
    let n = img.len();
    let mut best = vec![0.0f64; n];
    let mut normal = vec![(0i32, 0i32); n];
    let clamp = |y: isize, x: isize| (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize;
    for &r in radii {
        for (dy, dx) in [(0isize, 1isize), (1, 0), (1, 1), (1, -1)] {
            for y in 0..h {
                for x in 0..w {
                    let p = (y * w + x) as usize;
                    let a = clamp(y + dy * r, x + dx * r);
                    let b = clamp(y - dy * r, x - dx * r);
                    let s = (l[a] - l[p]).min(l[b] - l[p]);
                    if s > best[p] {
                        best[p] = s;
                        normal[p] = (dy as i32, dx as i32);
                    }
                }
            }
        }
    }
    LinePixels {
        strength: best,
        normal,
    }
}

/// The largest channel difference.
pub fn cheb(a: DVec3, b: DVec3) -> f64 {
    (a - b).abs().max_element()
}
