//! A path's coverage blurred by a Gaussian of standard deviation σ, as SVG's `feGaussianBlur`
//! draws it: the coverage rasterized past the window's edges by the kernel's reach, blurred
//! separably over the box it can reach, with the gradient by the points and by σ.

use glam::Vec2;

use crate::render::raster::{Winding, fill_coverage, rasterize, rasterize_backward};
use crate::render::scene::{FillRule, FlatPoint};

/// How many standard deviations the kernel reaches.
pub const REACH: f32 = 3.;

/// The pixels a blur of `sigma` reaches beyond the outline.
pub fn reach(sigma: f32) -> usize {
    (REACH * sigma).ceil() as usize
}

/// The kernel of a blur of `sigma`: a Gaussian integrated over each pixel, normalized, with
/// its derivative by `sigma`; `2 reach + 1` taps.
pub fn kernel(sigma: f32) -> (Vec<f32>, Vec<f32>) {
    let r = reach(sigma) as i32;
    let s = sigma as f64;
    // Φ(a / σ) and its derivative by σ
    let cdf = |a: f64| 0.5 * (1. + erf(a / (s * std::f64::consts::SQRT_2)));
    let d_cdf = |a: f64| {
        let z = a / s;
        -(-0.5 * z * z).exp() / (2. * std::f64::consts::PI).sqrt() * z / s
    };
    let raw: Vec<(f64, f64)> = (-r..=r)
        .map(|i| {
            let (lo, hi) = (i as f64 - 0.5, i as f64 + 0.5);
            (cdf(hi) - cdf(lo), d_cdf(hi) - d_cdf(lo))
        })
        .collect();
    let (sum, d_sum) = raw.iter().fold((0., 0.), |(s, d), (k, dk)| (s + k, d + dk));
    raw.iter()
        .map(|&(k, dk)| {
            (
                (k / sum) as f32,
                ((dk * sum - k * d_sum) / (sum * sum)) as f32,
            )
        })
        .unzip()
}

/// The error function, to within 1.2e-7 (Numerical Recipes' erfc by Chebyshev fit).
fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1. / (1. + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let erfc = t * poly.exp();
    if x >= 0. { 1. - erfc } else { erfc - 1. }
}

/// A blurred coverage over the box `x0, y0, w, h` of a `width` x `height` window, and what its
/// backward pass needs.
pub struct Blurred {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    pub values: Vec<f32>,
    pub(crate) polys: Vec<Vec<FlatPoint>>,
    winding: Winding,
    rule: FillRule,
    /// The horizontal pass over the coverage's rows, on the box's columns.
    rows: Vec<f32>,
    kernel: Vec<f32>,
    d_kernel: Vec<f32>,
    r: usize,
}

impl Blurred {
    /// `polys` filled under `rule` and blurred by `sigma`, in a `width` x `height` window.
    pub fn new(
        polys: Vec<Vec<FlatPoint>>,
        rule: FillRule,
        sigma: f32,
        width: usize,
        height: usize,
    ) -> Self {
        let r = reach(sigma);
        // Rasterized in the window grown by the reach, so coverage beyond its edges blurs in
        let shifted: Vec<Vec<FlatPoint>> = polys
            .iter()
            .map(|p| {
                p.iter()
                    .map(|q| FlatPoint {
                        pos: q.pos + Vec2::splat(r as f32),
                        ..*q
                    })
                    .collect()
            })
            .collect();
        let winding = rasterize(&shifted, (width + 2 * r) as u32, (height + 2 * r) as u32);
        let (kernel, d_kernel) = kernel(sigma);
        let mut out = Self {
            x0: 0,
            y0: 0,
            w: 0,
            h: 0,
            values: Vec::new(),
            polys: shifted,
            winding,
            rule,
            rows: Vec::new(),
            kernel,
            d_kernel,
            r,
        };
        let c = &out.winding;
        if c.w == 0 {
            return out;
        }
        // The box in the grown frame: the coverage's box grown by the reach, within the window
        let (cx0, cy0) = (c.x0 as usize, c.y0 as usize);
        let bx0 = cx0.saturating_sub(r).max(r);
        let by0 = cy0.saturating_sub(r).max(r);
        let bx1 = (cx0 + c.w + r).min(width + r);
        let by1 = (cy0 + c.h + r).min(height + r);
        if bx1 <= bx0 || by1 <= by0 {
            return out;
        }
        let (w, h) = (bx1 - bx0, by1 - by0);
        let coverage: Vec<f32> = c.values.iter().map(|&v| fill_coverage(v, rule).0).collect();
        let rows = out.pass_rows(&coverage, &out.kernel, bx0, w);
        let values = out.pass_columns(&rows, &out.kernel, by0, w, h);
        out.x0 = bx0 - r;
        out.y0 = by0 - r;
        out.w = w;
        out.h = h;
        out.values = values;
        out.rows = rows;
        out
    }

    /// The horizontal pass of `kernel` over the coverage's rows, on the `w` columns from `bx0`.
    fn pass_rows(&self, coverage: &[f32], kernel: &[f32], bx0: usize, w: usize) -> Vec<f32> {
        let c = &self.winding;
        let (cx0, r) = (c.x0 as usize, self.r);
        let mut rows = vec![0.; c.h * w];
        for y in 0..c.h {
            let src = &coverage[y * c.w..(y + 1) * c.w];
            let dst = &mut rows[y * w..(y + 1) * w];
            for (j, &v) in src.iter().enumerate() {
                if v == 0. {
                    continue;
                }
                // Coverage at grown column cx0 + j reaches columns cx0 + j - r ..= cx0 + j + r
                let x = cx0 + j;
                let lo = x.saturating_sub(r).max(bx0);
                let hi = (x + r + 1).min(bx0 + w);
                for t in lo..hi {
                    dst[t - bx0] += kernel[t + r - x] * v;
                }
            }
        }
        rows
    }

    /// The vertical pass of `kernel` over `rows`, on the box's `h` rows from `by0`.
    fn pass_columns(
        &self,
        rows: &[f32],
        kernel: &[f32],
        by0: usize,
        w: usize,
        h: usize,
    ) -> Vec<f32> {
        let c = &self.winding;
        let (cy0, r) = (c.y0 as usize, self.r);
        let mut out = vec![0.; w * h];
        for j in 0..c.h {
            let y = cy0 + j;
            let lo = y.saturating_sub(r).max(by0);
            let hi = (y + r + 1).min(by0 + h);
            let src = &rows[j * w..(j + 1) * w];
            for t in lo..hi {
                let k = kernel[t + r - y];
                let dst = &mut out[(t - by0) * w..(t - by0 + 1) * w];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += k * s;
                }
            }
        }
        out
    }

    /// The blurred coverage at pixel `(x, y)` of the window, zero outside the box.
    pub fn at(&self, x: usize, y: usize) -> f32 {
        if x < self.x0 || y < self.y0 || x >= self.x0 + self.w || y >= self.y0 + self.h {
            return 0.;
        }
        self.values[(y - self.y0) * self.w + x - self.x0]
    }

    /// Backpropagates `d_values` (laid out like `values`) to the flattened points (in the
    /// unshifted frame, as given) and to σ.
    pub fn backward(&self, d_values: &[f32]) -> (Vec<Vec<Vec2>>, f32) {
        let c = &self.winding;
        if self.w == 0 {
            let d = self
                .polys
                .iter()
                .map(|p| vec![Vec2::ZERO; p.len()])
                .collect();
            return (d, 0.);
        }
        let (r, w, h) = (self.r, self.w, self.h);
        let (bx0, by0) = (self.x0 + r, self.y0 + r);
        let (cx0, cy0) = (c.x0 as usize, c.y0 as usize);
        // Through the vertical pass to the rows, and σ's share there
        let mut d_rows = vec![0.; c.h * w];
        let mut d_sigma = 0f64;
        for j in 0..c.h {
            let y = cy0 + j;
            let lo = y.saturating_sub(r).max(by0);
            let hi = (y + r + 1).min(by0 + h);
            let src = &self.rows[j * w..(j + 1) * w];
            let dst = &mut d_rows[j * w..(j + 1) * w];
            for t in lo..hi {
                let (k, dk) = (self.kernel[t + r - y], self.d_kernel[t + r - y]);
                let d = &d_values[(t - by0) * w..(t - by0 + 1) * w];
                let mut dot = 0f32;
                for ((acc, &dv), &s) in dst.iter_mut().zip(d).zip(src) {
                    *acc += k * dv;
                    dot += dv * s;
                }
                d_sigma += (dk * dot) as f64;
            }
        }
        // Through the horizontal pass to the coverage, and σ's share there
        let mut d_winding = vec![0.; c.values.len()];
        for y in 0..c.h {
            let d = &d_rows[y * w..(y + 1) * w];
            for j in 0..c.w {
                let x = cx0 + j;
                let lo = x.saturating_sub(r).max(bx0);
                let hi = (x + r + 1).min(bx0 + w);
                if hi <= lo {
                    continue;
                }
                let (mut dc, mut ds) = (0f32, 0f32);
                for t in lo..hi {
                    dc += self.kernel[t + r - x] * d[t - bx0];
                    ds += self.d_kernel[t + r - x] * d[t - bx0];
                }
                let (alpha, slope) = fill_coverage(c.values[y * c.w + j], self.rule);
                d_winding[y * c.w + j] = dc * slope;
                d_sigma += (ds * alpha) as f64;
            }
        }
        (
            rasterize_backward(&self.polys, c, &d_winding),
            d_sigma as f32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::scene::{Path, Subpath, flatten};

    fn square(sigma: f32) -> Path {
        let corners = [(6., 5.), (19., 7.), (17., 18.), (4., 16.)].map(|(x, y)| Vec2::new(x, y));
        Path {
            subpaths: vec![Subpath::polygon(&corners)],
            color: [1.; 4],
            gradient: None,
            fill_rule: FillRule::EvenOdd,
            blur: sigma,
        }
    }

    /// A blur keeps the coverage's sum where it all lies within the window, and its
    /// derivatives by a point and by σ match finite differences.
    #[test]
    fn blur_keeps_area_and_its_gradient_matches() {
        let (w, h) = (24, 22);
        let weight: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 11) as f32 / 11.).collect();
        let energy = |path: &Path| -> (f32, Blurred) {
            let b = Blurred::new(flatten(path), FillRule::EvenOdd, path.blur, w, h);
            let mut e = 0.;
            for y in 0..h {
                for x in 0..w {
                    e += weight[y * w + x] * b.at(x, y);
                }
            }
            (e, b)
        };
        let path = square(1.3);
        let (_, b) = energy(&path);
        let hard = Blurred::new(flatten(&path), FillRule::EvenOdd, 0.01, w, h);
        let (s0, s1) = (
            hard.values.iter().sum::<f32>(),
            b.values.iter().sum::<f32>(),
        );
        assert!((s0 - s1).abs() < 1e-3 * s0, "{s0} {s1}");
        let d_values: Vec<f32> = (0..b.h)
            .flat_map(|y| (0..b.w).map(move |x| (x, y)))
            .map(|(x, y)| weight[(b.y0 + y) * w + b.x0 + x])
            .collect();
        let (d_flat, d_sigma) = b.backward(&d_values);
        let eps = 1e-2;
        let (up, _) = energy(&square(1.3 + eps));
        let (down, _) = energy(&square(1.3 - eps));
        let fd = (up - down) / (2. * eps);
        assert!(
            (fd - d_sigma).abs() < 2e-2 * (1. + fd.abs()),
            "{fd} {d_sigma}"
        );
        // The first corner's x through the flattened points
        let d_corner: f32 = flatten(&path)[0]
            .iter()
            .zip(&d_flat[0])
            .map(|(p, d)| {
                p.src
                    .iter()
                    .zip(&p.weights)
                    .filter(|(s, _)| **s == 0)
                    .map(|(_, wt)| wt * d.x)
                    .sum::<f32>()
            })
            .sum();
        let moved = |dx: f32| {
            let mut p = path.clone();
            p.subpaths[0].points[0].x += dx;
            energy(&p).0
        };
        let fd = (moved(eps) - moved(-eps)) / (2. * eps);
        assert!(
            (fd - d_corner).abs() < 2e-2 * (1. + fd.abs()),
            "{fd} {d_corner}"
        );
    }
}
