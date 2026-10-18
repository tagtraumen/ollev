//! The differentiable renderer: paths rasterized with exact coverage, and the gradient of the
//! squared error with respect to their points and colours, for fitting a scene to the picture.

pub mod blur;
pub mod coverage_cost;
pub mod distance;
pub mod dual;
pub mod optim;
pub mod raster;
pub mod scene;

use glam::{Vec2, Vec3};

pub use optim::{Adam, ParamKind};

use raster::{Winding, fill_coverage, rasterize, rasterize_backward};
pub use scene::{FillRule, Gradient, MAX_STOPS, Path, Scene, Stops, Subpath, opaque};
use scene::flatten;

#[cfg(test)]
mod backward;

/// The scene's paths composited bottom to top over `pixels` (row-major RGB in `0..=1`), such
/// as the already composited content below the paths being fitted.
pub fn render_over(scene: &Scene, mut pixels: Vec<Vec3>) -> Vec<Vec3> {
    let (width, height) = (scene.width, scene.height);
    assert_eq!(pixels.len(), (width * height) as usize);

    for path in &scene.paths {
        // A blurred path composites its blurred coverage; it has no backward pass
        if path.blur > 0. {
            let (w, h) = (width as usize, height as usize);
            let b = blur::Blurred::new(flatten(path), path.fill_rule, path.blur, w, h);
            for y in b.y0..b.y0 + b.h {
                for x in b.x0..b.x0 + b.w {
                    let alpha = b.values[(y - b.y0) * b.w + x - b.x0] * path.color[3];
                    if alpha != 0. {
                        let pixel = &mut pixels[y * w + x];
                        let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                        let (color, ramp, _) = path.paint(p);
                        let alpha = alpha * ramp;
                        *pixel = color * alpha + *pixel * (1. - alpha);
                    }
                }
            }
            continue;
        }
        let polys = flatten(path);
        let winding = rasterize(&polys, width, height);
        let a = path.color[3];
        for (&w, (index, p)) in winding.values.iter().zip(winding_pixels(&winding, width)) {
            let pixel = &mut pixels[index];
            let alpha = fill_coverage(w, path.fill_rule).0 * a;
            if alpha != 0. {
                let (color, ramp, _) = path.paint(p);
                let alpha = alpha * ramp;
                *pixel = color * alpha + *pixel * (1. - alpha);
            }
        }
    }
    pixels
}

/// Backpropagates `d_coverage` (d loss / d coverage over the path's box in a `width` x `height`
/// frame: its winding box, or its blurred box where it is blurred) to the points of each
/// subpath and to the blur's σ.
pub fn coverage_backward(
    path: &Path,
    width: u32,
    height: u32,
    d_coverage: &[f32],
) -> (Vec<Vec<Vec2>>, f32) {
    let polys = flatten(path);
    let (d_flat, d_sigma) = if path.blur > 0. {
        let b = blur::Blurred::new(
            polys.clone(),
            path.fill_rule,
            path.blur,
            width as usize,
            height as usize,
        );
        b.backward(d_coverage)
    } else {
        let winding = rasterize(&polys, width, height);
        let d_winding: Vec<f32> = winding
            .values
            .iter()
            .zip(d_coverage)
            .map(|(&w, &d)| d * fill_coverage(w, path.fill_rule).1)
            .collect();
        (rasterize_backward(&polys, &winding, &d_winding), 0.)
    };
    let mut d_points: Vec<Vec<Vec2>> = path
        .subpaths
        .iter()
        .map(|s| vec![Vec2::ZERO; s.points.len()])
        .collect();
    for ((poly, d_poly), d_subpath) in polys.iter().zip(&d_flat).zip(&mut d_points) {
        for (point, d_point) in poly.iter().zip(d_poly) {
            for (&src, &weight) in point.src.iter().zip(&point.weights) {
                d_subpath[src as usize] += weight * *d_point;
            }
        }
    }
    (d_points, d_sigma)
}

/// The image index and center of each pixel in the winding's box, in the order of its values.
pub(crate) fn winding_pixels(winding: &Winding, width: u32) -> impl Iterator<Item = (usize, Vec2)> {
    let (x0, y0) = (winding.x0 as usize, winding.y0 as usize);
    (y0..y0 + winding.h).flat_map(move |y| {
        (x0..x0 + winding.w).map(move |x| {
            let center = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            (y * width as usize + x, center)
        })
    })
}

impl Subpath {
    /// A closed polygon made of straight cubic curves.
    #[cfg(test)]
    pub fn polygon(corners: &[Vec2]) -> Self {
        let points = corners
            .iter()
            .zip(corners.iter().cycle().skip(1))
            .flat_map(|(&a, &b)| [a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.)])
            .collect();
        Self { points }
    }
}

#[cfg(test)]
mod tests;
