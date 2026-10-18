//! The whole scene's backward pass, the reference the tests check the renderer's coverage
//! gradients against (see [`super::coverage_cost`]).

use glam::{Vec2, Vec3};

use super::raster::{Winding, fill_coverage, rasterize, rasterize_backward};
use super::scene::{FlatPoint, Gradient, flatten};
use super::{Scene, render_over, winding_pixels};

struct PathTape {
    polys: Vec<Vec<FlatPoint>>,
    winding: Winding,
    /// The composited color under the path for each pixel of its winding box.
    below: Vec<Vec3>,
}

/// Intermediate values from [`render_taped`] needed by [`backward`].
pub struct Tape {
    paths: Vec<PathTape>,
}

/// Gradient of a scalar loss with respect to every parameter of a [`Scene`], laid out like it.
#[derive(Clone, Debug)]
pub struct SceneGrad {
    pub paths: Vec<PathGrad>,
}

#[derive(Clone, Debug)]
pub struct PathGrad {
    pub subpaths: Vec<Vec<Vec2>>,
    pub color: [f32; 4],
    pub gradient: Option<Gradient>,
}

impl SceneGrad {
    pub fn zeros(scene: &Scene) -> Self {
        let paths = scene
            .paths
            .iter()
            .map(|path| PathGrad {
                subpaths: path.zero_points(),
                color: [0.; 4],
                gradient: path.gradient.map(|g| g.zeroed()),
            })
            .collect();
        Self { paths }
    }

    /// Visits the gradient: per path its points, then its colour, then its gradient's.
    pub fn visit(&self, mut f: impl FnMut(f32)) {
        for path in &self.paths {
            for subpath in &path.subpaths {
                for point in subpath {
                    f(point.x);
                    f(point.y);
                }
            }
            path.color.iter().copied().for_each(&mut f);
            if let Some(gradient) = &path.gradient {
                let mut gradient = *gradient;
                gradient.visit_geometry(|v| f(*v));
                gradient.to().into_iter().for_each(&mut f);
            }
        }
    }
}

/// The scene over a uniform `background`, with its tape.
pub fn render(scene: &Scene, background: [f32; 3]) -> (Vec<Vec3>, Tape) {
    let pixels = vec![Vec3::from(background); (scene.width * scene.height) as usize];
    render_taped(scene, pixels)
}

/// [`render_over`] with the tape: each path's flattened outline, winding and what lies below
/// it (the paths beneath composited by [`render_over`] itself).
pub fn render_taped(scene: &Scene, pixels: Vec<Vec3>) -> (Vec<Vec3>, Tape) {
    let mut paths = Vec::with_capacity(scene.paths.len());
    for k in 0..scene.paths.len() {
        let beneath = Scene {
            paths: scene.paths[..k].to_vec(),
            ..scene.clone()
        };
        let under = render_over(&beneath, pixels.clone());
        let path = &scene.paths[k];
        if path.blur > 0. {
            paths.push(PathTape {
                polys: Vec::new(),
                winding: Winding::default(),
                below: Vec::new(),
            });
            continue;
        }
        let polys = flatten(path);
        let winding = rasterize(&polys, scene.width, scene.height);
        let below = winding_pixels(&winding, scene.width)
            .map(|(index, _)| under[index])
            .collect();
        paths.push(PathTape {
            polys,
            winding,
            below,
        });
    }
    (render_over(scene, pixels), Tape { paths })
}

/// Backpropagates `d_image` (d loss / d pixel) through the render to all scene parameters.
pub fn backward(scene: &Scene, tape: &Tape, d_image: &[Vec3]) -> SceneGrad {
    let mut grad = SceneGrad::zeros(scene);
    // d loss / d (image after compositing the current path), walked from the top path down
    let mut d_pixels = d_image.to_vec();

    for ((path, path_tape), path_grad) in scene
        .paths
        .iter()
        .zip(&tape.paths)
        .zip(&mut grad.paths)
        .rev()
    {
        assert!(path.blur == 0., "a blurred path has no backward pass");
        let PathTape {
            polys,
            winding,
            below,
        } = path_tape;
        let a = path.color[3];
        let mut d_from = Vec3::ZERO;
        let mut d_to = Vec3::ZERO;
        let mut d_geometry = path.gradient.map(|g| g.zeroed());
        let mut d_alpha_channel = 0.;
        let mut d_winding = vec![0.; winding.values.len()];

        let mut d_to_alpha = 0.;
        let pixels = winding_pixels(winding, scene.width);
        let opaque = path.gradient.is_none_or(|g| g.is_opaque());
        for (i, (&w, (index, p))) in winding.values.iter().zip(pixels).enumerate() {
            let (color, ramp_alpha, t) = path.paint(p);
            let d_out = &mut d_pixels[index];
            if let (false, Some(gradient)) = (opaque, &path.gradient) {
                // out = below + coverage a (P - A below), P and A the ramp's premultiplied
                // colour and alpha, each linear in the knots' (alpha, alpha colour)
                let (coverage, d_coverage_d_winding) = fill_coverage(w, path.fill_rule);
                let alpha = coverage * a * ramp_alpha;
                let d_alpha = d_out.dot(color - below[i]);
                d_alpha_channel += d_alpha * coverage * ramp_alpha;
                d_winding[i] = d_alpha * a * ramp_alpha * d_coverage_d_winding;
                let d_p = *d_out * (coverage * a);
                let d_a = -(coverage * a) * d_out.dot(below[i]);
                *d_out *= 1. - alpha;
                let [r, g, b, _] = path.color;
                let ramp = gradient.ramp(Vec3::new(r, g, b), t);
                d_from += d_p * ramp.from_weight;
                if let Some(d_geometry) = &mut d_geometry {
                    let [r, g, b, a1] = gradient.to();
                    d_to += d_p * (ramp.to_weight * a1);
                    d_to_alpha += ramp.to_weight * (d_p.dot(Vec3::new(r, g, b)) + d_a);
                    let d_t = d_p.dot(ramp.premultiplied_slope) + d_a * ramp.alpha_slope;
                    if d_t != 0. {
                        d_geometry.add_geometry(&gradient.d_t(p), d_t);
                    }
                }
                continue;
            }
            let (coverage, d_coverage_d_winding) = fill_coverage(w, path.fill_rule);
            let alpha = coverage * a;

            // out = color * alpha + below * (1 - alpha)
            let d_alpha = d_out.dot(color - below[i]);
            let d_color = *d_out * alpha;
            // What the end's alpha would change, were it moved off 1
            let d_below = alpha * d_out.dot(below[i]);
            d_alpha_channel += d_alpha * coverage;
            d_winding[i] = d_alpha * a * d_coverage_d_winding;
            *d_out *= 1. - alpha;

            // color = from + t * (to - from) on a two-stop ramp, the piece of the ramp t lies
            // on otherwise; the colors of the stops between the ends are held
            let Some(gradient) = &path.gradient else {
                d_from += d_color;
                continue;
            };
            let [r, g, b, _] = path.color;
            let ramp = gradient.ramp(Vec3::new(r, g, b), t);
            d_from += d_color * ramp.from_weight;
            if let Some(d_geometry) = &mut d_geometry {
                d_to += d_color * ramp.to_weight;
                d_to_alpha +=
                    ramp.to_weight * (d_color.dot(Vec3::from(gradient.to_rgb())) - d_below);
                let d_t = d_color.dot(ramp.slope);
                if d_t != 0. {
                    d_geometry.add_geometry(&gradient.d_t(p), d_t);
                }
            }
        }
        path_grad.color = [d_from.x, d_from.y, d_from.z, d_alpha_channel];
        if let Some(mut g) = d_geometry {
            *g.to_mut() = [d_to.x, d_to.y, d_to.z, d_to_alpha];
            path_grad.gradient = Some(g);
        }

        let d_flat = rasterize_backward(polys, winding, &d_winding);
        path_grad.subpaths = path.points_gradient(polys, &d_flat);
    }

    grad
}
