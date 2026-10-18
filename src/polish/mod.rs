//! The polish: a few joint gradient steps over the finished scene, on the objects still in
//! error, every one moved at once by the derivative of the whole scene's error (through what
//! lies above and below it), so neighbours meeting at a seam move together.

use crate::render::ParamKind;
use glam::{Vec2, Vec3};

use crate::edges::{EdgeField, EdgeTerm};
use crate::{Context, Scene, Window};

// Steps of the polish: the knee of the error removed against time.
pub const STEPS: usize = 10;
/// The most a step moves a point, in pixels: the anti-aliasing reach, as a polish refines an
/// edge where it is and does not move it to another.
const POINT_CAP: f32 = crate::edges::REACH as f32;
/// The most a step moves a colour channel: the noise, for the same reason.
const COLOR_CAP: f32 = crate::translucency::NOISE;
/// The first step's move, as a share of the most: it reads each parameter's curvature, so it
/// stays inside the quadratic range of the sharpest edge's coverage.
const PROBE: f32 = 1. / 50.;
/// The frame the scene's coverages are rasterized in (see [`crate::scene::coverage_raster`]).
const FRAME: u32 = i32::MAX as u32;
/// The frame its blurred coverages are rasterized in (see [`crate::scene::blurred_raster`]).
const BLUR_FRAME: u32 = 1 << 24;

/// What [`polish`] did, for the log.
#[derive(Default, Debug)]
pub struct Report {
    pub active: usize,
    pub steps: usize,
    pub before: f64,
    pub after: f64,
}

/// One polished shape's parameters within the flat vector.
struct Slot {
    /// Its place in the stack.
    k: usize,
    start: usize,
    /// Whether its outline moves (a stroke's and a primitive's do not).
    points: bool,
    gradient: usize,
}

/// The whole scene's error per pixel and its derivative by the pixel's colour: the plain
// squared error with the edge term, as the economy judges (with a tolerance instead, an edge
/// moved within the noise would cost nothing).
struct Loss {
    weights: Option<EdgeField>,
    lambda: f64,
}

impl Loss {
    fn pixel(&self, ctx: &Context, i: usize, r: Vec3) -> (f32, Vec3) {
        let w = self
            .weights
            .as_ref()
            .map_or(1., |f| f.weight(i, self.lambda));
        let d = r - ctx.target[i];
        (w * d.length_squared(), 2. * w * d)
    }
}

/// Moves the objects of `scene` still in error by [`STEPS`] joint steps down its error, keeping
/// the best scene seen. Each parameter steps by its gradient over its curvature, read off its
/// last move (a diagonal secant step), as an edge's minimum is sharper than any common step.
pub async fn polish(ctx: &Context, scene: &mut Scene, edge: Option<EdgeTerm>) -> Report {
    let mut report = Report::default();
    if scene.shapes.len() < 2 {
        return report;
    }
    let loss = Loss {
        weights: edge.map(|_| EdgeField::of_target(&ctx.target, ctx.width, ctx.height)),
        lambda: edge.map_or(0., |t| t.lambda),
    };
    let full = Window::full(ctx.width, ctx.height);
    // As the economy leaves them: the background, the translucent layers and what lies
    // beneath and above them, the clips; and a repeat's members, drawn as one prototype
    let fixed = crate::economy::fixed(scene);
    let members = crate::economy::repeated(scene);
    let movable: Vec<bool> = scene
        .shapes
        .iter()
        .map(|s| !fixed.contains(&s.id) && !members.contains(&s.id))
        .collect();
    // The objects still in error: any error they show
    let (pixels, _) = forward(scene, full, &[]);
    let shown = residuals(ctx, scene, full, &pixels, &loss, &movable);
    let active: Vec<bool> = (0..scene.shapes.len())
        .map(|k| movable[k] && shown[k] > 0.)
        .collect();
    report.active = active.iter().filter(|a| **a).count();
    if report.active == 0 {
        return report;
    }
    let window = active_window(ctx, scene, &active);
    let (slots, mut params, kinds) = layout(scene, &active);
    let cap = |kind: ParamKind| match kind {
        ParamKind::Point => POINT_CAP,
        ParamKind::Color => COLOR_CAP,
    };
    let mut best: Option<(f64, Vec<f32>)> = None;
    // The parameters and gradient of the last step, for the curvature
    let mut last: Option<(Vec<f32>, Vec<f32>)> = None;
    for step in 0..=STEPS {
        set_params(scene, &slots, &params);
        let (pixels, below) = forward(scene, window, &slots);
        let (value, d_image) = image_loss(ctx, window, &pixels, &loss);
        if step == 0 {
            report.before = value;
        }
        if best.as_ref().is_none_or(|b| value < b.0) {
            best = Some((value, params.clone()));
        }
        if step == STEPS {
            break;
        }
        crate::progress::poll().await;
        let grad = backward(scene, window, &slots, &below, d_image, params.len());
        let next: Vec<f32> = (0..params.len())
            .map(|j| {
                let (g, most) = (grad[j], cap(kinds[j]));
                let curvature = last.as_ref().and_then(|(p, q)| {
                    let d = params[j] - p[j];
                    (d != 0.).then(|| (g - q[j]) / d).filter(|h| *h > 0.)
                });
                let moved = match curvature {
                    Some(h) => params[j] - (g / h).clamp(-most, most),
                    None if g != 0. => params[j] - PROBE * most * g.signum(),
                    None => params[j],
                };
                match kinds[j] {
                    ParamKind::Color => moved.clamp(0., 1.),
                    ParamKind::Point => moved,
                }
            })
            .collect();
        last = Some((std::mem::replace(&mut params, next), grad));
        untangle(scene, &slots, &mut params);
        report.steps += 1;
    }
    let (value, params) = best.expect("step 0 measured");
    report.after = value;
    set_params(scene, &slots, &params);
    report
}

/// The pixels of every active shape's painted box, grown by the anti-aliasing reach.
fn active_window(ctx: &Context, scene: &Scene, active: &[bool]) -> Window {
    let (mut min, mut max) = (Vec2::INFINITY, Vec2::NEG_INFINITY);
    for (s, _) in scene.shapes.iter().zip(active).filter(|(_, a)| **a) {
        min = min.min(s.paint_min());
        max = max.max(s.paint_max());
    }
    Window::around(min, max, 2., ctx.width, ctx.height)
}

/// The flat parameters of the active shapes: per shape its control points (where its
/// outline moves), the blur's σ, its colour, its gradient's geometry and end colour.
fn layout(scene: &Scene, active: &[bool]) -> (Vec<Slot>, Vec<f32>, Vec<ParamKind>) {
    let (mut slots, mut params, mut kinds) = (Vec::new(), Vec::new(), Vec::new());
    for (k, s) in scene.shapes.iter().enumerate() {
        if !active[k] {
            continue;
        }
        let points = s.stroke.is_none() && s.primitive.is_none();
        let start = params.len();
        if points {
            for p in s.loops.iter().flatten().flat_map(|c| [c[0], c[1], c[2]]) {
                params.extend([p.x, p.y]);
                kinds.extend([ParamKind::Point; 2]);
            }
            if s.blur > 0. {
                params.push(s.blur);
                kinds.push(ParamKind::Point);
            }
        }
        params.extend(s.fill.color.to_array());
        kinds.extend([ParamKind::Color; 3]);
        let mut gradient = 0;
        if let Some(mut g) = s.fill.gradient {
            g.visit_geometry(|v| {
                params.push(*v);
                kinds.push(ParamKind::Point);
                gradient += 1;
            });
            let [r, gg, b, _] = g.to();
            params.extend([r, gg, b]);
            kinds.extend([ParamKind::Color; 3]);
        }
        slots.push(Slot {
            k,
            start,
            points,
            gradient,
        });
    }
    (slots, params, kinds)
}

/// Writes the flat parameters back into the shapes.
fn set_params(scene: &mut Scene, slots: &[Slot], params: &[f32]) {
    for slot in slots {
        let s = &mut scene.shapes[slot.k];
        let mut at = slot.start;
        if slot.points {
            for cubics in &mut s.loops {
                let n = cubics.len();
                let first = at;
                for (i, cubic) in cubics.iter_mut().enumerate() {
                    let p = |j: usize| Vec2::new(params[j], params[j + 1]);
                    let next = if i + 1 == n { first } else { at + 6 };
                    *cubic = [p(at), p(at + 2), p(at + 4), p(next)];
                    at += 6;
                }
            }
            if s.blur > 0. {
                s.blur = params[at].max(crate::refine::LEAST_SIGMA);
                at += 1;
            }
            s.update_bounds();
        }
        s.fill.color = Vec3::new(params[at], params[at + 1], params[at + 2]);
        at += 3;
        if let Some(g) = &mut s.fill.gradient {
            let mut j = at;
            g.visit_geometry(|v| {
                *v = params[j];
                j += 1;
            });
            let to = g.to_mut();
            to[..3].copy_from_slice(&params[j..j + 3]);
        }
        *s.paint.lock() = None;
    }
    scene.change();
}

/// Keeps every moved cubic free of loops, as the single shapes' fits do.
fn untangle(scene: &Scene, slots: &[Slot], params: &mut [f32]) {
    for slot in slots.iter().filter(|s| s.points) {
        let mut at = slot.start;
        for cubics in &scene.shapes[slot.k].loops {
            let n = cubics.len();
            let first = at;
            for i in 0..n {
                let next = if i + 1 == n { first } else { at + 6 };
                let p = |j: usize| Vec2::new(params[j], params[j + 1]);
                let c = crate::tracer::fit::untangle([p(at), p(at + 2), p(at + 4), p(next)]);
                params[at + 2..at + 6].copy_from_slice(&[c[1].x, c[1].y, c[2].x, c[2].y]);
                at += 6;
            }
        }
    }
}

/// The scene rendered over `window`, and what lies below each slot's shape over its box
/// within the window, row by row.
fn forward(scene: &Scene, window: Window, slots: &[Slot]) -> (Vec<Vec3>, Vec<Vec<Vec3>>) {
    let mut pixels = vec![Vec3::ONE; window.width * window.height];
    let mut below = Vec::with_capacity(slots.len());
    let mut next = slots.iter().peekable();
    for k in 0..scene.shapes.len() {
        if next.peek().is_some_and(|s| s.k == k) {
            next.next();
            let b = boxed(scene, k, window);
            below.push(
                b.map(|b| {
                    b.pixels(window.width)
                        .map(|j| pixels[j])
                        .collect::<Vec<Vec3>>()
                })
                .unwrap_or_default(),
            );
        }
        scene.composite_onto(window, k, &mut pixels);
    }
    (pixels, below)
}

/// The box of what the shape at `k` paints within `window`, in the window's frame.
fn boxed(scene: &Scene, k: usize, window: Window) -> Option<Window> {
    let r = scene.shapes[k].painted();
    let x0 = r.x0.max(window.x);
    let x1 = (r.x0 + r.w).min(window.x + window.width);
    let y0 = r.y0.max(window.y);
    let y1 = (r.y0 + r.h).min(window.y + window.height);
    (x1 > x0 && y1 > y0).then(|| Window {
        x: x0 - window.x,
        y: y0 - window.y,
        width: x1 - x0,
        height: y1 - y0,
    })
}

/// The error of `pixels` over `window` and its derivative by each pixel.
fn image_loss(ctx: &Context, window: Window, pixels: &[Vec3], loss: &Loss) -> (f64, Vec<Vec3>) {
    let mut total = 0f64;
    let d = window
        .pixels(ctx.width)
        .zip(pixels)
        .map(|(i, &r)| {
            let (v, d) = loss.pixel(ctx, i, r);
            total += v as f64;
            d
        })
        .collect();
    (total, d)
}

/// Per shape, the error over the pixels it shows, each counted by how much of it shows there.
fn residuals(
    ctx: &Context,
    scene: &Scene,
    window: Window,
    pixels: &[Vec3],
    loss: &Loss,
    movable: &[bool],
) -> Vec<f64> {
    let error: Vec<f32> = window
        .pixels(ctx.width)
        .zip(pixels)
        .map(|(i, &r)| loss.pixel(ctx, i, r).0)
        .collect();
    let mut through = vec![1f32; error.len()];
    let mut out = vec![0.; scene.shapes.len()];
    for k in (0..scene.shapes.len()).rev() {
        walk(scene, k, window, |j, alpha, _, _| {
            if movable[k] {
                out[k] += (error[j] * alpha * through[j]) as f64;
            }
            through[j] *= 1. - alpha;
        });
    }
    out
}

/// Calls `f(j, alpha, coverage share, (colour, alpha of the paint))` for each pixel `j` of
/// `window` the shape at `k` reaches, `alpha` what it composites at and the share its
/// coverage's derivative carries (the clip times the paint's alpha).
fn walk(scene: &Scene, k: usize, window: Window, mut f: impl FnMut(usize, f32, f32, (Vec3, f32))) {
    let shape = &scene.shapes[k];
    let Some(b) = boxed(scene, k, window) else {
        return;
    };
    let raster = shape.painted();
    let shared = Window {
        x: b.x + window.x,
        y: b.y + window.y,
        ..b
    };
    let clip = scene.clip_mask(shared, k);
    let paint = shape.paint();
    for y in shared.y..shared.y + shared.height {
        for x in shared.x..shared.x + shared.width {
            let c = raster.values[(y - raster.y0) * raster.w + x - raster.x0];
            if c == 0. {
                continue;
            }
            let m = clip
                .as_ref()
                .map_or(1., |m| m[(y - shared.y) * shared.width + x - shared.x]);
            let (color, a) = match &paint {
                Some(p) => p.at(x, y),
                None => shape.fill.paint_pixel(x, y),
            };
            let j = (y - window.y) * window.width + x - window.x;
            f(j, c * m * a, m * a, (color, a));
        }
    }
}

/// The derivative of the error by the flat parameters, `d_image` walked down the stack.
fn backward(
    scene: &Scene,
    window: Window,
    slots: &[Slot],
    below: &[Vec<Vec3>],
    mut d_image: Vec<Vec3>,
    n: usize,
) -> Vec<f32> {
    let mut grad = vec![0f32; n];
    let mut next = slots.len();
    for k in (0..scene.shapes.len()).rev() {
        let slot = (next > 0 && slots[next - 1].k == k).then(|| {
            next -= 1;
            next
        });
        let Some(si) = slot else {
            walk(scene, k, window, |j, alpha, _, _| d_image[j] *= 1. - alpha);
            continue;
        };
        let s = &slots[si];
        let shape = &scene.shapes[k];
        let raster = shape.painted();
        let mut d_cov = vec![0f32; raster.values.len()];
        let (mut d_color, mut d_to) = (Vec3::ZERO, Vec3::ZERO);
        let mut d_geometry = shape.fill.gradient.map(|g| g.zeroed());
        let Some(b) = boxed(scene, k, window) else {
            continue;
        };
        let under = &below[si];
        walk(scene, k, window, |j, alpha, share, (color, a)| {
            let d = d_image[j];
            let (x, y) = (window.x + j % window.width, window.y + j / window.width);
            let bj = (y - window.y - b.y) * b.width + (x - window.x - b.x);
            let u = under[bj];
            d_cov[(y - raster.y0) * raster.w + x - raster.x0] = share * d.dot(color - u);
            let coverage = alpha / a.max(1e-12);
            match &shape.fill.gradient {
                None => d_color += d * alpha,
                Some(g) => {
                    let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let beta = coverage * shape.fill.opacity;
                    let ramp = g.ramp(shape.fill.color, g.t(p));
                    d_color += d * (beta * ramp.from_weight);
                    d_to += d * (beta * ramp.to_weight * g.to()[3]);
                    let d_t = beta * d.dot(ramp.premultiplied_slope - ramp.alpha_slope * u);
                    if d_t != 0.
                        && let Some(dg) = &mut d_geometry
                    {
                        dg.add_geometry(&g.d_t(p), d_t);
                    }
                }
            }
            d_image[j] *= 1. - alpha;
        });
        let mut at = s.start;
        if s.points {
            let frame = if shape.blur > 0. { BLUR_FRAME } else { FRAME };
            let (d_points, d_sigma) =
                crate::render::coverage_backward(&shape.path(Vec2::ZERO), frame, frame, &d_cov);
            for p in d_points.iter().flatten() {
                grad[at] = p.x;
                grad[at + 1] = p.y;
                at += 2;
            }
            if shape.blur > 0. {
                grad[at] = d_sigma;
                at += 1;
            }
        }
        grad[at..at + 3].copy_from_slice(&d_color.to_array());
        at += 3;
        if let Some(mut dg) = d_geometry {
            let mut j = at;
            dg.visit_geometry(|v| {
                grad[j] = *v;
                j += 1;
            });
            debug_assert_eq!(j - at, s.gradient);
            // A circle stays a circle and a centred radial centred, as the fit drew them: the
            // transform's entries (the last four) and the focus (the two before) hold
            if let Some(crate::render::Gradient::Radial {
                transform, focus, ..
            }) = shape.fill.gradient
            {
                if transform == glam::Mat2::IDENTITY {
                    grad[j - 4..j].fill(0.);
                }
                if focus == Vec2::ZERO {
                    grad[j - 6..j - 4].fill(0.);
                }
            }
            grad[j..j + 3].copy_from_slice(&d_to.to_array());
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::tests::square;
    use crate::{Fill, Shape};

    /// The walked gradient matches the error's finite differences, for the points and colour
    /// of a flat shape, a blurred one's σ, a linear gradient's ends and a clipped shape's.
    #[test]
    fn the_gradient_matches_finite_differences() {
        let (w, h) = (24, 20);
        let rgb: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                [(x * 9) as u8, (y * 11) as u8, ((x + y) * 5) as u8]
            })
            .collect();
        let ctx = Context::new(&rgb, w, h, 1.);
        let mut scene = Scene::with_background(w, h, Vec3::splat(0.4));
        let a = scene.push(Shape::new(
            square(Vec2::new(3.3, 2.6), Vec2::new(15.2, 14.7)),
            Vec3::new(0.8, 0.2, 0.3),
        ));
        scene.push(
            Shape::new(
                square(Vec2::new(9.4, 6.2), Vec2::new(20.1, 17.3)),
                Fill {
                    color: Vec3::new(0.1, 0.6, 0.9),
                    gradient: Some(crate::render::Gradient::Linear {
                        start: Vec2::new(10., 7.),
                        end: Vec2::new(19., 15.),
                        to: [0.9, 0.8, 0.1, 1.],
                        stops: crate::render::Stops::NONE,
                    }),
                    opacity: 0.8,
                },
            )
            .with_blur(1.3),
        );
        scene.push(Shape {
            clip: Some(a),
            ..Shape::new(
                square(Vec2::new(1.7, 8.3), Vec2::new(8.6, 18.4)),
                Vec3::new(0.3, 0.9, 0.5),
            )
        });
        let loss = Loss {
            weights: None,
            lambda: 0.,
        };
        let window = Window::full(w, h);
        // the clip's parent stays, as the pass leaves it
        let active = [false, false, true, true];
        let (slots, params, _) = layout(&scene, &active);
        let value = |scene: &mut Scene, p: &[f32]| {
            set_params(scene, &slots, p);
            let (pixels, _) = forward(scene, window, &slots);
            image_loss(&ctx, window, &pixels, &loss).0
        };
        set_params(&mut scene, &slots, &params);
        let (pixels, below) = forward(&scene, window, &slots);
        let (_, d) = image_loss(&ctx, window, &pixels, &loss);
        let grad = backward(&scene, window, &slots, &below, d, params.len());
        for (k, &g) in grad.iter().enumerate() {
            let step = 1e-2;
            let mut p = params.clone();
            p[k] += step;
            let up = value(&mut scene, &p);
            p[k] -= 2. * step;
            let down = value(&mut scene, &p);
            let fd = ((up - down) / (2. * step as f64)) as f32;
            assert!(
                (fd - g).abs() <= 0.05 * fd.abs().max(g.abs()) + 0.05,
                "parameter {k}: {fd} by differences, {g} walked"
            );
        }
    }
}
