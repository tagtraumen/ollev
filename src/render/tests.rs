use super::backward::{backward, render};
use super::{FillRule, Gradient, Path, Scene, Subpath};
use glam::{Vec2, Vec3};

/// What the test scene is composited over.
const BACKGROUND: [f32; 3] = [0.9, 0.9, 0.85];

fn test_scene(fill_rule: FillRule) -> Scene {
    // A curved blob that sticks out of the left image edge, so the left cover bucket is exercised
    let blob = Subpath {
        points: [
            (-4., 6.),
            (6., -2.),
            (18., 3.),
            (20., 10.),
            (22., 17.),
            (10., 22.),
            (3., 18.),
            (-3., 15.),
            (-8., 12.),
        ]
        .map(|(x, y)| Vec2::new(x + 0.3, y + 0.4))
        .to_vec(),
    };
    // A self-intersecting star, which differs between the fill rules, with one horizontal edge
    let star = Subpath::polygon(
        &[(14., 2.2), (19.5, 19.3), (5., 8.5), (23., 8.5), (8.5, 19.3)]
            .map(|(x, y)| Vec2::new(x + 0.3, y)),
    );
    // A rectangle with edges tilted by a hundred thousandth of a pixel within their rows
    let sliver = Subpath::polygon(
        &[
            (3.2, 13.4),
            (15.7, 13.40001),
            (15.70001, 21.6),
            (3.2, 21.59999),
        ]
        .map(Vec2::from),
    );
    Scene {
        width: 24,
        height: 24,
        paths: vec![
            Path {
                subpaths: vec![blob],
                color: [0.8, 0.2, 0.1, 1.],
                gradient: None,
                fill_rule,
                blur: 0.,
            },
            Path {
                subpaths: vec![star],
                color: [0.1, 0.3, 0.9, 0.7],
                gradient: Some(Gradient::Linear {
                    start: Vec2::new(6.3, 4.1),
                    end: Vec2::new(19.2, 17.6),
                    to: [0.9, 0.7, 0.2, 0.6],
                    stops: super::Stops::new(&[(0.4, [0.2, 0.9, 0.4, 0.3])]),
                }),
                fill_rule,
                blur: 0.,
            },
            Path {
                subpaths: vec![sliver],
                color: [0.2, 0.8, 0.3, 0.6],
                gradient: Some(Gradient::Radial {
                    center: Vec2::new(9.7, 17.3),
                    focus: Vec2::ZERO,
                    radius: 7.4,
                    transform: glam::Mat2::from_cols(Vec2::new(1.3, 0.4), Vec2::new(-0.2, 0.8)),
                    to: [0.1, 0.1, 0.6, 1.],
                    stops: super::Stops::NONE,
                }),
                fill_rule,
                blur: 0.,
            },
            Path {
                subpaths: vec![Subpath::polygon(
                    &[(12.2, 1.3), (22.6, 4.1), (20.3, 15.8), (11.4, 12.7)]
                        .map(Vec2::from),
                )],
                color: [0.9, 0.8, 0.1, 0.8],
                gradient: Some(Gradient::Radial {
                    center: Vec2::new(16.6, 8.3),
                    focus: Vec2::new(2.1, -1.4),
                    radius: 6.9,
                    transform: glam::Mat2::IDENTITY,
                    to: [0.3, 0.1, 0.5, 0.],
                    stops: super::Stops::new(&[(0.6, [0.8, 0.3, 0.2, 1.])]),
                }),
                fill_rule,
                blur: 0.,
            },
        ],
    }
}

/// A fixed pseudo-random linear functional of the image, which gives every pixel a distinct weight.
fn loss_weights(len: usize) -> Vec<Vec3> {
    let mut state = 0x2545_f491_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as f32 / u32::MAX as f32 - 0.5
    };
    (0..len)
        .map(|_| Vec3::new(next(), next(), next()))
        .collect()
}

fn loss(scene: &Scene, weights: &[Vec3]) -> f64 {
    let (image, _) = render(scene, BACKGROUND);
    image
        .iter()
        .zip(weights)
        .map(|(p, w)| p.dot(*w) as f64)
        .sum()
}

fn gradient_of(scene: &mut Scene, path: usize) -> &mut Gradient {
    scene.paths[path].gradient.as_mut().unwrap()
}

fn check(fill_rule: FillRule) {
    let scene = test_scene(fill_rule);
    let weights = loss_weights((scene.width * scene.height) as usize);
    let (_, tape) = render(&scene, BACKGROUND);
    let grad = backward(&scene, &tape, &weights);

    let eps = 2e-3;
    let mut worst: f64 = 0.;
    let mut compare = |name: String, analytic: f32, perturb: &dyn Fn(&mut Scene, f32)| {
        let mut plus = scene.clone();
        perturb(&mut plus, eps);
        let mut minus = scene.clone();
        perturb(&mut minus, -eps);
        let numeric = (loss(&plus, &weights) - loss(&minus, &weights)) / (2. * eps as f64);
        let error = (analytic as f64 - numeric).abs()
            / (numeric.abs().max(analytic.abs() as f64).max(0.05));
        worst = worst.max(error);
        assert!(
            error < 0.05,
            "{name}: analytic {analytic} vs numeric {numeric}"
        );
    };

    for (p, path) in scene.paths.iter().enumerate() {
        for c in 0..4 {
            compare(
                format!("path {p} color {c}"),
                grad.paths[p].color[c],
                &|s, e| s.paths[p].color[c] += e,
            );
        }
        if let Some(mut g) = grad.paths[p].gradient {
            for c in 0..4 {
                compare(format!("path {p} gradient to {c}"), g.to()[c], &|s, e| {
                    gradient_of(s, p).to_mut()[c] += e;
                });
            }
            let mut analytic = Vec::new();
            g.visit_geometry(|v| analytic.push(*v));
            for (k, value) in analytic.into_iter().enumerate() {
                compare(format!("path {p} gradient geometry {k}"), value, &|s, e| {
                    let mut index = 0;
                    gradient_of(s, p).visit_geometry(|v| {
                        if index == k {
                            *v += e;
                        }
                        index += 1;
                    });
                });
            }
        }
        for (sp, subpath) in path.subpaths.iter().enumerate() {
            for i in 0..subpath.points.len() {
                let g = grad.paths[p].subpaths[sp][i];
                compare(format!("path {p} point {i} x"), g.x, &|s, e| {
                    s.paths[p].subpaths[sp].points[i].x += e;
                });
                compare(format!("path {p} point {i} y"), g.y, &|s, e| {
                    s.paths[p].subpaths[sp].points[i].y += e;
                });
            }
        }
    }
    println!("{fill_rule:?}: worst relative error {worst:.4}");
}

#[test]
fn gradients_nonzero() {
    check(FillRule::NonZero);
}

#[test]
fn gradients_even_odd() {
    check(FillRule::EvenOdd);
}

#[test]
fn pixel_aligned_square_has_exact_coverage() {
    let square =
        Subpath::polygon(&[(2., 2.), (6., 2.), (6., 6.), (2., 6.)].map(Vec2::from));
    let scene = Scene {
        width: 8,
        height: 8,
        paths: vec![Path {
            subpaths: vec![square],
            color: [1., 1., 1., 1.],
            gradient: None,
            fill_rule: FillRule::NonZero,
            blur: 0.,
        }],
    };
    let (image, _) = render(&scene, [0.; 3]);
    for y in 0..8 {
        for x in 0..8 {
            let inside = (2..6).contains(&x) && (2..6).contains(&y);
            let value = image[y * 8 + x].x;
            assert!(
                (value - if inside { 1. } else { 0. }).abs() < 1e-5,
                "pixel ({x}, {y}) = {value}"
            );
        }
    }
}

/// Pixel-aligned edges nudged by an optimizer straddle a pixel boundary with a tiny slope. Their
/// gradients must stay well conditioned, even though finite differences cannot check them there.
#[test]
fn gradients_stay_bounded_for_edges_straddling_pixel_boundaries() {
    let tilt = 1e-5;
    let square = Subpath::polygon(
        &[
            (4., 4. - tilt),
            (12. + tilt, 4. + tilt),
            (12. - tilt, 12. - tilt),
            (4. + tilt, 12. + tilt),
        ]
        .map(Vec2::from),
    );
    let scene = Scene {
        width: 16,
        height: 16,
        paths: vec![Path {
            subpaths: vec![square],
            color: [0., 0., 0., 1.],
            gradient: None,
            fill_rule: FillRule::NonZero,
            blur: 0.,
        }],
    };
    let weights = loss_weights((scene.width * scene.height) as usize);
    let (_, tape) = render(&scene, [1.; 3]);
    let grad = backward(&scene, &tape, &weights);
    // Each pixel weight is at most 0.5 per channel, so a point can move at most a few pixels' worth
    let mut worst: f32 = 0.;
    grad.visit(|g| worst = worst.max(g.abs()));
    assert!(worst.is_finite() && worst < 20., "max |gradient| {worst}");
}
