use super::*;
use crate::render::Stops;
use crate::render::distance::Contour;
use glam::{DVec4, Mat2};

/// The parameter a host computes from the shape alone.
fn host_t(shape: &GradientShape, p: DVec2) -> f64 {
    let t = match shape {
        GradientShape::Linear { start, end } => {
            let axis = *end - *start;
            (p - *start).dot(axis) / axis.length_squared()
        }
        GradientShape::Radial {
            transform: m,
            focus,
        } => {
            // Into the unit circle's plane, then the circle through the point
            let (a, b, c, d) = (m[0], m[1], m[2], m[3]);
            let q = p - DVec2::new(m[4], m[5]);
            let det = a * d - b * c;
            let u = DVec2::new(d * q.x - c * q.y, a * q.y - b * q.x) / det;
            let (d, e) = (u - *focus, -*focus);
            let k = e.length_squared() - 1.;
            let de = d.dot(e);
            (de - (de * de - k * d.length_squared()).max(0.).sqrt()) / k
        }
        GradientShape::Path {
            curve,
            closed,
            closure,
            near,
            far,
        } => {
            let mut line: Vec<DVec2> = Vec::new();
            for c in curve {
                for s in 0..64 {
                    let t = s as f64 / 64.;
                    let u = 1. - t;
                    line.push(
                        c[0] * (u * u * u)
                            + c[1] * (3. * u * u * t)
                            + c[2] * (3. * u * t * t)
                            + c[3] * (t * t * t),
                    );
                }
            }
            line.push(curve[curve.len() - 1][3]);
            // The rays measure for an open curve, the closure's corners only close the side
            let mut measured = line.clone();
            let mut ring = line.clone();
            if !*closed {
                measured.insert(0, closure[closure.len() - 1]);
                measured.push(closure[0]);
                ring.extend(closure.iter().copied());
            }
            let distance = measured
                .windows(2)
                .map(|w| {
                    let ab = w[1] - w[0];
                    let t = ((p - w[0]).dot(ab) / ab.length_squared().max(1e-12)).clamp(0., 1.);
                    (w[0] + ab * t).distance(p)
                })
                .fold(f64::INFINITY, f64::min);
            let mut inside = false;
            for k in 0..ring.len() {
                let (a, b) = (ring[k], ring[(k + 1) % ring.len()]);
                if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x)
                {
                    inside = !inside;
                }
            }
            let d = if inside { -distance } else { distance };
            (d - near) / (far - near)
        }
    };
    t.clamp(0., 1.)
}

/// The stops interpolated in straight colour, as SVG and Graphite draw them, premultiplied.
fn host_color(stops: &[Stop], t: f64) -> DVec4 {
    let k = stops
        .windows(2)
        .position(|w| t <= w[1].offset)
        .unwrap_or(stops.len() - 2);
    let (a, b) = (stops[k], stops[k + 1]);
    let u = ((t - a.offset) / (b.offset - a.offset).max(1e-12)).clamp(0., 1.);
    let c = DVec4::from(a.color).lerp(DVec4::from(b.color), u);
    DVec4::new(c.x * c.w, c.y * c.w, c.z * c.w, c.w)
}

fn premultiplied(c: [f64; 4]) -> DVec4 {
    DVec4::new(c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3])
}

fn gradients() -> Vec<Gradient> {
    let alpha = Stops::new(&[(0.3, [0.2, 0.8, 0.1, 0.6]), (0.7, [0.1, 0.1, 0.9, 1.])]);
    let square = [
        [(4., 4.), (12., 4.), (20., 4.), (28., 4.)],
        [(28., 4.), (28., 12.), (28., 20.), (28., 28.)],
        [(28., 28.), (20., 28.), (12., 28.), (4., 28.)],
        [(4., 28.), (4., 20.), (4., 12.), (4., 4.)],
    ]
    .map(|c| c.map(|(x, y)| Vec2::new(x, y)));
    let line = [[(2., 16.), (10., 14.), (22., 18.), (30., 16.)].map(|(x, y)| Vec2::new(x, y))];
    let from = Vec3::new(0.9, 0.2, 0.1);
    [
        Ramp::Linear {
            start: Vec2::new(2., 3.),
            end: Vec2::new(30., 20.),
            to: [1., 1., 0., 0.],
            stops: alpha,
        },
        Ramp::Radial {
            center: Vec2::splat(16.),
            focus: Vec2::new(3., -2.),
            radius: 10.,
            transform: Mat2::from_cols_array(&[1.2, 0.3, 0., 0.7]),
            to: [0.1, 0.3, 0.9, 1.],
            stops: Stops::opaque(&[(0.4, [0.5, 0.5, 0.5])]),
        },
        Ramp::Radial {
            center: Vec2::new(12., 18.),
            focus: Vec2::ZERO,
            radius: 9.,
            transform: Mat2::IDENTITY,
            to: [0.9, 0.2, 0.1, 0.],
            stops: Stops::NONE,
        },
        Ramp::Path {
            contour: Contour::new(&square).unwrap(),
            near: -6.,
            far: 10.,
            to: [0.1, 0.9, 0.2, 1.],
            stops: alpha,
        },
        Ramp::Path {
            contour: Contour::open(&line, Vec2::ZERO, Vec2::splat(32.)).unwrap(),
            near: -8.,
            far: 8.,
            to: [0., 0., 0., 1.],
            stops: Stops::NONE,
        },
    ]
    .into_iter()
    .map(|r| Gradient::new(from, r))
    .collect()
}

/// A host reading only the shape and the stops paints what the model renders: within half an
/// 8-bit level by the stops, and the curve's flattening's tenth of a pixel for a path.
#[test]
fn the_host_form_paints_as_the_model() {
    for g in gradients() {
        let (shape, stops) = (g.shape(), g.stops());
        assert_eq!(stops[0].offset, 0.);
        assert_eq!(stops[stops.len() - 1].offset, 1.);
        let slack = match shape {
            GradientShape::Path { near, far, .. } => 0.15 / (far - near),
            _ => 1e-4,
        };
        for y in 0..32 {
            for x in 0..32 {
                let p = DVec2::new(x as f64 + 0.5, y as f64 + 0.5);
                let (t, want) = (g.t(p), host_t(&shape, p));
                assert!(
                    (t - want).abs() < slack,
                    "{shape:?} at {p}: t {t} vs {want}"
                );
                let got = host_color(&stops, want);
                let model = premultiplied(g.color(t));
                let off = (got - model).abs().max_element();
                assert!(
                    off < 1. / 255. + 4. * slack,
                    "{shape:?} at {p}: {got} vs {model}"
                );
            }
        }
    }
}

/// A gradient under a similarity reads at the mapped point what it read at the point.
#[test]
fn a_mapped_gradient_reads_the_same() {
    for g in gradients() {
        // Turned by 30 degrees, scaled by 2.5 and moved
        let (k, (sin, cos)) = (2.5, std::f64::consts::FRAC_PI_6.sin_cos());
        let m = [k * cos, k * sin, -k * sin, k * cos, 7., -3.];
        let big = g.mapped(&m);
        for p in [
            DVec2::new(5., 7.),
            DVec2::new(16., 16.),
            DVec2::new(27., 3.),
        ] {
            let q = DVec2::new(
                m[0] * p.x + m[2] * p.y + m[4],
                m[1] * p.x + m[3] * p.y + m[5],
            );
            assert!((big.t(q) - g.t(p)).abs() < 1e-4, "{:?}", g.shape());
        }
    }
}
