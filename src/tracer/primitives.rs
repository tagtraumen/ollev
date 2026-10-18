use glam::Vec2;

use crate::tracer::Cubic;

/// Handle length for a quarter circle of radius 1 drawn as one cubic.
const KAPPA: f32 = 0.552_284_8;
/// Angles this close to an axis, in radians, snap to it.
const AXIS_SNAP: f32 = 3. * std::f32::consts::PI / 180.;

#[derive(Clone, Copy, Debug)]
pub enum Primitive {
    Circle {
        center: Vec2,
        radius: f32,
    },
    Ellipse {
        center: Vec2,
        radii: Vec2,
        angle: f32,
    },
    /// A rectangle with all four corners rounded by `radius`, which may be zero.
    RoundedRect {
        center: Vec2,
        half: Vec2,
        radius: f32,
        angle: f32,
    },
}

impl Primitive {
    /// Signed distance from `p` to the outline, negative inside.
    pub fn distance(&self, p: Vec2) -> f32 {
        match *self {
            Primitive::Circle { center, radius } => p.distance(center) - radius,
            Primitive::Ellipse {
                center,
                radii,
                angle,
            } => {
                let q = Vec2::from_angle(-angle).rotate(p - center);
                let radii = radii.max(Vec2::splat(1e-3));
                // Inigo Quilez's first order approximation of the distance to an ellipse
                let k0 = (q / radii).length();
                let k1 = (q / (radii * radii)).length();
                if k1 < 1e-6 {
                    -radii.min_element()
                } else {
                    k0 * (k0 - 1.) / k1
                }
            }
            Primitive::RoundedRect {
                center,
                half,
                radius,
                angle,
            } => {
                let q = Vec2::from_angle(-angle).rotate(p - center).abs()
                    - (half - Vec2::splat(radius));
                q.max(Vec2::ZERO).length() + q.max_element().min(0.) - radius
            }
        }
    }

    fn segments(&self) -> usize {
        self.cubics().len()
    }

    pub fn params(&self) -> Vec<f32> {
        match *self {
            Primitive::Circle { center, radius } => vec![center.x, center.y, radius],
            Primitive::Ellipse {
                center,
                radii,
                angle,
            } => vec![center.x, center.y, radii.x, radii.y, angle],
            Primitive::RoundedRect {
                center,
                half,
                radius,
                angle,
            } => vec![center.x, center.y, half.x, half.y, radius, angle],
        }
    }

    pub fn with_params(&self, p: &[f32]) -> Primitive {
        match self {
            Primitive::Circle { .. } => Primitive::Circle {
                center: Vec2::new(p[0], p[1]),
                radius: p[2].max(0.1),
            },
            Primitive::Ellipse { .. } => Primitive::Ellipse {
                center: Vec2::new(p[0], p[1]),
                radii: Vec2::new(p[2], p[3]).max(Vec2::splat(0.1)),
                angle: p[4],
            },
            Primitive::RoundedRect { .. } => {
                let half = Vec2::new(p[2], p[3]).max(Vec2::splat(0.1));
                Primitive::RoundedRect {
                    center: Vec2::new(p[0], p[1]),
                    half,
                    radius: p[4].clamp(0., half.min_element()),
                    angle: p[5],
                }
            }
        }
    }

    /// The outline as closed cubics.
    pub fn cubics(&self) -> Vec<Cubic> {
        match *self {
            Primitive::Circle { center, radius } => ellipse_cubics(center, Vec2::splat(radius), 0.),
            Primitive::Ellipse {
                center,
                radii,
                angle,
            } => ellipse_cubics(center, radii, angle),
            Primitive::RoundedRect {
                center,
                half,
                radius,
                angle,
            } => rounded_rect_cubics(center, half, radius, angle),
        }
    }
}

fn ellipse_cubics(center: Vec2, radii: Vec2, angle: f32) -> Vec<Cubic> {
    let rotation = Vec2::from_angle(angle);
    let map = |p: Vec2| center + rotation.rotate(p * radii);
    (0..4)
        .map(|q| {
            let a = Vec2::from_angle(q as f32 * std::f32::consts::FRAC_PI_2);
            let b = a.perp();
            [map(a), map(a + b * KAPPA), map(b + a * KAPPA), map(b)]
        })
        .collect()
}

fn rounded_rect_cubics(center: Vec2, half: Vec2, radius: f32, angle: f32) -> Vec<Cubic> {
    let rotation = Vec2::from_angle(angle);
    let map = |p: Vec2| center + rotation.rotate(p);
    let (a, b, r) = (half.x, half.y, radius);
    // Around the outline: each side as (start, end, direction of travel), then a quarter arc
    // around the corner into the next side
    let sides = [
        (Vec2::new(a, -b + r), Vec2::new(a, b - r), Vec2::Y),
        (Vec2::new(a - r, b), Vec2::new(-a + r, b), Vec2::NEG_X),
        (Vec2::new(-a, b - r), Vec2::new(-a, -b + r), Vec2::NEG_Y),
        (Vec2::new(-a + r, -b), Vec2::new(a - r, -b), Vec2::X),
    ];
    let mut out = Vec::new();
    for (k, &(start, end, direction)) in sides.iter().enumerate() {
        if start.distance(end) > 1e-4 {
            out.push([
                map(start),
                map(start.lerp(end, 1. / 3.)),
                map(start.lerp(end, 2. / 3.)),
                map(end),
            ]);
        }
        if r > 0.3 {
            let (next_start, _, next_direction) = sides[(k + 1) % 4];
            out.push([
                map(end),
                map(end + direction * KAPPA * r),
                map(next_start - next_direction * KAPPA * r),
                map(next_start),
            ]);
        }
    }
    out
}

fn mean_distance(primitive: &Primitive, samples: &[Vec2]) -> f32 {
    samples
        .iter()
        .map(|&p| primitive.distance(p).abs())
        .sum::<f32>()
        / samples.len() as f32
}

/// Coordinate descent on the primitive's parameters with shrinking steps.
fn optimize(mut primitive: Primitive, samples: &[Vec2], scale: f32) -> Primitive {
    let mut params = primitive.params();
    let mut steps: Vec<f32> = params
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if i + 1 == params.len() && !matches!(primitive, Primitive::Circle { .. }) {
                0.05
            } else {
                0.05 * scale
            }
        })
        .collect();
    let mut best = mean_distance(&primitive, samples);
    for _ in 0..60 {
        let mut improved = false;
        for i in 0..params.len() {
            for sign in [1., -1.] {
                let mut trial = params.clone();
                trial[i] += sign * steps[i];
                let candidate = primitive.with_params(&trial);
                let d = mean_distance(&candidate, samples);
                if d < best {
                    best = d;
                    params = candidate.params();
                    primitive = candidate;
                    improved = true;
                }
            }
        }
        if !improved {
            steps.iter_mut().for_each(|s| *s *= 0.5);
        }
    }
    primitive
}

/// The dominant direction of a closed outline's edges, modulo a quarter turn, from a histogram of
/// edge directions weighted by length. Unlike principal axes it is defined for squares.
fn edge_direction(samples: &[Vec2]) -> f32 {
    const BINS: usize = 90;
    let mut histogram = [0f32; BINS];
    for (a, b) in samples.iter().zip(samples.iter().cycle().skip(1)) {
        let d = *b - *a;
        let length = d.length();
        if length < 1e-6 {
            continue;
        }
        let angle = d.y.atan2(d.x).rem_euclid(std::f32::consts::FRAC_PI_2);
        histogram[((angle / std::f32::consts::FRAC_PI_2 * BINS as f32) as usize).min(BINS - 1)] +=
            length;
    }
    // Smooth over neighboring bins so a slightly noisy edge still forms one peak
    let peak = (0..BINS).max_by(|&a, &b| {
        let around = |i: usize| {
            (0..5)
                .map(|k| histogram[(i + BINS + k - 2) % BINS])
                .sum::<f32>()
        };
        around(a).total_cmp(&around(b))
    });
    (peak.unwrap_or(0) as f32 + 0.5) / BINS as f32 * std::f32::consts::FRAC_PI_2
}

fn snap_angle(angle: f32) -> f32 {
    let quarter = std::f32::consts::FRAC_PI_2;
    let nearest = (angle / quarter).round() * quarter;
    if (angle - nearest).abs() < AXIS_SNAP {
        nearest
    } else {
        angle
    }
}

/// The primitive that matches the visible samples closest, if within the tolerance.
fn fit(samples: &[Vec2], all: &[Vec2], tolerance: f32) -> Option<Primitive> {
    let n = all.len() as f32;
    let center = all.iter().sum::<Vec2>() / n;
    let (min, max) = all
        .iter()
        .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &p| {
            (lo.min(p), hi.max(p))
        });
    let scale = (max - min).max_element().max(1.);

    // Circle by the algebraic least squares fit, which also works on a partly hidden arc
    let circle = {
        let (mut sxx, mut sxy, mut syy, mut sx, mut sy, mut sz, mut sxz, mut syz) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for p in samples {
            let (x, y) = (p.x as f64, p.y as f64);
            let z = x * x + y * y;
            sxx += x * x;
            sxy += x * y;
            syy += y * y;
            sx += x;
            sy += y;
            sz += z;
            sxz += x * z;
            syz += y * z;
        }
        let m = samples.len() as f64;
        // Solve [sxx sxy sx; sxy syy sy; sx sy m] [d e f] = -[sxz syz sz]
        let a = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, m]];
        let b = [-sxz, -syz, -sz];
        solve3(a, b).map(|[d, e, f]| {
            let c = Vec2::new((-d / 2.) as f32, (-e / 2.) as f32);
            let r = ((c.x as f64).powi(2) + (c.y as f64).powi(2) - f)
                .max(0.)
                .sqrt() as f32;
            Primitive::Circle {
                center: c,
                radius: r,
            }
        })
    };

    // Ellipse along the principal axes of the outline
    let ellipse = {
        let (mut cxx, mut cxy, mut cyy) = (0., 0., 0.);
        for p in all {
            let d = *p - center;
            cxx += d.x * d.x;
            cxy += d.x * d.y;
            cyy += d.y * d.y;
        }
        let angle = snap_angle(0.5 * (2. * cxy).atan2(cxx - cyy));
        let local = all
            .iter()
            .map(|&p| Vec2::from_angle(-angle).rotate(p - center));
        let extent = local.fold(Vec2::ZERO, |e, q| e.max(q.abs()));
        Primitive::Ellipse {
            center,
            radii: extent,
            angle,
        }
    };

    // Rounded rectangle along the dominant edge direction
    let rect = {
        let angle = snap_angle(edge_direction(all));
        let local: Vec<Vec2> = all
            .iter()
            .map(|&p| Vec2::from_angle(-angle).rotate(p - center))
            .collect();
        let (lo, hi) = local
            .iter()
            .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &q| {
                (lo.min(q), hi.max(q))
            });
        let mid = center + Vec2::from_angle(angle).rotate((lo + hi) / 2.);
        Primitive::RoundedRect {
            center: mid,
            half: (hi - lo) / 2.,
            radius: 0.1 * (hi - lo).min_element(),
            angle,
        }
    };

    let mut fitted: Vec<Primitive> = [circle, Some(ellipse), Some(rect)]
        .into_iter()
        .flatten()
        .map(|p| optimize(p, samples, scale))
        .collect();
    for p in &mut fitted {
        // Snap the optimized angles and nearly equal sides, which an artist would make exact
        match p {
            Primitive::Ellipse { radii, angle, .. } => {
                *angle = snap_angle(*angle);
                if (radii.x - radii.y).abs() < 0.03 * radii.max_element() {
                    *radii = Vec2::splat((radii.x + radii.y) / 2.);
                }
            }
            Primitive::RoundedRect {
                half,
                radius,
                angle,
                ..
            } => {
                *angle = snap_angle(*angle);
                if (half.x - half.y).abs() < 0.03 * half.max_element() {
                    *half = Vec2::splat((half.x + half.y) / 2.);
                }
                // Ends rounded all the way make a capsule
                if *radius > 0.85 * half.min_element() {
                    *radius = half.min_element();
                }
            }
            Primitive::Circle { .. } => {}
        }
    }
    // The closest fit, then the fewest segments
    fitted.sort_by(|a, b| {
        mean_distance(a, samples)
            .total_cmp(&mean_distance(b, samples))
            .then(a.segments().cmp(&b.segments()))
    });
    fitted
        .into_iter()
        .find(|p| mean_distance(p, samples) <= tolerance)
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    if d.abs() < 1e-9 {
        return None;
    }
    let column = |i: usize| {
        let mut m = a;
        for r in 0..3 {
            m[r][i] = b[r];
        }
        det(m) / d
    };
    Some([column(0), column(1), column(2)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::fit::eval;

    /// Every primitive's cubics must form a closed loop lying on its own distance function.
    #[test]
    fn cubics_follow_the_primitive() {
        let primitives = [
            Primitive::Circle {
                center: Vec2::new(10., 12.),
                radius: 7.,
            },
            Primitive::Ellipse {
                center: Vec2::new(30., 20.),
                radii: Vec2::new(12., 5.),
                angle: 0.4,
            },
            Primitive::RoundedRect {
                center: Vec2::new(40., 40.),
                half: Vec2::new(15., 9.),
                radius: 4.,
                angle: -0.3,
            },
            Primitive::RoundedRect {
                center: Vec2::new(40., 40.),
                half: Vec2::new(15., 9.),
                radius: 0.,
                angle: 0.2,
            },
        ];
        for primitive in primitives {
            let cubics = primitive.cubics();
            assert_eq!(cubics.len(), primitive.segments(), "{primitive:?}");
            for (a, b) in cubics.iter().zip(cubics.iter().cycle().skip(1)) {
                assert!(a[3].distance(b[0]) < 1e-3, "{primitive:?} is not closed");
            }
            for c in &cubics {
                for i in 0..=8 {
                    let d = primitive.distance(eval(c, i as f32 / 8.)).abs();
                    // Cubic arcs deviate from true circles by about 0.03% of the radius
                    assert!(d < 0.05, "{primitive:?}: distance {d}");
                }
            }
        }
    }
}

/// The circle, ellipse or (rounded) rectangle that matches `visible`, the samples of an outline
/// that show, within `tolerance` pixels on average, sized from `all` its samples.
pub fn fit_primitive(visible: &[Vec2], all: &[Vec2], tolerance: f32) -> Option<Primitive> {
    fit(visible, all, tolerance)
}
