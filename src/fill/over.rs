//! A gradient with alpha fitted over what lies below it: the composite `below (1 - A(t)) + P(t)`
//! is linear in each knot's alpha and premultiplied colour, so for fixed geometry and stops it
//! is one least squares solve, then held to `0 <= a <= 1` and to colours that can be painted.

use crate::render::{Gradient, Stops};
use glam::{Vec2, Vec3};

use super::{RIDGE, ramp};
use crate::scene::Fill;

/// A pixel a fill is fitted on: its centre, the colour wanted, and what shows below the fill.
#[derive(Clone, Copy, Debug)]
pub struct Over {
    pub p: Vec2,
    pub target: Vec3,
    pub below: Vec3,
}

/// The two knots `t` lies between on a ramp through `offsets`, and its weights on them.
fn hat(offsets: &[f32], t: f32) -> (usize, f32, f32) {
    let mut k = 0;
    while k + 2 < offsets.len() && offsets[k + 1] <= t {
        k += 1;
    }
    let w = offsets[k + 1] - offsets[k];
    let u = if w > 1e-9 {
        ((t - offsets[k]) / w).clamp(0., 1.)
    } else {
        1.
    };
    (k, 1. - u, u)
}

/// Solves the symmetric positive definite `m x = b` (row-major, `n` by `n`) by Cholesky, None
/// where it is singular.
fn cholesky(mut m: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for j in 0..n {
        let mut d = m[j * n + j];
        for k in 0..j {
            d -= m[j * n + k] * m[j * n + k];
        }
        if d <= 1e-15 {
            return None;
        }
        let d = d.sqrt();
        m[j * n + j] = d;
        for i in j + 1..n {
            let mut s = m[i * n + j];
            for k in 0..j {
                s -= m[i * n + k] * m[j * n + k];
            }
            m[i * n + j] = s / d;
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= m[i * n + k] * b[k];
        }
        b[i] = s / m[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= m[k * n + i] * b[k];
        }
        b[i] = s / m[i * n + i];
    }
    Some(b)
}

/// The knots' alphas and premultiplied colours (`x[4k]` the alpha, `x[4k + 1..4k + 4]` the
/// colour) that fit `samples` least, the last knot held transparent when `clear_end`; the ridge
/// pulls each knot toward `prior` (opaque, its colour), deciding what the samples leave open.
fn solve(
    samples: &[Over],
    gradient: &Gradient,
    offsets: &[f32],
    prior: &[Vec3],
    clear_end: bool,
    alphas: Option<&[f64]>,
) -> Option<Vec<f64>> {
    let knots = offsets.len() - usize::from(clear_end);
    // With the alphas given only the colours are free, one channel at a time
    let per = if alphas.is_some() { 1 } else { 4 };
    let n = knots * per;
    let ridge = RIDGE * samples.len() as f64 / knots as f64;
    let mut out = vec![0.; knots * 4];
    for channel in 0..if alphas.is_some() { 3 } else { 1 } {
        let (mut m, mut b) = (vec![0f64; n * n], vec![0f64; n]);
        for k in 0..knots {
            for j in 0..per {
                m[(k * per + j) * (n + 1)] += ridge;
            }
            let p = prior[k].as_dvec3();
            match alphas {
                Some(_) => b[k] += ridge * p[channel],
                None => {
                    b[4 * k] += ridge;
                    for c in 0..3 {
                        b[4 * k + 1 + c] += ridge * p[c];
                    }
                }
            }
        }
        for s in samples {
            let (k, w0, w1) = hat(offsets, gradient.t(s.p));
            let (target, below) = (s.target.as_dvec3(), s.below.as_dvec3());
            let ws = [(k, w0 as f64), (k + 1, w1 as f64)];
            match alphas {
                // target - below (1 - A) = P, A known
                Some(a) => {
                    let alpha: f64 = ws.iter().map(|&(j, w)| w * a[j]).sum();
                    let y = target[channel] - below[channel] * (1. - alpha);
                    for &(i, wi) in ws.iter().filter(|w| w.0 < knots) {
                        b[i] += wi * y;
                        for &(j, wj) in ws.iter().filter(|w| w.0 < knots) {
                            m[i * n + j] += wi * wj;
                        }
                    }
                }
                // target - below = sum w (P - a below), per channel
                None => {
                    for c in 0..3 {
                        let y = target[c] - below[c];
                        let row = |j: usize, w: f64| [(4 * j, -w * below[c]), (4 * j + 1 + c, w)];
                        let entries: Vec<(usize, f64)> = ws
                            .iter()
                            .filter(|w| w.0 < knots)
                            .flat_map(|&(j, w)| row(j, w))
                            .collect();
                        for &(i, vi) in &entries {
                            b[i] += vi * y;
                            for &(j, vj) in &entries {
                                m[i * n + j] += vi * vj;
                            }
                        }
                    }
                }
            }
        }
        let x = cholesky(m, b, n)?;
        for k in 0..knots {
            match alphas {
                Some(a) => {
                    out[4 * k] = a[k];
                    out[4 * k + 1 + channel] = x[k];
                }
                None => out[4 * k..4 * k + 4].copy_from_slice(&x[4 * k..4 * k + 4]),
            }
        }
    }
    Some(out)
}

/// `fill` with its gradient's ramp fitted with alpha over `samples`: the stops placed where an
/// opaque ramp through the samples' colours puts them (those paying `stop_cost`), each knot's
/// alpha and colour by least squares, alphas held to `0..=1` and colours to what can be
/// painted, the end transparent when `clear_end`. The start is opaque in the ramp, so a start
/// below the others' alpha becomes the fill's opacity. None for a solid fill or no fit.
pub fn with_alpha(samples: &[Over], fill: &Fill, stop_cost: f32, clear_end: bool) -> Option<Fill> {
    let gradient = fill.gradient?;
    let plain: Vec<(Vec2, Vec3)> = samples.iter().map(|s| (s.p, s.target)).collect();
    let (from, opaque) = ramp(&plain, &gradient, stop_cost)?;
    let offsets: Vec<f32> = std::iter::once(0.)
        .chain(opaque.stops().iter().map(|(o, _)| o))
        .chain(std::iter::once(1.))
        .collect();
    let prior: Vec<Vec3> = std::iter::once(from)
        .chain(
            opaque
                .stops()
                .iter()
                .map(|(_, c)| glam::Vec4::from(c).truncate()),
        )
        .chain(std::iter::once(Vec3::from(opaque.to_rgb())))
        .map(|c| c.clamp(Vec3::ZERO, Vec3::ONE))
        .collect();
    let free = solve(samples, &gradient, &offsets, &prior, clear_end, None)?;
    let knots = offsets.len();
    let mut alphas: Vec<f64> = (0..knots)
        .map(|k| free.get(4 * k).map_or(0., |a| a.clamp(0., 1.)))
        .collect();
    // The colours again with the alphas held, each within what its alpha can paint
    let held = solve(
        samples,
        &gradient,
        &offsets,
        &prior,
        clear_end,
        Some(&alphas),
    )?;
    let mut colours: Vec<Vec3> = (0..knots)
        .map(|k| {
            let a = alphas[k];
            match (k * 4 + 3 < held.len(), a > 0.) {
                (true, true) => {
                    let p = glam::DVec3::new(held[4 * k + 1], held[4 * k + 2], held[4 * k + 3]);
                    (p.clamp(glam::DVec3::ZERO, glam::DVec3::splat(a)) / a).as_vec3()
                }
                _ => prior[k],
            }
        })
        .collect();
    // A transparent knot paints nothing; it takes a neighbour's colour, as straight-alpha
    // renderers interpolate toward it
    for k in 0..knots {
        if alphas[k] == 0. {
            let near = (k > 0 && alphas[k - 1] > 0.)
                .then(|| colours[k - 1])
                .or_else(|| (k + 1 < knots && alphas[k + 1] > 0.).then(|| colours[k + 1]));
            colours[k] = near.unwrap_or(colours[k]);
        }
    }
    // The ramp starts opaque: the fill's opacity carries a start below the others
    let top = alphas.iter().copied().fold(0., f64::max);
    if top <= 0. {
        return None;
    }
    let opacity = alphas[0].max(1e-3);
    for a in &mut alphas {
        *a = (*a / opacity).min(1.);
    }
    let mut out = gradient;
    let rgba = |c: Vec3, a: f64| [c.x, c.y, c.z, a as f32];
    *out.to_mut() = rgba(colours[knots - 1], alphas[knots - 1]);
    let stops: Vec<(f32, [f32; 4])> = (1..knots - 1)
        .map(|k| (offsets[k], rgba(colours[k], alphas[k])))
        .collect();
    *out.stops_mut() = Stops::new(&stops);
    Some(Fill {
        color: colours[0],
        gradient: Some(out),
        opacity: (fill.opacity as f64 * opacity) as f32,
    })
}

/// The squared error of `fill` drawn over `samples`.
#[cfg(test)]
fn error_over(samples: &[Over], fill: &Fill) -> f32 {
    samples
        .iter()
        .map(|s| {
            let (c, a) = fill.paint(s.p);
            (s.below + a * (c - s.below) - s.target).length_squared()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glow(below: impl Fn(Vec2) -> Vec3) -> (Vec<Over>, Fill) {
        let (center, colour) = (Vec2::splat(32.), Vec3::new(0.95, 0.8, 0.3));
        let samples = (0..64 * 64)
            .map(|i| Vec2::new((i % 64) as f32 + 0.5, (i / 64) as f32 + 0.5))
            .map(|p| {
                let a = (1. - p.distance(center) / 28.).clamp(0., 1.);
                let b = below(p);
                Over {
                    p,
                    target: b + a * (colour - b),
                    below: b,
                }
            })
            .collect();
        let guess = Fill {
            color: colour,
            gradient: Some(Gradient::Radial {
                center,
                focus: Vec2::ZERO,
                radius: 28.,
                transform: glam::Mat2::IDENTITY,
                to: [0.2, 0.2, 0.2, 1.],
                stops: Stops::NONE,
            }),
            opacity: 1.,
        };
        (samples, guess)
    }

    /// A glow fading to transparent over a flat ground: the end comes out transparent in the
    /// glow's colour, and the composite matches.
    #[test]
    fn a_glow_over_a_flat_ground() {
        let (samples, guess) = glow(|_| Vec3::new(0.16, 0.13, 0.2));
        let fill = with_alpha(&samples, &guess, 1., true).expect("a fit");
        let g = fill.gradient.unwrap();
        assert_eq!(g.to()[3], 0.);
        assert!(
            Vec3::from(g.to_rgb()).distance(fill.color) < 0.05,
            "{fill:?}"
        );
        assert!(
            error_over(&samples, &fill) < 0.5,
            "{}",
            error_over(&samples, &fill)
        );
    }

    /// Over a ground that varies, the alpha is fixed by what shows through: half the glow's
    /// alpha at its centre is found as the fill's opacity.
    #[test]
    fn a_half_glow_over_stripes() {
        let (mut samples, guess) =
            glow(|p| Vec3::splat(if (p.x as i32 / 4) % 2 == 0 { 0.1 } else { 0.6 }));
        let colour = Vec3::new(0.95, 0.8, 0.3);
        for s in &mut samples {
            let a = 0.5 * (1. - s.p.distance(Vec2::splat(32.)) / 28.).clamp(0., 1.);
            s.target = s.below + a * (colour - s.below);
        }
        let fill = with_alpha(&samples, &guess, 1., false).expect("a fit");
        assert!((fill.opacity - 0.5).abs() < 0.05, "{fill:?}");
        assert!(
            error_over(&samples, &fill) < 0.5,
            "{}",
            error_over(&samples, &fill)
        );
    }
}
