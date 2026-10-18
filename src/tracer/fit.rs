use glam::{IVec2, Vec2};

/// A cubic Bézier as `[start, handle, handle, end]`.
pub type Cubic = [Vec2; 4];

#[derive(Clone, Copy, Debug)]
pub struct FitParams {
    /// Turning angle in radians above which a point on a boundary is kept as a sharp corner.
    pub corner_angle: f32,
    /// Max distance in pixels between the fitted curve and the smoothed boundary.
    pub tolerance: f32,
    /// Points before and after a boundary point this far along it define its turning angle. The
    /// window must see past single pixel steps (which turn by 90 degrees); larger windows only
    /// find corners at a coarser scale.
    pub corner_window: usize,
    /// Whether a sharper point within the window suppresses a corner only when the angle stays
    /// above the corner angle between them, so two turns within one window stay two corners.
    /// Off, any sharper point in the window wins.
    pub separate_turns: bool,
}

/// Fits a chain of pixel corners with cubics, keeping its endpoints fixed and its sharp corners
/// sharp; also the positions of the sharp corners it kept (not the end points, fixed anyway).
pub fn fit_chain_corners(
    points: &[IVec2],
    closed: bool,
    params: &FitParams,
) -> (Vec<Cubic>, Vec<Vec2>) {
    let points: Vec<Vec2> = points.iter().map(|p| p.as_vec2()).collect();
    fit_points_corners(&points, closed, params)
}

/// [`fit_chain_corners`] for a chain of points off the pixel grid.
pub fn fit_points_corners(
    points: &[Vec2],
    closed: bool,
    params: &FitParams,
) -> (Vec<Cubic>, Vec<Vec2>) {
    let chain = points;
    let mut points: Vec<Vec2> = points.to_vec();
    if closed {
        points.pop();
    }
    let n = points.len();
    let at = |i: isize| points[i.rem_euclid(n as isize) as usize];

    let angle = |i: usize, k: usize| {
        let a = at(i as isize) - at(i as isize - k as isize);
        let b = at(i as isize + k as isize) - at(i as isize);
        crate::tracer::angle::turn(a, b).map(f32::abs)
    };
    let window = |i: usize| {
        if closed {
            params.corner_window
        } else {
            params.corner_window.min(i).min(n - 1 - i)
        }
    };

    // None where a point repeats the one a window away (a step without a direction), as on a
    // closed loop of no more points than the window: no corners at the window's scale
    let angles: Vec<Option<f32>> = (0..n)
        .map(|i| {
            if window(i) == 0 {
                Some(0.)
            } else {
                angle(i, window(i))
            }
        })
        .collect();
    let neighbor = |j: isize| {
        if closed {
            Some(angles[j.rem_euclid(n as isize) as usize])
        } else {
            (j >= 0 && j < n as isize).then(|| angles[j as usize])
        }
    };
    // Only the sharpest point of a turn counts, so one corner is not split into several. Ties go
    // to the first point of a plateau.
    let outdone = |i: usize, here: f32, direction: isize, ties_win: bool| {
        for d in 1..=params.corner_window as isize {
            let Some(b) = neighbor(i as isize + direction * d) else {
                break;
            };
            // a point without a turn neither ends the scan nor outdoes this one
            let Some(b) = b else {
                continue;
            };
            if params.separate_turns && b <= params.corner_angle {
                break;
            }
            if b > here || (ties_win && b == here) {
                return true;
            }
        }
        false
    };
    let corners: Vec<usize> = (0..n)
        .filter(|&i| {
            angles[i].is_some_and(|a| {
                a > params.corner_angle && !outdone(i, a, -1, true) && !outdone(i, a, 1, false)
            })
        })
        .collect();
    let positions: Vec<Vec2> = corners.iter().map(|&c| points[c]).collect();
    (fit_at_corners(chain, &corners, closed, params), positions)
}

/// Fits a chain of points with cubics, sharp at `corners` (ascending indices into `points`, none
/// at the ends of an open chain) and smooth elsewhere. The last point of a closed chain repeats
/// its first.
pub fn fit_at_corners(
    points: &[Vec2],
    corners: &[usize],
    closed: bool,
    params: &FitParams,
) -> Vec<Cubic> {
    let mut points: Vec<Vec2> = points.to_vec();
    if closed {
        points.pop();
    }
    let n = points.len();
    let at = |i: isize| points[i.rem_euclid(n as isize) as usize];
    let mut corners = corners.to_vec();
    if !closed {
        corners.insert(0, 0);
        corners.push(n - 1);
        corners.dedup();
    }

    if corners.is_empty() {
        // A smooth closed loop: one piece with matching tangents at its seam, taken over a few
        // points either way, since the neighbors alone give a pixel step's direction
        let ring = smooth(&[&points[..], &points[..1]].concat(), true);
        let reach = 3.min(ring.len() / 2 - 1).max(1);
        let seam = (ring[reach] - ring[ring.len() - 1 - reach]).normalize_or_zero();
        return fit_cubics(&ring, seam, -seam, params.tolerance);
    }

    let mut cubics = Vec::new();
    let pieces = if closed {
        corners.len()
    } else {
        corners.len() - 1
    };
    for c in 0..pieces {
        let (start, end) = (corners[c], corners[(c + 1) % corners.len()]);
        let len = if end > start {
            end - start
        } else {
            end + n - start
        };
        let piece: Vec<Vec2> = (0..=len).map(|i| at((start + i) as isize)).collect();
        let piece = smooth(&piece, false);
        let m = piece.len();
        let reach = 3.min(m - 1);
        let t0 = (piece[reach] - piece[0]).normalize_or_zero();
        let t1 = (piece[m - 1 - reach] - piece[m - 1]).normalize_or_zero();
        cubics.extend(fit_cubics(&piece, t0, t1, params.tolerance));
    }
    cubics
}

/// Smooths away the pixel staircase with a few passes of a [1, 2, 1] filter. The endpoints stay
/// put unless `wrap` is set, in which case the first and last point are the same point.
fn smooth(points: &[Vec2], wrap: bool) -> Vec<Vec2> {
    let mut p = points.to_vec();
    let n = p.len();
    if n < 3 {
        return p;
    }
    for _ in 0..3 {
        let prev = p.clone();
        for i in 1..n - 1 {
            p[i] = (prev[i - 1] + 2. * prev[i] + prev[i + 1]) * 0.25;
        }
        if wrap {
            p[0] = (prev[n - 2] + 2. * prev[0] + prev[1]) * 0.25;
            p[n - 1] = p[0];
        }
    }
    p
}

/// Schneider's algorithm: least squares fit with given end tangents, refining the parametrization
/// with Newton steps and splitting at the worst point when the error stays above the tolerance.
pub(crate) fn fit_cubics(points: &[Vec2], t0: Vec2, t1: Vec2, tolerance: f32) -> Vec<Cubic> {
    let n = points.len();
    let (first, last) = (points[0], points[n - 1]);
    if n == 2 {
        let d = first.distance(last) / 3.;
        return vec![[first, first + t0 * d, last + t1 * d, last]];
    }

    let mut u = chord_lengths(points);
    let mut cubic = generate(points, &u, t0, t1);
    let (mut error, mut split) = max_error(points, &cubic, &u);
    if error < tolerance * tolerance {
        return vec![cubic];
    }
    if error < 4. * tolerance * tolerance {
        for _ in 0..4 {
            u = u
                .iter()
                .zip(points)
                .map(|(&t, &p)| newton(&cubic, p, t))
                .collect();
            cubic = generate(points, &u, t0, t1);
            (error, split) = max_error(points, &cubic, &u);
            if error < tolerance * tolerance {
                return vec![cubic];
            }
        }
    }

    let split = split.clamp(1, n - 2);
    let center = (points[split - 1] - points[split + 1]).normalize_or_zero();
    let mut left = fit_cubics(&points[..=split], t0, center, tolerance);
    left.extend(fit_cubics(&points[split..], -center, t1, tolerance));
    left
}

fn chord_lengths(points: &[Vec2]) -> Vec<f32> {
    let mut u = vec![0.];
    for w in points.windows(2) {
        u.push(u.last().unwrap() + w[0].distance(w[1]));
    }
    let total = u.last().copied().unwrap_or(1.).max(1e-6);
    u.iter().map(|x| x / total).collect()
}

fn bernstein(t: f32) -> [f32; 4] {
    let mt = 1. - t;
    [mt * mt * mt, 3. * mt * mt * t, 3. * mt * t * t, t * t * t]
}

/// Moves the handles of a cubic along its chord, as little as possible, so that their projections
/// onto the chord lie in order between the endpoints. The curve then advances along the chord the
/// whole way and cannot loop or cross itself, which in traced outlines is noise rather than shape.
pub fn untangle(c: Cubic) -> Cubic {
    let chord = c[3] - c[0];
    let length_squared = chord.length_squared();
    if length_squared < 1e-6 {
        return c;
    }
    let along = |p: Vec2| (p - c[0]).dot(chord) / length_squared;
    let (a, b) = (along(c[1]), along(c[2]));
    let (mut a2, mut b2) = (a.clamp(0., 1.), b.clamp(0., 1.));
    if a2 > b2 {
        let middle = 0.5 * (a2 + b2);
        (a2, b2) = (middle, middle);
    }
    [c[0], c[1] + (a2 - a) * chord, c[2] + (b2 - b) * chord, c[3]]
}

pub fn eval(c: &Cubic, t: f32) -> Vec2 {
    let b = bernstein(t);
    c[0] * b[0] + c[1] * b[1] + c[2] * b[2] + c[3] * b[3]
}

/// Solves for the handle lengths along the fixed end tangents that minimize the squared error.
fn generate(points: &[Vec2], u: &[f32], t0: Vec2, t1: Vec2) -> Cubic {
    let (first, last) = (points[0], points[points.len() - 1]);
    let (mut c00, mut c01, mut c11, mut x0, mut x1) = (0., 0., 0., 0., 0.);
    for (&p, &t) in points.iter().zip(u) {
        let b = bernstein(t);
        let (a0, a1) = (t0 * b[1], t1 * b[2]);
        c00 += a0.dot(a0);
        c01 += a0.dot(a1);
        c11 += a1.dot(a1);
        let rest = p - (first * (b[0] + b[1]) + last * (b[2] + b[3]));
        x0 += a0.dot(rest);
        x1 += a1.dot(rest);
    }
    let det = c00 * c11 - c01 * c01;
    let (alpha0, alpha1) = if det.abs() > 1e-12 {
        ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det)
    } else {
        (0., 0.)
    };

    // Degenerate or backwards handles fall back to a third of the chord
    let chord = first.distance(last);
    let eps = 1e-6 * chord;
    if alpha0 < eps || alpha1 < eps {
        let d = chord / 3.;
        return [first, first + t0 * d, last + t1 * d, last];
    }
    [first, first + t0 * alpha0, last + t1 * alpha1, last]
}

/// The least-squares cubic through `points` leaving the first along the unit tangent `t0` and
/// arriving at the last against `t1`, parameters refined by `iterations` Newton steps: the best
/// such cubic, where [`fit_cubics`] stops at the first within its tolerance.
pub fn fit_cubic(points: &[Vec2], t0: Vec2, t1: Vec2, iterations: usize) -> Cubic {
    let n = points.len();
    let (first, last) = (points[0], points[n - 1]);
    if n < 3 {
        let d = first.distance(last) / 3.;
        return [first, first + t0 * d, last + t1 * d, last];
    }
    let sum_of_squares = |cubic: &Cubic, u: &[f32]| -> f32 {
        points
            .iter()
            .zip(u)
            .map(|(&p, &t)| eval(cubic, t).distance_squared(p))
            .sum()
    };
    let mut u = chord_lengths(points);
    let mut cubic = generate(points, &u, t0, t1);
    let mut best = (sum_of_squares(&cubic, &u), cubic);
    for _ in 0..iterations {
        u = u
            .iter()
            .zip(points)
            .map(|(&t, &p)| newton(&cubic, p, t).clamp(0., 1.))
            .collect();
        cubic = generate(points, &u, t0, t1);
        let e = sum_of_squares(&cubic, &u);
        if e < best.0 {
            best = (e, cubic);
        }
    }
    best.1
}

fn max_error(points: &[Vec2], cubic: &Cubic, u: &[f32]) -> (f32, usize) {
    let mut worst = (0., points.len() / 2);
    for (i, (&p, &t)) in points
        .iter()
        .zip(u)
        .enumerate()
        .skip(1)
        .take(points.len().saturating_sub(2))
    {
        let d = eval(cubic, t).distance_squared(p);
        if d >= worst.0 {
            worst = (d, i);
        }
    }
    worst
}

/// One Newton step towards the parameter of the curve point closest to `p`.
fn newton(c: &Cubic, p: Vec2, t: f32) -> f32 {
    let d1 = [3. * (c[1] - c[0]), 3. * (c[2] - c[1]), 3. * (c[3] - c[2])];
    let d2 = [2. * (d1[1] - d1[0]), 2. * (d1[2] - d1[1])];
    let mt = 1. - t;
    let q = eval(c, t);
    let q1 = d1[0] * mt * mt + d1[1] * 2. * mt * t + d1[2] * t * t;
    let q2 = d2[0] * mt + d2[1] * t;
    let numerator = (q - p).dot(q1);
    let denominator = q1.dot(q1) + (q - p).dot(q2);
    if denominator.abs() < 1e-9 {
        t
    } else {
        (t - numerator / denominator).clamp(0., 1.)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::beautify::beautify_chain;

    const PARAMS: FitParams = FitParams {
        corner_angle: 1.,
        tolerance: 0.5,
        corner_window: 2,
        separate_turns: true,
    };

    /// Repeated points (a step of zero length, which has no direction) and a chain standing
    /// still fit without a panic under glam's asserts.
    #[test]
    fn repeated_points_fit() {
        let square = [
            (0, 0),
            (0, 0),
            (4, 0),
            (4, 0),
            (4, 4),
            (0, 4),
            (0, 4),
            (0, 0),
        ];
        let square: Vec<IVec2> = square.iter().map(|&(x, y)| IVec2::new(x, y)).collect();
        for closed in [true, false] {
            let (cubics, _) = fit_chain_corners(&square, closed, &PARAMS);
            assert!(cubics.iter().flatten().all(|p| p.is_finite()));
        }
        for closed in [true, false] {
            fit_points_corners(&[Vec2::ONE; 5], closed, &PARAMS);
        }
    }

    /// A cubic of zero length in a chain (no tangent at either end) beautifies.
    #[test]
    fn a_zero_length_cubic_beautifies() {
        let (a, b) = (Vec2::ZERO, Vec2::new(6., 0.));
        let chain = [
            [a, a, b, b],
            [b; 4],
            [b, b, Vec2::new(6., 6.), Vec2::new(6., 6.)],
        ];
        let out = beautify_chain(&chain, 0.5, 1.);
        assert!(out.iter().flatten().all(|p| p.is_finite()));
    }
}
