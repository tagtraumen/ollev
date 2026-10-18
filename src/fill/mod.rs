//! Fitting a shape's fill to the pixels it shows: a solid color, or a linear or radial
//! gradient where that removes enough of the error to be worth one more thing to describe,
//! with stops between its ends where each removes enough more.

use crate::render::{Gradient, MAX_STOPS, Stops};
use glam::{DVec3, Vec2, Vec3};

use crate::scene::Fill;

mod over;
mod ramp;

pub use over::*;
pub use ramp::*;

/// What a gradient costs, in segments: an artist sets it once on the shape, like a couple of
/// points of its outline.
pub const GRADIENT_SEGMENTS: f64 = 2.;
/// What each stop between a gradient's ends costs, in segments: an offset and a color, set
/// with one click like one point of an outline, far cheaper than splitting the shape.
pub const STOP_SEGMENTS: f64 = 1.;
/// What moving a radial gradient's focal point off its center costs, in segments: one more
/// point to place; an ellipse's axes cost as much.
pub const FOCUS_SEGMENTS: f64 = 1.;
/// How many pixels from anything else a pixel has to lie for its color to be the shape's own
/// and not a mixture across the boundary, which a fill fitted to the rim would take for
/// shading toward it.
pub const RIM: usize = 3;
/// The share of a shape's pixels the fit keeps at least: a thin ring is mostly rim, and its
/// rim is all there is to go by.
pub const RIM_SHARE: f32 = 0.5;

/// What the parts of a fill cost, in the units of the error.
#[derive(Clone, Copy, Debug)]
pub struct Costs {
    /// A gradient rather than one color.
    pub gradient: f32,
    /// Each stop between a gradient's ends.
    pub stop: f32,
    /// A radial gradient's focus off its center.
    pub focus: f32,
    /// Each cubic of a path gradient's curve.
    pub segment: f32,
}

impl Costs {
    /// The costs at `segment_cost` per segment.
    pub fn at(segment_cost: f64) -> Self {
        Self {
            gradient: (GRADIENT_SEGMENTS * segment_cost) as f32,
            stop: (STOP_SEGMENTS * segment_cost) as f32,
            focus: (FOCUS_SEGMENTS * segment_cost) as f32,
            segment: segment_cost as f32,
        }
    }

    /// What `fill` costs on top of one color.
    pub fn of(&self, fill: &Fill) -> f32 {
        let Some(g) = &fill.gradient else {
            return 0.;
        };
        let focus = match g {
            Gradient::Radial {
                focus, transform, ..
            } => {
                let off = if *focus == Vec2::ZERO { 0. } else { self.focus };
                off + if *transform == glam::Mat2::IDENTITY { 0. } else { self.focus }
            }
            Gradient::Path { contour, .. } => self.segment * contour.len() as f32,
            Gradient::Linear { .. } => 0.,
        };
        self.gradient + self.stop * g.stops().len() as f32 + focus
    }
}

/// The segments `fill` costs on top of one color, see [`Costs`].
pub fn cost_segments(fill: &Fill) -> f64 {
    Costs::at(1.).of(fill) as f64
}

/// How deep a rim to leave out of a fit, given the shape's pixel count per depth (the last
/// index for those beyond [`RIM`]): as deep as [`RIM`] while [`RIM_SHARE`] of the pixels
/// remain, one pixel at least.
pub fn erosion(counts: &[usize]) -> usize {
    let total = counts.iter().sum::<usize>() as f32;
    (1..=RIM)
        .rev()
        .find(|&e| counts[e + 1..].iter().sum::<usize>() as f32 >= RIM_SHARE * total)
        .unwrap_or(1)
}

/// The fill for `samples` (pixel centers and colors) and its error over them: `solid` unless a
/// linear or radial gradient, with paying stops and focus, lowers the error (beyond `tolerance`)
/// by more than `costs`. A gradient spans `extent`, the whole shape's pixels (empty: samples).
pub fn fit(
    samples: &[(Vec2, Vec3)],
    tolerance: &[f32],
    solid: Vec3,
    center: Option<Vec2>,
    costs: Costs,
    extent: &[Vec2],
) -> (Fill, f32) {
    let n = samples.len() as f32;
    let beyond =
        |k: usize, d2: f32| crate::uncertainty::excess(d2, tolerance.get(k).copied().unwrap_or(0.));
    let flat = samples
        .iter()
        .enumerate()
        .map(|(k, (_, c))| beyond(k, (*c - solid).length_squared()))
        .sum::<f32>();
    let mut best = (Fill::solid(solid), flat, 0f32);
    if samples.len() < 4 {
        return (best.0, best.1);
    }
    let centroid = samples.iter().map(|(p, _)| *p).sum::<Vec2>() / n;
    let mean = samples.iter().map(|(_, c)| *c).sum::<Vec3>() / n;
    let positions: Vec<Vec2> = samples.iter().map(|(p, _)| *p).collect();
    let extent = if extent.is_empty() {
        &positions
    } else {
        extent
    };
    let concentric = radial_geometry(samples, center.unwrap_or(centroid))
        .and_then(|(c, limit)| radial(samples, c, limit, mean, extent));
    let mut candidates = vec![linear(samples, centroid, mean, extent), concentric];
    let mut k = 0;
    while k < candidates.len() {
        let Some(candidate) = candidates[k] else {
            k += 1;
            continue;
        };
        k += 1;
        let candidate = with_stops(samples, &candidate, costs.stop);
        // Measured as painted, since the colors are clamped to what can be painted: a white
        // ground with a dark rim fits a ramp past white, which clamped would gray it
        let sse = samples
            .iter()
            .enumerate()
            .map(|(k, (p, c))| beyond(k, (candidate.at(*p) - *c).length_squared()))
            .sum::<f32>();
        let cost = costs.of(&candidate);
        if sse + cost < best.1 + best.2 {
            best = (candidate, sse, cost);
            // A focus or an ellipse refines the concentric gradient only where that is the best
            // fill, as a shape lit off its center or an oval still fits a circle better than a line
            if k == 2
                && let Some(c) = concentric
            {
                candidates.push(focal(samples, &c, extent));
                candidates.push(elliptical(samples, &c));
            }
        }
    }
    (best.0, best.1)
}

/// The least squares radial gradient fitting `samples`, its centre searched from `start` and
/// its radius where the colours stop changing, spanning `extent`.
pub fn concentric(samples: &[(Vec2, Vec3)], start: Vec2, extent: &[Vec2]) -> Option<Fill> {
    if samples.len() < 4 {
        return None;
    }
    let mean = samples.iter().map(|(_, c)| *c).sum::<Vec3>() / samples.len() as f32;
    let (center, limit) = radial_geometry(samples, start)?;
    radial(samples, center, limit, mean, extent)
}

/// The path gradient along `contour` fitted to `samples` (on the plain error) and its error:
/// the ends at the 2nd and 98th percentile of the signed distance, the two colours by least
/// squares, then the stops that pay at `costs`; `None` where the distances hardly vary.
pub fn along(
    samples: &[(Vec2, Vec3)],
    contour: crate::render::distance::Contour,
    costs: Costs,
) -> Option<(Fill, f32)> {
    let mut d: Vec<f32> = samples.iter().map(|(p, _)| contour.distance(*p)).collect();
    let n = d.len();
    let mut sorted = d.clone();
    sorted.sort_by(f32::total_cmp);
    let (near, far) = (sorted[n / 50], sorted[(n * 49 / 50).min(n - 1)]);
    if far - near < 1. {
        return None;
    }
    // The colours at the ends: least squares of c = a (1 - t) + b t
    let (mut s00, mut s01, mut s11, mut c0, mut c1) = (0f32, 0f32, 0f32, Vec3::ZERO, Vec3::ZERO);
    for (t, (_, c)) in d.iter_mut().zip(samples) {
        *t = ((*t - near) / (far - near)).clamp(0., 1.);
        let u = 1. - *t;
        (s00, s01, s11) = (s00 + u * u, s01 + u * *t, s11 + *t * *t);
        (c0, c1) = (c0 + *c * u, c1 + *c * *t);
    }
    let det = s00 * s11 - s01 * s01;
    if det.abs() < 1e-6 {
        return None;
    }
    let a = (c0 * s11 - c1 * s01) / det;
    let b = (c1 * s00 - c0 * s01) / det;
    let two = Fill {
        color: a.clamp(Vec3::ZERO, Vec3::ONE),
        gradient: Some(Gradient::Path {
            contour,
            near,
            far,
            to: crate::render::opaque(b.clamp(Vec3::ZERO, Vec3::ONE).to_array()),
            stops: Stops::NONE,
        }),
        opacity: 1.,
    };
    let fill = with_stops(samples, &two, costs.stop);
    let sse = samples
        .iter()
        .map(|(p, c)| (fill.at(*p) - *c).length_squared())
        .sum();
    Some((fill, sse))
}

/// The least squares linear gradient: the affine color model over the plane, reduced to its
/// main direction, since a linear gradient varies along one axis only.
fn linear(samples: &[(Vec2, Vec3)], centroid: Vec2, mean: Vec3, extent: &[Vec2]) -> Option<Fill> {
    // Normal equations of c - mean = G (p - centroid): G = T S^-1 with S the position scatter
    // and T the color-position cross products
    let (mut sxx, mut sxy, mut syy) = (0f32, 0f32, 0f32);
    let (mut tx, mut ty) = (Vec3::ZERO, Vec3::ZERO);
    for (p, c) in samples {
        let d = *p - centroid;
        let e = *c - mean;
        sxx += d.x * d.x;
        sxy += d.x * d.y;
        syy += d.y * d.y;
        tx += e * d.x;
        ty += e * d.y;
    }
    let det = sxx * syy - sxy * sxy;
    if det <= 1e-6 * (sxx + syy).powi(2).max(1e-12) {
        return None;
    }
    let gx = (tx * syy - ty * sxy) / det;
    let gy = (ty * sxx - tx * sxy) / det;
    // The direction the colors change most along: the top eigenvector of G^T G
    let (a, b, c) = (gx.dot(gx), gx.dot(gy), gy.dot(gy));
    let axis = if b.abs() < 1e-12 && a >= c {
        Vec2::X
    } else if b.abs() < 1e-12 {
        Vec2::Y
    } else {
        let top = 0.5 * (a + c) + (0.25 * (a - c).powi(2) + b * b).sqrt();
        Vec2::new(b, top - a).normalize()
    };
    let slope = gx * axis.x + gy * axis.y;
    if slope.length_squared() < 1e-12 {
        return None;
    }
    let (mut t0, mut t1) = (f32::MAX, f32::MIN);
    for p in extent {
        let t = (*p - centroid).dot(axis);
        t0 = t0.min(t);
        t1 = t1.max(t);
    }
    let (from, to) = (mean + slope * t0, mean + slope * t1);
    Some(Fill {
        opacity: 1.,
        color: from.clamp(Vec3::ZERO, Vec3::ONE),
        gradient: Some(Gradient::Linear {
            start: centroid + axis * t0,
            end: centroid + axis * t1,
            to: crate::render::opaque(to.clamp(Vec3::ZERO, Vec3::ONE).to_array()),
            stops: Stops::NONE,
        }),
    })
}

/// At most this many samples take part in the search for a radial gradient's center.
const CENTER_SAMPLES: usize = 2000;

/// Shares of the farthest sample's distance tried for the radius while the center is searched.
const COARSE_LIMITS: [f32; 5] = [0.375, 0.5, 0.625, 0.75, 1.];

/// The center and radius a radial gradient fits `samples` best with: the center by a pattern
/// search from `start` scored at the best of a few coarse radii (a gradient often ends inside
/// the shape), then the radius refined and the center polished with it; `None` without samples.
fn radial_geometry(samples: &[(Vec2, Vec3)], start: Vec2) -> Option<(Vec2, f32)> {
    if samples.is_empty() {
        return None;
    }
    let stride = samples.len().div_ceil(CENTER_SAMPLES);
    let few = Few::new(samples.iter().step_by(stride).copied().collect());
    let farthest = |d: &[f32]| d.iter().copied().fold(0f32, f32::max);
    let coarse = |center: Vec2| {
        let d = few.distances(center);
        let far = farthest(&d);
        COARSE_LIMITS
            .iter()
            .map(|share| few.error(&d, far * share))
            .fold(f32::INFINITY, f32::min)
    };
    let center = radial_center(&few.samples, start, coarse)?;
    let d = few.distances(center);
    let far = farthest(&d);
    let error = |limit: f32| few.error(&d, limit);
    let (mut limit, mut best_error) = (far, error(far));
    for share in COARSE_LIMITS {
        let e = error(far * share);
        if e < best_error {
            (limit, best_error) = (far * share, e);
        }
    }
    let mut step = far / 16.;
    while step >= 0.25 {
        let mut moved = false;
        for candidate in [limit - step, limit + step] {
            if candidate > 0. && candidate <= far {
                let e = error(candidate);
                if e < best_error {
                    (limit, best_error, moved) = (candidate, e, true);
                }
            }
        }
        if !moved {
            step /= 2.;
        }
    }
    let center = radial_center(&few.samples, center, |c| {
        few.error(&few.distances(c), limit)
    })?;
    // A radius at the farthest sample is no end at all: the samples leave out the rim, and
    // the gradient goes on to the shape's edge
    if limit >= far - 0.5 {
        limit = f32::INFINITY;
    }
    Some((center, limit))
}

/// The center with the least `error`, by a pattern search from `start` in steps from a quarter
/// of the shape's extent down to a quarter pixel, within one extent of the shape (farther, a
/// radial only imitates a linear gradient). None without samples.
fn radial_center(few: &[(Vec2, Vec3)], start: Vec2, error: impl Fn(Vec2) -> f32) -> Option<Vec2> {
    if few.is_empty() {
        return None;
    }
    let (min, max) = few.iter().fold((Vec2::MAX, Vec2::MIN), |(lo, hi), (p, _)| {
        (lo.min(*p), hi.max(*p))
    });
    let size = max - min;
    let (lo, hi) = (min - size, max + size);
    let (mut center, mut best) = (start.clamp(lo, hi), error(start.clamp(lo, hi)));
    let mut step = size.max_element() / 4.;
    // Eight directions: along the axes alone the search stalls in a valley running diagonally,
    // as a quarter ring has toward its center
    let directions: Vec<Vec2> = (0..8)
        .map(|k| Vec2::from_angle(k as f32 * std::f32::consts::FRAC_PI_4))
        .collect();
    // The best of the eight neighbors at each step, not the first better one, which can lead
    // away from the light toward a far center
    while step >= 0.25 {
        let best_neighbor = directions
            .iter()
            .map(|d| center + *d * step)
            .filter(|c| *c == c.clamp(lo, hi))
            .map(|c| (c, error(c)))
            .min_by(|x, y| x.1.total_cmp(&y.1));
        match best_neighbor {
            Some((c, e)) if e < best => (center, best) = (c, e),
            _ => step /= 2.,
        }
    }
    Some(center)
}

/// The samples a radial gradient's geometry is searched on, with their mean colour and its
/// squared spread, which no center or radius changes.
struct Few {
    samples: Vec<(Vec2, Vec3)>,
    mean: Vec3,
    spread: f32,
}

impl Few {
    fn new(samples: Vec<(Vec2, Vec3)>) -> Self {
        let n = samples.len() as f32;
        let mean = samples
            .iter()
            .map(|(_, c)| *c)
            .fold(Vec3::ZERO, |a, c| a + c)
            / n;
        let spread = samples
            .iter()
            .map(|(_, c)| (*c - mean).length_squared())
            .fold(0f32, |a, v| a + v);
        Self {
            samples,
            mean,
            spread,
        }
    }

    /// Each sample's distance from `center`, which every radius tried there reads.
    fn distances(&self, center: Vec2) -> Vec<f32> {
        self.samples
            .iter()
            .map(|(p, _)| p.distance(center))
            .collect()
    }

    /// The squared error of the best radial gradient at the distances `d` ending at `limit`:
    /// the color linear in the distance up to the limit, constant beyond.
    fn error(&self, d: &[f32], limit: f32) -> f32 {
        let n = self.samples.len() as f32;
        let r_mean = d.iter().fold(0f32, |a, r| a + r.min(limit)) / n;
        let (mut srr, mut src) = (0f32, Vec3::ZERO);
        for ((_, c), r) in self.samples.iter().zip(d) {
            let (dr, dc) = (r.min(limit) - r_mean, *c - self.mean);
            srr += dr * dr;
            src += dc * dr;
        }
        if srr < 1e-6 {
            return self.spread;
        }
        // The residual of the regression of the colors on the radius
        (self.spread - src.length_squared() / srr).max(0.)
    }
}

/// The least squares radial gradient around `center` ending at `limit`: the color linear in
/// the distance from the center up to the limit, constant beyond.
fn radial(
    samples: &[(Vec2, Vec3)],
    center: Vec2,
    limit: f32,
    mean: Vec3,
    extent: &[Vec2],
) -> Option<Fill> {
    let radii: Vec<f32> = samples
        .iter()
        .map(|(p, _)| p.distance(center).min(limit))
        .collect();
    let n = samples.len() as f32;
    let r_mean = radii.iter().sum::<f32>() / n;
    let (mut srr, mut src) = (0f32, Vec3::ZERO);
    for ((_, c), r) in samples.iter().zip(&radii) {
        srr += (r - r_mean).powi(2);
        src += (*c - mean) * (r - r_mean);
    }
    if srr < 1e-6 {
        return None;
    }
    let slope = src / srr;
    if slope.length_squared() < 1e-12 {
        return None;
    }
    let at_center = mean - slope * r_mean;
    let radius = extent
        .iter()
        .map(|p| p.distance(center))
        .fold(0f32, f32::max)
        .min(limit);
    Some(Fill {
        opacity: 1.,
        color: at_center.clamp(Vec3::ZERO, Vec3::ONE),
        gradient: Some(Gradient::Radial {
            center,
            focus: Vec2::ZERO,
            radius,
            transform: glam::Mat2::IDENTITY,
            to: crate::render::opaque(
                (at_center + slope * radius)
                    .clamp(Vec3::ZERO, Vec3::ONE)
                    .to_array(),
            ),
            stops: Stops::NONE,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radial_center_without_samples_is_none() {
        assert_eq!(radial_center(&[], Vec2::ZERO, |_| 0.), None);
        assert_eq!(radial_geometry(&[], Vec2::ZERO), None);
        let grid: Vec<(Vec2, Vec3)> = (0..64)
            .map(|k| {
                let p = Vec2::new((k % 8) as f32, (k / 8) as f32);
                (p, Vec3::splat(p.distance(Vec2::splat(3.5)) / 5.))
            })
            .collect();
        let found = radial_center(&grid, Vec2::ZERO, |c| c.distance(Vec2::splat(3.5)));
        assert!(found.is_some_and(|c| c.distance(Vec2::splat(3.5)) < 0.5));
    }
}
