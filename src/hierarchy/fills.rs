//! The gradient models a node can be drawn with: linear, radial and multi-stop ramps.

use super::*;

/// A color varying with the squared distance from a center: `color + slope * |p - center|^2`,
/// the shading of a radial gradient, which with a stop or two paints a profile like it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialFill {
    pub center: DVec2,
    pub color: DVec3,
    pub slope: DVec3,
}

impl RadialFill {
    pub fn at(&self, p: DVec2) -> DVec3 {
        self.color + self.slope * p.distance_squared(self.center)
    }
}

/// A color varying linearly over the plane: `color + slope * ((p - origin) . axis)`, which a
/// linear gradient along `axis` paints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearFill {
    pub origin: DVec2,
    pub axis: DVec2,
    pub color: DVec3,
    pub slope: DVec3,
}

impl LinearFill {
    pub fn at(&self, p: DVec2) -> DVec3 {
        self.color + self.slope * (p - self.origin).dot(self.axis)
    }
}

/// At most this many of a node's pixels go into fitting its ramp, a few per bin along a 256 px
/// axis: the plan only asks whether a node is one ramp; its error is taken over all pixels.
const RAMP_SAMPLES: usize = 1024;

/// A color varying along an axis through stops, which a linear gradient with stops paints, in
/// the colors of the statistics, with the squared error of each leaf of the node it was fitted
/// to around it.
#[derive(Debug)]
pub struct RampFill {
    pub from: Vec3,
    /// A linear gradient with at least one stop.
    pub gradient: Gradient,
    /// The first leaf rank of the node, and the summed errors of its leaves before each rank
    /// from there on.
    pub(super) lo: u32,
    pub(super) errors: Vec<f64>,
}

impl RampFill {
    pub fn at(&self, p: DVec2) -> DVec3 {
        let p = p.as_vec2();
        self.gradient
            .color(self.from, self.gradient.t(p))
            .as_dvec3()
    }

    /// The ramp along the axis of `linear` that fits the pixels of `node` best, each stop
    /// paying `stop_cost` in error (see [`crate::fill::ramp`]), reaching over the node's
    /// bounding box. None when no stop pays, where the linear fill is the node's gradient.
    pub fn fit(
        hierarchy: &Hierarchy,
        node: u32,
        linear: &LinearFill,
        stop_cost: f64,
    ) -> Option<Self> {
        let along = |p: DVec2| (p - linear.origin).dot(linear.axis);
        let corners = corners(hierarchy, node);
        let q0 = corners.iter().map(|&c| along(c)).fold(f64::MAX, f64::min);
        let q1 = corners.iter().map(|&c| along(c)).fold(f64::MIN, f64::max);
        if q1 - q0 < 1. {
            return None;
        }
        let start = linear.origin + linear.axis * q0;
        let end = linear.origin + linear.axis * q1;
        let pixels = hierarchy.pixels(node);
        let samples = ramp_samples(hierarchy, pixels);
        let two = Gradient::Linear {
            start: start.as_vec2(),
            end: end.as_vec2(),
            to: crate::render::opaque(linear.at(end).as_vec3().to_array()),
            stops: Stops::NONE,
        };
        let share = samples.len() as f64 / pixels.len() as f64;
        let (from, gradient) = crate::fill::ramp(&samples, &two, (stop_cost * share) as f32)?;
        if gradient.stops().is_empty() {
            return None;
        }
        Some(Self::tabled(hierarchy, node, from, gradient))
    }

    /// The radial gradient about the centre of `radial`, linear in the radius and stretched into
    /// the ellipse that fits the pixels of `node` best (see [`crate::fill::stretch`]). Two
    /// colours only: stops between them would fit the flat regions a leaf merges.
    pub fn fit_radial(
        hierarchy: &Hierarchy,
        node: u32,
        radial: &RadialFill,
        segment_cost: f64,
    ) -> Option<Self> {
        let center = radial.center.as_vec2();
        let radius = corners(hierarchy, node)
            .iter()
            .map(|c| c.as_vec2().distance(center))
            .fold(0f32, f32::max);
        let samples = ramp_samples(hierarchy, hierarchy.pixels(node));
        let (radius, angle, squash) = crate::fill::stretch(&samples, center, radius);
        let two = Gradient::Radial {
            center,
            focus: Vec2::ZERO,
            radius,
            transform: crate::fill::ellipse_transform(angle, squash),
            to: [0., 0., 0., 1.],
            stops: Stops::NONE,
        };
        let (from, gradient) = crate::fill::ramp(&samples, &two, f32::INFINITY)?;
        let ramp = Self::tabled(hierarchy, node, from, gradient);
        // Shading read on the leaves' interiors, clear of the rims that mix their neighbours'
        // colours, and no leaf the picture steps across within a unit's length
        let unit = crate::translucency::raster::unit_of(hierarchy.width()) as u32;
        let paint = crate::fill::cost_segments(&crate::Fill {
            color: from,
            gradient: Some(gradient),
            opacity: 1.,
        }) * segment_cost;
        let mut shading = 0.;
        for &leaf in hierarchy.leaves(node) {
            let (steps, gain) = ramp.interior(hierarchy, leaf, unit);
            if steps > unit {
                return None;
            }
            shading += gain;
        }
        (shading > paint).then_some(ramp)
    }

    /// Over the pixels of `leaf` clear of its rim: how many neighbouring pairs step beyond the
    /// ramp's step between them by more than the noise of two pixels (counted up to `most + 1`),
    /// and how much less squared error the ramp leaves there than their mean colour.
    fn interior(&self, hierarchy: &Hierarchy, leaf: u32, most: u32) -> (u32, f64) {
        let (w, n) = (hierarchy.width(), hierarchy.colors.len());
        let h = n / w;
        let limit = 2. * crate::translucency::NOISE as f64;
        let inside = |i: usize| {
            let (x, y) = (i % w, i / w);
            x > 0
                && y > 0
                && x + 1 < w
                && y + 1 < h
                && [i - 1, i + 1, i - w, i + w]
                    .iter()
                    .all(|&j| hierarchy.leaf(j) == leaf)
        };
        let at = |i: usize| self.at(DVec2::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5));
        let (mut steps, mut count, mut sum, mut squares, mut ramp) = (0, 0., DVec3::ZERO, 0., 0.);
        for &i in hierarchy.pixels(leaf) {
            let i = i as usize;
            if !inside(i) {
                continue;
            }
            let c = hierarchy.colors[i];
            (count, sum, squares) = (count + 1., sum + c, squares + c.length_squared());
            ramp += (c - at(i)).length_squared();
            for j in [i + 1, i + w] {
                if j < n && inside(j) {
                    let seen = c - hierarchy.colors[j];
                    if (seen - (at(i) - at(j))).length() > limit {
                        steps += 1;
                        if steps > most {
                            return (steps, 0.);
                        }
                    }
                }
            }
        }
        let flat = if count > 0. {
            squares - sum.length_squared() / count
        } else {
            0.
        };
        (steps, flat - ramp)
    }

    /// The ramp with the errors of the leaves of `node` around it tabled.
    fn tabled(hierarchy: &Hierarchy, node: u32, from: Vec3, gradient: Gradient) -> Self {
        let mut ramp = Self {
            from,
            gradient,
            lo: 0,
            errors: Vec::new(),
        };
        let (lo, hi) = hierarchy.leaf_ranks(node);
        let mut errors = Vec::with_capacity((hi - lo + 1) as usize);
        errors.push(0.);
        let mut total = 0.;
        let pieces = ramp.pieces();
        for &leaf in hierarchy.leaves(node) {
            total += ramp.leaf_error(hierarchy, leaf, &pieces);
            errors.push(total);
        }
        ramp.lo = lo;
        ramp.errors = errors;
        ramp
    }

    /// The pieces of the ramp as linear fills, each with the range of the parameter it holds
    /// over: constant before the start and after the end.
    fn pieces(&self) -> Vec<(f64, f64, LinearFill)> {
        let Gradient::Linear { start, end, .. } = self.gradient else {
            return Vec::new();
        };
        let (start, end) = (start.as_dvec2(), end.as_dvec2());
        let length = start.distance(end);
        let axis = (end - start) / length;
        let stops = self.gradient.stops();
        let mut knots = vec![(0., self.from.as_dvec3())];
        knots.extend(
            stops
                .iter()
                .map(|(o, c)| (o as f64, glam::Vec4::from(c).truncate().as_dvec3())),
        );
        knots.push((1., Vec3::from(self.gradient.to_rgb()).as_dvec3()));
        let flat = |color: DVec3| LinearFill {
            origin: start,
            axis,
            color,
            slope: DVec3::ZERO,
        };
        let mut pieces = vec![(f64::MIN, 0., flat(knots[0].1))];
        for w in knots.windows(2) {
            let ((a, ca), (b, cb)) = (w[0], w[1]);
            if b - a < 1e-9 {
                continue;
            }
            pieces.push((
                a,
                b,
                LinearFill {
                    origin: start + axis * a * length,
                    axis,
                    color: ca,
                    slope: (cb - ca) / ((b - a) * length),
                },
            ));
        }
        pieces.push((1., f64::MAX, flat(knots[knots.len() - 1].1)));
        pieces
    }

    /// The squared error of the pixels of `leaf` around the ramp: in closed form where the
    /// leaf lies within one piece of it, pixel by pixel where it crosses a stop.
    fn leaf_error(
        &self,
        hierarchy: &Hierarchy,
        leaf: u32,
        pieces: &[(f64, f64, LinearFill)],
    ) -> f64 {
        if let Gradient::Linear { start, end, .. } = self.gradient {
            let (start, end) = (start.as_dvec2(), end.as_dvec2());
            let d = end - start;
            let t = |p: DVec2| (p - start).dot(d) / d.length_squared();
            let ts = corners(hierarchy, leaf).map(t);
            let (t0, t1) = (
                ts.iter().copied().fold(f64::MAX, f64::min),
                ts.iter().copied().fold(f64::MIN, f64::max),
            );
            if let Some((_, _, fill)) = pieces.iter().find(|&&(a, b, _)| a <= t0 && t1 <= b) {
                return hierarchy.nodes[leaf as usize].stats.sse_about_linear(fill);
            }
        }
        let width = hierarchy.width();
        hierarchy
            .pixels(leaf)
            .iter()
            .map(|&i| {
                let i = i as usize;
                let p = DVec2::new((i % width) as f64 + 0.5, (i / width) as f64 + 0.5);
                (hierarchy.colors[i] - self.at(p)).length_squared()
            })
            .sum()
    }

    /// The squared error of the pixels of `node` around the ramp: from the table where the
    /// node lies within the one the ramp was fitted to, leaf by leaf otherwise.
    pub fn sse(&self, hierarchy: &Hierarchy, node: u32) -> f64 {
        let (lo, hi) = hierarchy.leaf_ranks(node);
        let top = self.lo + self.errors.len() as u32 - 1;
        if self.lo <= lo && hi <= top {
            return (self.errors[(hi - self.lo) as usize] - self.errors[(lo - self.lo) as usize])
                .max(0.);
        }
        let pieces = self.pieces();
        hierarchy
            .leaves(node)
            .iter()
            .map(|&leaf| self.leaf_error(hierarchy, leaf, &pieces))
            .sum()
    }
}

/// The pixel centres at the corners of the bounding box of `node`.
fn corners(hierarchy: &Hierarchy, node: u32) -> [DVec2; 4] {
    let n = &hierarchy.nodes[node as usize];
    let (min, max) = (n.min.as_dvec2() + 0.5, n.max.as_dvec2() - 0.5);
    [min, max, DVec2::new(min.x, max.y), DVec2::new(max.x, min.y)]
}

/// At most [`RAMP_SAMPLES`] of `pixels`, strided across the width's residues so the samples
/// of a row-major run do not line up in columns, as pixel centres and colours.
fn ramp_samples(hierarchy: &Hierarchy, pixels: &[u32]) -> Vec<(Vec2, Vec3)> {
    let width = hierarchy.width();
    let mut stride = pixels.len().div_ceil(RAMP_SAMPLES).max(1);
    while stride > 1 && gcd(stride, width) != 1 {
        stride += 1;
    }
    pixels
        .iter()
        .step_by(stride)
        .map(|&i| {
            let i = i as usize;
            let p = Vec2::new((i % width) as f32 + 0.5, (i / width) as f32 + 0.5);
            (p, hierarchy.colors[i].as_vec3())
        })
        .collect()
}

/// Solves the symmetric 4 x 4 system `m x = rhs` for three right sides at once, by Gaussian
/// elimination with partial pivoting; None when it is singular.
#[allow(clippy::needless_range_loop)]
pub(super) fn solve4(mut m: [[f64; 4]; 4], mut rhs: [DVec3; 4]) -> Option<[DVec3; 4]> {
    let scale = (0..4).map(|i| m[i][i].abs()).fold(0f64, f64::max);
    for col in 0..4 {
        let pivot = (col..4).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[pivot][col].abs() <= 1e-12 * scale.max(1e-300) {
            return None;
        }
        m.swap(col, pivot);
        rhs.swap(col, pivot);
        for row in col + 1..4 {
            let f = m[row][col] / m[col][col];
            for k in col..4 {
                m[row][k] -= f * m[col][k];
            }
            rhs[row] -= rhs[col] * f;
        }
    }
    let mut x = [DVec3::ZERO; 4];
    for row in (0..4).rev() {
        let mut acc = rhs[row];
        for k in row + 1..4 {
            acc -= x[k] * m[row][k];
        }
        x[row] = acc / m[row][row];
    }
    Some(x)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}
