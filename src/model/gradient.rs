//! A gradient as a host draws it: where it runs ([`GradientShape`]) and one complete list of
//! stops ([`Gradient::stops`]). The tracer's own form is kept inside, for the model's render
//! and the SVG emitter, so both stay exact.

use crate::render::Gradient as Ramp;
use glam::{DVec2, Vec2, Vec3};

use super::{Affine, Curve};

/// A colour stop: its place along the gradient (0 at the start, 1 at the end) and the straight
/// (not premultiplied) sRGB-encoded RGBA colour there, each channel in 0..1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stop {
    /// The place along the gradient.
    pub offset: f64,
    /// The colour, RGBA.
    pub color: [f64; 4],
}

/// Where a gradient runs: the parameter `t` at each point of the image, in pixels. The colour
/// at a point is the stops' at `t` clamped to 0..1 (SVG's pad spread).
#[derive(Clone, Debug, PartialEq)]
pub enum GradientShape {
    /// `t` is 0 at `start`, 1 at `end`, and constant on the lines across the axis between.
    Linear {
        /// Where `t` is 0.
        start: DVec2,
        /// Where `t` is 1.
        end: DVec2,
    },
    /// `transform` maps the unit circle onto the circle where `t` is 1. `focus` is where `t` is
    /// 0, in the unit circle's plane and inside it (the origin for a centred gradient); between,
    /// `t` is that of the circle through the point centred `t` of the way from the focus to the
    /// origin, of radius `t` (SVG's `fx` and `fy` under a `gradientTransform`).
    Radial {
        /// The map from the unit circle's plane into the image.
        transform: Affine,
        /// Where `t` is 0, in the unit circle's plane.
        focus: DVec2,
    },
    /// `t` is 0 at signed distance `near` from `curve` and 1 at `far`, in pixels. A closed curve
    /// is negative inside (even-odd). An open one is measured on with its end tangents as rays,
    /// and is negative inside the region it bounds with `closure`: the points from where its end
    /// ray leaves the gradient's box, round the box's corners, to where its start ray leaves it.
    Path {
        /// The cubics distances are measured from, each starting where the one before ends.
        curve: Vec<Curve>,
        /// Whether the curve closes on itself.
        closed: bool,
        /// An open curve's closing points, none for a closed one.
        closure: Vec<DVec2>,
        /// The signed distance where `t` is 0.
        near: f64,
        /// The signed distance where `t` is 1.
        far: f64,
    },
}

/// A gradient paint. Hosts read it by [`Gradient::shape`] and [`Gradient::stops`].
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    from: Vec3,
    ramp: Ramp,
}

impl Gradient {
    /// The tracer's gradient `ramp` starting from the opaque colour `from`.
    pub(crate) fn new(from: Vec3, ramp: Ramp) -> Self {
        Self { from, ramp }
    }

    /// The tracer's own form: the start colour and the ramp, as the model renders it.
    pub(crate) fn ramp(&self) -> (Vec3, &Ramp) {
        (self.from, &self.ramp)
    }

    /// Where the gradient runs.
    #[must_use]
    pub fn shape(&self) -> GradientShape {
        let d = |p: Vec2| p.as_dvec2();
        match self.ramp {
            Ramp::Linear { start, end, .. } => GradientShape::Linear {
                start: d(start),
                end: d(end),
            },
            Ramp::Radial {
                center,
                focus,
                radius,
                transform,
                ..
            } => {
                // The radius the tracer measures by, as `focal_t` floors it
                let radius = radius.max(1e-3);
                let m = transform
                    .to_cols_array()
                    .map(|v| f64::from(v) * f64::from(radius));
                GradientShape::Radial {
                    transform: [m[0], m[1], m[2], m[3], center.x.into(), center.y.into()],
                    focus: d(focus.clamp_length_max(0.999 * radius) / radius),
                }
            }
            Ramp::Path {
                contour, near, far, ..
            } => GradientShape::Path {
                curve: contour.cubics().map(|c| c.map(d)).collect(),
                closed: contour.is_closed(),
                closure: contour.closure().iter().map(|&p| d(p)).collect(),
                near: near.into(),
                far: far.into(),
            },
        }
    }

    /// The stops from the start (offset 0) to the end (offset 1), for a renderer that
    /// interpolates straight colour between neighbours (SVG, Graphite). Where the ramp's alpha
    /// changes it is split so that this matches the model within half an 8-bit level.
    #[must_use]
    pub fn stops(&self) -> Vec<Stop> {
        self.ramp
            .svg_stops(self.from)
            .into_iter()
            .map(|(offset, c, a)| Stop {
                offset: offset.into(),
                color: [c.x.into(), c.y.into(), c.z.into(), a.into()],
            })
            .collect()
    }

    /// Whether every stop is opaque.
    #[must_use]
    pub fn is_opaque(&self) -> bool {
        self.ramp.is_opaque()
    }

    /// The gradient's parameter at `p`, clamped to 0..1.
    #[must_use]
    pub fn t(&self, p: DVec2) -> f64 {
        self.ramp.t(p.as_vec2()).into()
    }

    /// The straight RGBA colour the model paints at parameter `t`.
    #[must_use]
    pub fn color(&self, t: f64) -> [f64; 4] {
        let (c, a) = self.ramp.rgba(self.from, t as f32);
        [c.x.into(), c.y.into(), c.z.into(), a.into()]
    }

    /// The gradient under the similarity `m` (a turn, a uniform scale and a move).
    #[must_use]
    pub fn mapped(&self, m: &Affine) -> Self {
        let k = (m[0] * m[3] - m[1] * m[2]).abs().sqrt() as f32;
        let linear = glam::Mat2::from_cols_array(&[m[0], m[1], m[2], m[3]].map(|v| v as f32));
        let move_by = Vec2::new(m[4] as f32, m[5] as f32);
        let f = |p: Vec2| linear * p + move_by;
        let mut ramp = self.ramp;
        match &mut ramp {
            Ramp::Linear { start, end, .. } => (*start, *end) = (f(*start), f(*end)),
            Ramp::Radial {
                center,
                focus,
                radius,
                transform,
                ..
            } => {
                // The plane keeps its unit circle: the map's turn joins its transform
                (*center, *focus, *radius) = (f(*center), *focus * k, *radius * k);
                *transform = linear * *transform / k.max(f32::MIN_POSITIVE);
            }
            Ramp::Path {
                contour, near, far, ..
            } => {
                contour.map(f);
                (*near, *far) = (*near * k, *far * k);
            }
        }
        Self { ramp, ..*self }
    }
}

#[cfg(test)]
mod tests;
