use glam::{Mat2, Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

/// A closed cubic Bézier spline stored as `[anchor, out handle, in handle]` triples, so it has `3n`
/// points for `n` curves. The last curve ends at the first anchor.
#[derive(Clone, Debug, Default)]
pub struct Subpath {
    pub points: Vec<Vec2>,
}

#[derive(Clone, Debug)]
pub struct Path {
    pub subpaths: Vec<Subpath>,
    /// Straight (not premultiplied) RGBA. With a gradient this is the color at its start.
    pub color: [f32; 4],
    pub gradient: Option<Gradient>,
    pub fill_rule: FillRule,
    /// The standard deviation in pixels of the Gaussian blur over the path's coverage, 0 for
    /// none (see [`crate::render::blur`]).
    pub blur: f32,
}

/// The most stops a gradient holds between its two ends, with room for ramps a fit splits
/// finer than they were drawn.
pub const MAX_STOPS: usize = 10;

/// The stops between a gradient's ends: nondecreasing offsets in `0..=1` (two at one offset
/// a hard step), each with a straight RGBA colour. The ramp runs in premultiplied colour from
/// the opaque start colour at 0 through these to the end colour at 1.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stops {
    len: u8,
    offsets: [f32; MAX_STOPS],
    colors: [[f32; 4]; MAX_STOPS],
}

impl Stops {
    /// No stops between the ends: a two-stop gradient.
    pub const NONE: Stops = Stops {
        len: 0,
        offsets: [0.; MAX_STOPS],
        colors: [[0.; 4]; MAX_STOPS],
    };

    /// Opaque stops of the colours `stops`, see [`Stops::new`].
    pub fn opaque(stops: &[(f32, [f32; 3])]) -> Self {
        let rgba: Vec<(f32, [f32; 4])> = stops
            .iter()
            .map(|&(o, [r, g, b])| (o, [r, g, b, 1.]))
            .collect();
        Self::new(&rgba)
    }

    /// Whether every stop is opaque.
    pub fn is_opaque(&self) -> bool {
        self.iter().all(|(_, c)| c[3] == 1.)
    }

    /// The stops `stops`, which must be at most [`MAX_STOPS`]; offsets are clamped to `0..=1`
    /// and sorted.
    pub fn new(stops: &[(f32, [f32; 4])]) -> Self {
        assert!(stops.len() <= MAX_STOPS, "{} stops", stops.len());
        let mut sorted = stops.to_vec();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut out = Self::NONE;
        for (k, (offset, color)) in sorted.into_iter().enumerate() {
            out.offsets[k] = offset.clamp(0., 1.);
            out.colors[k] = color;
        }
        out.len = stops.len() as u8;
        out
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The offset and colour of stop `k`.
    pub fn get(&self, k: usize) -> (f32, [f32; 4]) {
        assert!(k < self.len());
        (self.offsets[k], self.colors[k])
    }

    pub fn iter(&self) -> impl Iterator<Item = (f32, [f32; 4])> + '_ {
        (0..self.len()).map(|k| self.get(k))
    }

    /// The colours of the stops, to change in place.
    pub fn colors_mut(&mut self) -> &mut [[f32; 4]] {
        let len = self.len();
        &mut self.colors[..len]
    }
}

/// Where a gradient's ramp is at some `t`: its straight colour and alpha, how much of it comes
/// from the start and the end knot (each nonzero only on the first or last piece of the ramp),
/// and how fast the straight colour, the premultiplied colour and the alpha change with `t`.
#[derive(Clone, Copy, Debug)]
pub struct RampPoint {
    pub color: Vec3,
    pub alpha: f32,
    pub from_weight: f32,
    pub to_weight: f32,
    #[cfg_attr(not(test), allow(dead_code))] // read by the tests' backward pass
    pub slope: Vec3,
    pub premultiplied_slope: Vec3,
    pub alpha_slope: f32,
}

/// A gradient from the path's color to `to`, through `stops` between, held constant beyond its
/// ends. A path gradient holds its curve inline, so every gradient stays `Copy`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gradient {
    /// The path's color at `start`, `to` at `end`.
    Linear {
        start: Vec2,
        end: Vec2,
        to: [f32; 4],
        stops: Stops,
    },
    /// The path's color at the focal point `center + focus` (SVG's `fx`, `fy`, inside the circle),
    /// `to` on the circle of `radius`; `transform` maps the gradient's plane, where the focus
    /// lies, onto the image about `center` (SVG's `gradientTransform`, an ellipse's axes).
    Radial {
        center: Vec2,
        focus: Vec2,
        radius: f32,
        transform: Mat2,
        to: [f32; 4],
        stops: Stops,
    },
    /// The path's color at the signed distance `near` from `contour` (negative inside it), `to`
    /// at `far`, in pixels: the shading along a curve, a radial gradient around a point and a
    /// linear one along a line its special cases.
    Path {
        contour: crate::render::distance::Contour,
        near: f32,
        far: f32,
        to: [f32; 4],
        stops: Stops,
    },
}

/// How far the derivative of a radial gradient with a focus steps its parameters, in pixels.
const FOCAL_STEP: f32 = 1e-2;

impl Gradient {
    pub fn to(&self) -> [f32; 4] {
        match self {
            Gradient::Linear { to, .. }
            | Gradient::Radial { to, .. }
            | Gradient::Path { to, .. } => *to,
        }
    }

    pub fn to_mut(&mut self) -> &mut [f32; 4] {
        match self {
            Gradient::Linear { to, .. }
            | Gradient::Radial { to, .. }
            | Gradient::Path { to, .. } => to,
        }
    }

    pub fn stops(&self) -> &Stops {
        match self {
            Gradient::Linear { stops, .. }
            | Gradient::Radial { stops, .. }
            | Gradient::Path { stops, .. } => stops,
        }
    }

    pub fn stops_mut(&mut self) -> &mut Stops {
        match self {
            Gradient::Linear { stops, .. }
            | Gradient::Radial { stops, .. }
            | Gradient::Path { stops, .. } => stops,
        }
    }

    /// The end colour without its alpha.
    pub fn to_rgb(self) -> [f32; 3] {
        let [r, g, b, _] = self.to();
        [r, g, b]
    }

    /// Whether the end and every stop are opaque, so the ramp paints opaque colour throughout.
    pub fn is_opaque(&self) -> bool {
        self.to()[3] == 1. && self.stops().is_opaque()
    }

    /// Knot `k` of the ramp from `from`: the start, the stops, then the end, as (offset,
    /// straight colour, alpha).
    #[inline]
    fn knot(&self, from: Vec3, k: usize) -> (f32, Vec3, f32) {
        let stops = self.stops();
        let (o, [r, g, b, a]) = if k == 0 {
            (0., [from.x, from.y, from.z, 1.])
        } else if k == stops.len() + 1 {
            (1., self.to())
        } else {
            stops.get(k - 1)
        };
        (o, Vec3::new(r, g, b), a)
    }

    /// The ramp from `from` (opaque) at 0 through the stops to the end colour at 1, at `t`,
    /// interpolated in premultiplied colour; a piece of one alpha interpolates straight colour,
    /// which is the same.
    pub fn ramp(&self, from: Vec3, t: f32) -> RampPoint {
        let n = self.stops().len() + 2;
        let knot = |k| self.knot(from, k);
        // The last piece starting at or before t
        let mut k = 0;
        while k + 2 < n && knot(k + 1).0 <= t {
            k += 1;
        }
        let ((o0, c0, a0), (o1, c1, a1)) = (knot(k), knot(k + 1));
        let width = o1 - o0;
        let u = if width > 1e-9 {
            ((t - o0) / width).clamp(0., 1.)
        } else {
            1.
        };
        let per = |d: Vec3| if width > 1e-9 { d / width } else { Vec3::ZERO };
        let (p0, p1) = (c0 * a0, c1 * a1);
        let alpha = a0 + (a1 - a0) * u;
        let color = if a0 != a1 && alpha > 0. {
            p0.lerp(p1, u) / alpha
        } else {
            c0.lerp(c1, u)
        };
        RampPoint {
            color,
            alpha,
            from_weight: if k == 0 { 1. - u } else { 0. },
            to_weight: if k + 2 == n { u } else { 0. },
            slope: per(c1 - c0),
            premultiplied_slope: per(p1 - p0),
            alpha_slope: per(Vec3::splat(a1 - a0)).x,
        }
    }

    /// The straight colour at `t` of the ramp starting from `from`.
    pub fn color(&self, from: Vec3, t: f32) -> Vec3 {
        if self.stops().is_empty() && self.to()[3] == 1. {
            from.lerp(Vec3::from(self.to_rgb()), t)
        } else {
            self.ramp(from, t).color
        }
    }

    /// The stops (offset, straight colour, alpha) that draw this premultiplied ramp in a renderer
    /// interpolating straight colour: a transparent knot takes its neighbours' colours, and a
    /// piece changing alpha and colour is split to within half a level of 8 bits.
    pub fn svg_stops(&self, from: Vec3) -> Vec<(f32, Vec3, f32)> {
        let n = self.stops().len() + 2;
        let knots: Vec<(f32, Vec3, f32)> = (0..n).map(|k| self.knot(from, k)).collect();
        let mut out = Vec::new();
        for k in 0..n {
            let (o, c, a) = knots[k];
            if a > 0. {
                out.push(knots[k]);
            } else {
                let left = (k > 0 && knots[k - 1].2 > 0.).then(|| knots[k - 1].1);
                let right = (k + 1 < n && knots[k + 1].2 > 0.).then(|| knots[k + 1].1);
                out.push((o, left.or(right).unwrap_or(c), 0.));
                if let (Some(l), Some(r)) = (left, right)
                    && l != r
                {
                    out.push((o, r, 0.));
                }
            }
            if k + 1 < n {
                let (o1, c1, a1) = knots[k + 1];
                if a > 0. && a1 > 0. && a != a1 && c != c1 && o1 > o {
                    split_piece((o, c, a), (o1, c1, a1), 0., 1., 0, &mut out);
                }
            }
        }
        out
    }

    /// The straight colour and the alpha at `t` of the ramp starting from `from`.
    pub fn rgba(&self, from: Vec3, t: f32) -> (Vec3, f32) {
        if self.is_opaque() {
            return (self.color(from, t), 1.);
        }
        let r = self.ramp(from, t);
        (r.color, r.alpha)
    }

    /// The interpolation parameter at `p`, before clamping to `0..=1`.
    fn unclamped(&self, p: Vec2) -> f32 {
        match *self {
            Gradient::Linear { start, end, .. } => {
                let d = end - start;
                (p - start).dot(d) / d.length_squared().max(1e-6)
            }
            Gradient::Radial {
                center,
                focus,
                radius,
                transform,
                ..
            } => radial_t(p, center, focus, radius, transform),
            Gradient::Path {
                contour, near, far, ..
            } => (contour.distance(p) - near) / (far - near).max(1e-6),
        }
    }

    pub fn t(&self, p: Vec2) -> f32 {
        self.unclamped(p).clamp(0., 1.)
    }

    /// The derivative of `t` at `p` by the geometry, as a gradient of the same kind (`to` and
    /// stops zero), zero where `t` is clamped. A radial focus is differentiated numerically, and
    /// so is the rest of a radial gradient with one; a path gradient's curve is not.
    pub fn d_t(&self, p: Vec2) -> Gradient {
        let t = self.unclamped(p);
        let inside = t > 0. && t < 1.;
        match *self {
            // The curve is fitted outside the rasterizer
            Gradient::Path { .. } => self.zeroed(),
            Gradient::Linear { start, end, .. } => {
                if !inside {
                    return self.zeroed();
                }
                let axis = end - start;
                let len_sq = axis.length_squared();
                let d_axis = (p - start - 2. * t * axis) / len_sq;
                Gradient::Linear {
                    start: -axis / len_sq - d_axis,
                    end: d_axis,
                    to: [0.; 4],
                    stops: Stops::NONE,
                }
            }
            Gradient::Radial {
                center,
                focus,
                radius,
                transform,
                ..
            } => {
                let offset = p - center;
                let distance = offset.length();
                if !inside || distance < 1e-6 {
                    return self.zeroed();
                }
                let h = FOCAL_STEP;
                let d = |c: Vec2, f: Vec2, r: f32, m: Mat2| {
                    (radial_t(p, center + c, focus + f, radius + r, transform + m)
                        - radial_t(p, center - c, focus - f, radius - r, transform - m))
                        / (2. * h)
                };
                let (z, o) = (Vec2::ZERO, Mat2::ZERO);
                let d_focus = Vec2::new(d(z, Vec2::X * h, 0., o), d(z, Vec2::Y * h, 0., o));
                if focus == Vec2::ZERO {
                    // t = |v| / r with v = M⁻¹ (p - c) in the gradient's plane
                    let inverse = transform.inverse();
                    let v = inverse * offset;
                    let length = v.length();
                    if length < 1e-6 {
                        return self.zeroed();
                    }
                    let w = inverse.transpose() * v / (length * radius);
                    return Gradient::Radial {
                        center: -w,
                        focus: d_focus,
                        radius: -length / (radius * radius),
                        transform: Mat2::from_cols(-w * v.x, -w * v.y),
                        to: [0.; 4],
                        stops: Stops::NONE,
                    };
                }
                let entry = |k: usize| {
                    let mut m = [0.; 4];
                    m[k] = h;
                    d(z, z, 0., Mat2::from_cols_array(&m))
                };
                Gradient::Radial {
                    center: Vec2::new(d(Vec2::X * h, z, 0., o), d(Vec2::Y * h, z, 0., o)),
                    focus: d_focus,
                    radius: d(z, z, h, o),
                    transform: Mat2::from_cols_array(&[entry(0), entry(1), entry(2), entry(3)]),
                    to: [0.; 4],
                    stops: Stops::NONE,
                }
            }
        }
    }

    /// Adds `k` times the geometry of `other`, which must be of the same kind.
    pub fn add_geometry(&mut self, other: &Gradient, k: f32) {
        match (self, other) {
            (
                Gradient::Linear { start, end, .. },
                Gradient::Linear {
                    start: s, end: e, ..
                },
            ) => {
                *start += k * *s;
                *end += k * *e;
            }
            (
                Gradient::Radial {
                    center,
                    focus,
                    radius,
                    transform,
                    ..
                },
                Gradient::Radial {
                    center: c,
                    focus: f,
                    radius: r,
                    transform: m,
                    ..
                },
            ) => {
                *center += k * *c;
                *focus += k * *f;
                *radius += k * *r;
                *transform += *m * k;
            }
            (Gradient::Path { .. }, Gradient::Path { .. }) => {}
            _ => unreachable!("gradients of different kinds"),
        }
    }

    /// A gradient of the same kind with all parameters zero.
    pub fn zeroed(&self) -> Gradient {
        match self {
            Gradient::Linear { .. } => Gradient::Linear {
                start: Vec2::ZERO,
                end: Vec2::ZERO,
                to: [0.; 4],
                stops: Stops::NONE,
            },
            Gradient::Radial { .. } => Gradient::Radial {
                center: Vec2::ZERO,
                focus: Vec2::ZERO,
                radius: 0.,
                transform: Mat2::ZERO,
                to: [0.; 4],
                stops: Stops::NONE,
            },
            Gradient::Path { contour, .. } => Gradient::Path {
                contour: *contour,
                near: 0.,
                far: 0.,
                to: [0.; 4],
                stops: Stops::NONE,
            },
        }
    }

    /// Moves the gradient by `offset`.
    pub fn translate(&mut self, offset: Vec2) {
        match self {
            Gradient::Linear { start, end, .. } => {
                *start += offset;
                *end += offset;
            }
            Gradient::Radial { center, .. } => *center += offset,
            Gradient::Path { contour, .. } => contour.translate(offset),
        }
    }

    /// The SVG element defining the gradient as `g{id}` in user space, starting from `from`,
    /// each color written by `color`; a ramp with alpha written as [`Gradient::svg_stops`]
    /// gives it, an off-center focus as `fx` and `fy`.
    #[cfg(feature = "svg")]
    pub fn svg(&self, id: usize, from: [f32; 3], color: impl Fn([f32; 3]) -> String) -> String {
        let mut stops = String::new();
        if self.is_opaque() {
            stops = format!("<stop offset=\"0\" stop-color=\"{}\"/>", color(from));
            for (offset, [r, g, b, _]) in self.stops().iter() {
                stops += &format!(
                    "<stop offset=\"{offset:.4}\" stop-color=\"{}\"/>",
                    color([r, g, b])
                );
            }
            stops += &format!(
                "<stop offset=\"1\" stop-color=\"{}\"/>",
                color(self.to_rgb())
            );
        } else {
            for (offset, c, a) in self.svg_stops(Vec3::from(from)) {
                let opacity = if a < 1. {
                    format!(" stop-opacity=\"{a:.3}\"")
                } else {
                    String::new()
                };
                stops += &format!(
                    "<stop offset=\"{offset:.4}\" stop-color=\"{}\"{opacity}/>",
                    color(c.to_array())
                );
            }
        }
        match *self {
            Gradient::Linear { start, end, .. } => format!(
                "<linearGradient id=\"g{id}\" gradientUnits=\"userSpaceOnUse\" x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\">{stops}</linearGradient>",
                start.x, start.y, end.x, end.y
            ),
            Gradient::Radial {
                center,
                focus,
                radius,
                transform,
                ..
            } => {
                let focal = if focus == Vec2::ZERO {
                    String::new()
                } else {
                    let f = center + focus.clamp_length_max(0.999 * radius);
                    format!(" fx=\"{:.2}\" fy=\"{:.2}\"", f.x, f.y)
                };
                // The plane's map about the centre, as SVG's matrix
                let map = if transform == Mat2::IDENTITY {
                    String::new()
                } else {
                    let (m, e) = (transform.to_cols_array(), center - transform * center);
                    format!(
                        " gradientTransform=\"matrix({:.5} {:.5} {:.5} {:.5} {:.3} {:.3})\"",
                        m[0], m[1], m[2], m[3], e.x, e.y
                    )
                };
                format!(
                    "<radialGradient id=\"g{id}\" gradientUnits=\"userSpaceOnUse\" cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.2}\"{focal}{map}>{stops}</radialGradient>",
                    center.x, center.y, radius
                )
            }
            // No SVG paint draws it: emitters draw it as bands (`emit`), this its stops only
            Gradient::Path { .. } => {
                format!("<linearGradient id=\"g{id}\">{stops}</linearGradient>")
            }
        }
    }

    /// Visits the geometric parameters in a fixed order.
    pub fn visit_geometry(&mut self, mut f: impl FnMut(&mut f32)) {
        match self {
            Gradient::Linear { start, end, .. } => {
                [&mut start.x, &mut start.y, &mut end.x, &mut end.y]
                    .into_iter()
                    .for_each(&mut f);
            }
            Gradient::Radial {
                center,
                focus,
                radius,
                transform,
                ..
            } => {
                let mut m = transform.to_cols_array();
                let [a, b, c, d] = &mut m;
                [
                    &mut center.x,
                    &mut center.y,
                    radius,
                    &mut focus.x,
                    &mut focus.y,
                    a,
                    b,
                    c,
                    d,
                ]
                .into_iter()
                .for_each(&mut f);
                *transform = Mat2::from_cols_array(&m);
            }
            Gradient::Path { .. } => {}
        }
    }
}

/// Pushes the stops inside `u0..u1` of the piece between knots `a` and `b` (offset, straight
/// colour, alpha) that straight interpolation needs to stay within half a level of 8 bits of
/// the premultiplied piece, halving where the midpoint misses (at most 6 times deep).
fn split_piece(
    a: (f32, Vec3, f32),
    b: (f32, Vec3, f32),
    u0: f32,
    u1: f32,
    depth: u32,
    out: &mut Vec<(f32, Vec3, f32)>,
) {
    let at = |u: f32| {
        let alpha = a.2 + (b.2 - a.2) * u;
        ((a.1 * a.2).lerp(b.1 * b.2, u) / alpha, alpha)
    };
    let mid = 0.5 * (u0 + u1);
    let (c, alpha) = at(mid);
    let straight = at(u0).0.lerp(at(u1).0, 0.5);
    if depth >= 6 || ((c - straight) * alpha).abs().max_element() <= 0.5 / 255. {
        return;
    }
    split_piece(a, b, u0, mid, depth + 1, out);
    out.push((a.0 + (b.0 - a.0) * mid, c, alpha));
    split_piece(a, b, mid, u1, depth + 1, out);
}

/// The unclamped parameter at `p` of the radial gradient whose plane `transform` maps onto
/// the image about `center`, see [`focal_t`].
fn radial_t(p: Vec2, center: Vec2, focus: Vec2, radius: f32, transform: Mat2) -> f32 {
    if transform == Mat2::IDENTITY {
        return focal_t(p, center, focus, radius);
    }
    let q = center + transform.inverse() * (p - center);
    focal_t(q, center, focus, radius)
}

/// The unclamped parameter of a radial gradient at `p`: the `t` whose circle (centered `t` of
/// the way from the focal point to the center, radius `t` times the radius) passes through `p`.
/// A focus on or beyond the circle is moved just inside it, as SVG does.
fn focal_t(p: Vec2, center: Vec2, focus: Vec2, radius: f32) -> f32 {
    let radius = radius.max(1e-3);
    if focus == Vec2::ZERO {
        return p.distance(center) / radius;
    }
    let focus = focus.clamp_length_max(0.999 * radius);
    // |d - t e| = t r with d from the focal point to p and e from it to the center
    let (d, e) = (p - center - focus, -focus);
    let a = e.length_squared() - radius * radius;
    let de = d.dot(e);
    (de - (de * de - a * d.length_squared()).max(0.).sqrt()) / a
}

impl Path {
    /// The paint's straight colour and its ramp's alpha at `p` (the path's opacity aside), and
    /// the gradient parameter `t` they were interpolated at.
    pub fn paint(&self, p: Vec2) -> (glam::Vec3, f32, f32) {
        let [r, g, b, _] = self.color;
        let from = glam::Vec3::new(r, g, b);
        let Some(gradient) = &self.gradient else {
            return (from, 1., 0.);
        };
        let t = gradient.t(p);
        let (color, alpha) = gradient.rgba(from, t);
        (color, alpha, t)
    }

    /// Zero gradients laid out like the subpaths' points.
    pub(crate) fn zero_points(&self) -> Vec<Vec<Vec2>> {
        let zeros = |s: &Subpath| vec![Vec2::ZERO; s.points.len()];
        self.subpaths.iter().map(zeros).collect()
    }

    /// The gradient by the subpaths' points of `d_flat`, given by the points of `polys`, the
    /// path flattened.
    pub(crate) fn points_gradient(
        &self,
        polys: &[Vec<FlatPoint>],
        d_flat: &[Vec<Vec2>],
    ) -> Vec<Vec<Vec2>> {
        let mut d_points = self.zero_points();
        for ((poly, d_poly), d_subpath) in polys.iter().zip(d_flat).zip(&mut d_points) {
            for (point, d_point) in poly.iter().zip(d_poly) {
                for (&src, &weight) in point.src.iter().zip(&point.weights) {
                    d_subpath[src as usize] += weight * *d_point;
                }
            }
        }
        d_points
    }
}

/// Paths composited bottom to top (see [`super::render_over`]).
#[derive(Clone, Debug)]
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub paths: Vec<Path>,
}

/// A point on a flattened curve as a weighted sum of four control points of its subpath.
#[derive(Clone, Copy, Debug)]
pub struct FlatPoint {
    pub pos: Vec2,
    pub src: [u32; 4],
    pub weights: [f32; 4],
}

/// Max distance in pixels between a curve and its flattened polyline.
const TOLERANCE: f32 = 0.2;

/// Flattens each subpath to a closed polyline. The subdivision count is picked from the current
/// control points but the resulting points are linear in them, so gradients flow through the
/// Bernstein weights while the topology of the polyline stays fixed for this step.
pub fn flatten(path: &Path) -> Vec<Vec<FlatPoint>> {
    path.subpaths
        .iter()
        .map(|subpath| {
            let pts = &subpath.points;
            let n = pts.len();
            let mut out = Vec::new();
            for start in (0..n).step_by(3) {
                let src = [start, start + 1, start + 2, (start + 3) % n].map(|i| i as u32);
                let [p0, p1, p2, p3] = src.map(|i| pts[i as usize]);

                // Wang's formula for the number of segments needed to stay within the tolerance
                let m = (p0 - 2. * p1 + p2)
                    .length()
                    .max((p1 - 2. * p2 + p3).length());
                let segments = (0.75 * m / TOLERANCE).sqrt().ceil().max(1.) as usize;

                // The end point is the next curve's start point, so it is skipped here
                for i in 0..segments {
                    let t = i as f32 / segments as f32;
                    let mt = 1. - t;
                    let weights = [mt * mt * mt, 3. * mt * mt * t, 3. * mt * t * t, t * t * t];
                    let pos = weights[0] * p0 + weights[1] * p1 + weights[2] * p2 + weights[3] * p3;
                    out.push(FlatPoint { pos, src, weights });
                }
            }
            out
        })
        .collect()
}

/// The colour `rgb`, opaque.
pub fn opaque([r, g, b]: [f32; 3]) -> [f32; 4] {
    [r, g, b, 1.]
}
