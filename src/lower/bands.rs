//! A path gradient as bands along its curve, for a host without path gradients: the colour
//! at signed distance `d` from the curve drawn as strokes along it, a stroke of width `2r`
//! with round joins covering exactly `|d| < r`, stacked widest first on each side of the curve
//! and clipped to that side and to the object. The levels, cheapest first: the profile as one
//! blurred step (a terminator) or one blurred stroke (a ridge) where it is one, 4 and 8 bands
//! of equal colour change under a blur of half the narrowest, and 32 exact bands. The SVG
//! emitter writes the same levels.

use crate::render::{FillRule, Gradient};
use glam::{DVec2, Vec2, Vec3};

use crate::model::{Cap, Clip, Curve, Geometry, Join, Object, Paint, StrokeGeometry};

/// The finest level: 32 bands without a blur.
pub const FINEST: u8 = 3;

/// The bands of each level, and whether a blur of half a band hides their steps.
pub const LEVELS: [(usize, bool); 4] = [(4, true), (4, true), (8, true), (32, false)];

/// The profile: the colour at signed distance `d` from the curve.
pub struct Profile<'a> {
    pub gradient: &'a Gradient,
    pub from: Vec3,
    pub near: f32,
    pub far: f32,
}

impl Profile<'_> {
    pub fn t(&self, d: f32) -> f32 {
        ((d - self.near) / (self.far - self.near).max(1e-6)).clamp(0., 1.)
    }

    pub fn at(&self, d: f32) -> Vec3 {
        self.gradient.color(self.from, self.t(d))
    }

    /// The ramp's alpha at signed distance `d`.
    pub fn alpha(&self, d: f32) -> f32 {
        self.gradient.rgba(self.from, self.t(d)).1
    }

    /// The `bands + 1` breakpoints from `near` to `far` that cut the profile into bands of equal
    /// colour change, so the steep stretches get the narrow bands.
    pub fn breaks(&self, bands: usize) -> Vec<f32> {
        let knots = self.knots();
        let mut steps: Vec<f32> = knots.windows(2).map(|w| w[0].1.distance(w[1].1)).collect();
        if !self.gradient.is_opaque() {
            let alphas = self.alphas();
            for (s, a) in steps.iter_mut().zip(alphas.windows(2)) {
                *s += (a[1] - a[0]).abs();
            }
        }
        let total: f32 = steps.iter().sum();
        if total <= 0. {
            let step = (self.far - self.near) / bands as f32;
            return (0..=bands).map(|i| self.near + step * i as f32).collect();
        }
        (0..=bands)
            .map(|i| {
                let mut want = total * i as f32 / bands as f32;
                for (k, &s) in steps.iter().enumerate() {
                    if want <= s || k + 1 == steps.len() {
                        let u = (want / s.max(1e-9)).clamp(0., 1.);
                        return knots[k].0 + u * (knots[k + 1].0 - knots[k].0);
                    }
                    want -= s;
                }
                self.far
            })
            .collect()
    }

    /// The knots: the ends and the stops, as distances and colours, premultiplied where the
    /// ramp has alpha.
    pub fn knots(&self) -> Vec<(f32, Vec3)> {
        let w = self.far - self.near;
        let opaque = self.gradient.is_opaque();
        let colour = |c: [f32; 4]| {
            let straight = glam::Vec4::from(c).truncate();
            if opaque { straight } else { straight * c[3] }
        };
        std::iter::once((self.near, self.from))
            .chain(
                self.gradient
                    .stops()
                    .iter()
                    .map(|(t, c)| (self.near + t * w, colour(c))),
            )
            .chain(std::iter::once((self.far, colour(self.gradient.to()))))
            .collect()
    }

    /// The knots' alphas, in the order of [`Profile::knots`].
    pub fn alphas(&self) -> Vec<f32> {
        std::iter::once(1.)
            .chain(self.gradient.stops().iter().map(|(_, c)| c[3]))
            .chain(std::iter::once(self.gradient.to()[3]))
            .collect()
    }
}

/// What the cheapest level draws: one step (`step` its middle) blurred over the width of the
/// profile, or a ridge (a stroke of `width` in `colour`) blurred over its width.
pub enum Simple {
    Step {
        middle: f32,
        sigma: f32,
    },
    Ridge {
        colour: Vec3,
        width: f32,
        sigma: f32,
    },
}

/// The cheapest drawing of the profile, where it is a step (colours along it monotone) or a
/// ridge on the curve (one inner extreme, within half its width of the curve).
pub fn simple(profile: &Profile) -> Option<Simple> {
    let knots = profile.knots();
    let (a, b) = (knots[0].1, knots[knots.len() - 1].1);
    let axis = (b - a).normalize_or_zero();
    let along: Vec<f32> = knots.iter().map(|(_, c)| (*c - a).dot(axis)).collect();
    let monotone = along.windows(2).all(|w| w[1] >= w[0] - 1e-4);
    let width = profile.far - profile.near;
    if monotone {
        // Where the colour is half way
        let half = 0.5 * (b - a).dot(axis);
        let k = along.iter().position(|&v| v >= half).unwrap_or(0).max(1);
        let (d0, d1, v0, v1) = (knots[k - 1].0, knots[k].0, along[k - 1], along[k]);
        let middle = d0 + (d1 - d0) * ((half - v0) / (v1 - v0).max(1e-6)).clamp(0., 1.);
        let sigma = width / (2. * std::f32::consts::PI).sqrt();
        return Some(Simple::Step { middle, sigma });
    }
    // One extreme away from the line between the ends
    let off: Vec<f32> = knots
        .iter()
        .map(|(_, c)| {
            let v = *c - a;
            (v - axis * v.dot(axis)).length()
        })
        .collect();
    let k = (1..knots.len() - 1).max_by(|&i, &j| off[i].total_cmp(&off[j]))?;
    let ridge = knots[k + 1].0 - knots[k - 1].0;
    if knots[k].0.abs() > 0.25 * ridge {
        return None;
    }
    Some(Simple::Ridge {
        colour: knots[k].1,
        width: 0.5 * ridge,
        sigma: 0.25 * ridge,
    })
}

/// A level's drawing: the breakpoints of its bands, its cheapest drawing where it is the
/// first level and the profile is one, and the blur over it all.
pub struct Level {
    pub breaks: Vec<f32>,
    pub simple: Option<Simple>,
    pub sigma: f32,
}

/// The drawing of `profile` at `level` (see [`LEVELS`]).
pub fn level(profile: &Profile, level: u8) -> Level {
    let (bands, blurred) = LEVELS[level.min(FINEST) as usize];
    let simple = (level == 0 && profile.gradient.is_opaque())
        .then(|| simple(profile))
        .flatten();
    let breaks = profile.breaks(bands);
    let narrowest = breaks
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&w| w > 0.)
        .fold(f32::INFINITY, f32::min);
    let sigma = match &simple {
        Some(Simple::Step { sigma, .. } | Simple::Ridge { sigma, .. }) => *sigma,
        None if blurred && narrowest.is_finite() => 0.5 * narrowest,
        None => 0.,
    };
    Level {
        breaks,
        simple,
        sigma,
    }
}

/// The object, painted with a path gradient, as the bands of the coarsest level within its
/// price that the host draws (the blurred ones only with a blur).
pub(super) fn lower(l: &mut super::Lowering, o: &Object) -> Option<Vec<Object>> {
    let Paint::Gradient(g) = &o.paint else {
        return None;
    };
    let (from, gradient) = g.ramp();
    let Gradient::Path {
        contour, near, far, ..
    } = gradient
    else {
        return None;
    };
    l.report.path_gradients += 1;
    let profile = Profile {
        gradient,
        from,
        near: *near,
        far: *far,
    };
    let (loops, rule) = match &o.geometry {
        Geometry::Fill { loops, rule } => (loops.clone(), *rule),
        stroke @ Geometry::Stroke(_) => {
            let (loops, rule) = stroke.outline();
            (crate::model::loops64(&loops), rule)
        }
    };
    // What the strokes run along (an open curve with its rays to the box), and the region of
    // negative distance (an open curve closed by its rays and the box)
    let cubics: Vec<Curve> = contour.cubics().map(|c| c.map(glam::Vec2::as_dvec2)).collect();
    let (centerline, region) = if contour.is_closed() {
        (cubics.clone(), cubics)
    } else {
        let closure: Vec<DVec2> = contour.closure().iter().map(|p| p.as_dvec2()).collect();
        let (first, last) = (cubics[0][0], cubics[cubics.len() - 1][3]);
        let (end, start) = (closure[0], closure[closure.len() - 1]);
        let centerline = std::iter::once(line(start, first))
            .chain(cubics.iter().copied())
            .chain(std::iter::once(line(last, end)))
            .collect();
        let mut region = cubics.clone();
        region.push(line(last, end));
        region.extend(closure.windows(2).map(|w| line(w[0], w[1])));
        region.push(line(start, first));
        (centerline, region)
    };
    let (min, max) = crate::model::bounds(&loops);
    let object = l.clip(Clip {
        loops,
        rule,
        parent: o.clip,
    });
    let side = l.clip(Clip {
        loops: vec![region.clone()],
        rule: FillRule::EvenOdd,
        parent: Some(object),
    });
    let drawing = Drawing {
        o,
        profile: &profile,
        centerline,
        closed: contour.is_closed(),
        region,
        clips: [object, side],
        bounds: (min, max),
    };
    // The first level differs from the second only where the profile is a step or a ridge
    let levels: Vec<u8> = if l.host.blur {
        (0..=FINEST).collect()
    } else {
        vec![FINEST]
    };
    let forms = levels.into_iter().filter_map(|k| {
        let lv = level(&profile, k);
        (k > 0 || lv.simple.is_some()).then(|| drawing.objects(&lv))
    });
    Some(l.coarsest(o, forms))
}

/// The breakpoints on the side of `sign` as radii, widest first, each with the next radius in,
/// the inner one of its band.
pub fn rings(breaks: &[f32], sign: f32) -> Vec<(f32, f32)> {
    let mut radii: Vec<f32> = breaks.iter().map(|s| sign * s).filter(|&r| r > 0.).collect();
    radii.sort_by(|a, b| b.total_cmp(a));
    let inner = radii.iter().skip(1).copied().chain([0.]);
    radii.iter().copied().zip(inner).collect()
}

/// The straight line from `a` to `b` as a cubic.
pub(crate) fn line(a: DVec2, b: DVec2) -> Curve {
    [a, a + (b - a) / 3., a + (b - a) * (2. / 3.), b]
}

/// What every level of one path gradient draws with.
struct Drawing<'a> {
    o: &'a Object,
    profile: &'a Profile<'a>,
    centerline: Vec<Curve>,
    closed: bool,
    region: Vec<Curve>,
    /// The object's clip and the inside's, within it.
    clips: [usize; 2],
    bounds: (Vec2, Vec2),
}

impl Drawing<'_> {
    /// The level's objects: each side's far value over the box, then a stroke per breakpoint on
    /// it, widest first, in the value of the band inside it; the outside is drawn whole and the
    /// inside over it clipped, so the curve's anti-aliasing lets nothing through.
    fn objects(&self, level: &Level) -> Vec<Object> {
        let sigma = level.sigma;
        let pad = 3. * sigma + 2.;
        let (lo, hi) = (
            (self.bounds.0 - pad).as_dvec2(),
            (self.bounds.1 + pad).as_dvec2(),
        );
        let corners = [lo, DVec2::new(hi.x, lo.y), hi, DVec2::new(lo.x, hi.y)];
        let rect: Vec<Curve> = (0..4)
            .map(|k| line(corners[k], corners[(k + 1) % 4]))
            .collect();
        // A ramp with alpha takes its alpha at the band as opacity, which stacking only
        // approximates (the tracer fits path gradients opaque)
        let object = |geometry: Geometry, d: f32, clip: usize| Object {
            geometry,
            paint: Paint::Solid(self.profile.at(d).as_dvec3()),
            opacity: self.o.opacity * f64::from(self.profile.alpha(d)),
            blur: f64::from(sigma),
            clip: Some(clip),
        };
        let fill = |loops: Vec<Curve>, rule: FillRule| Geometry::Fill {
            loops: vec![loops],
            rule,
        };
        let stroke = |width: f32| {
            Geometry::Stroke(StrokeGeometry {
                centerline: self.centerline.clone(),
                closed: self.closed,
                width: f64::from(width),
                cap: Cap::Butt,
                join: Join::Round,
                corners: vec![true; self.centerline.len()],
                dashes: None,
            })
        };
        let [outside, inside] = self.clips;
        if let Some(simple) = &level.simple {
            // The inside's colour over the outside's, blurred, the curve moved by a stroke
            let (a, b) = (f32::NEG_INFINITY, f32::INFINITY);
            let mut out = vec![
                object(fill(rect, FillRule::NonZero), b, outside),
                object(fill(self.region.clone(), FillRule::EvenOdd), a, outside),
            ];
            match *simple {
                Simple::Step { middle, .. } if middle.abs() > 0.05 => out.push(object(
                    stroke(2. * middle.abs()),
                    if middle > 0. { a } else { b },
                    outside,
                )),
                Simple::Ridge { colour, width, .. } => out.push(Object {
                    paint: Paint::Solid(colour.as_dvec3()),
                    ..object(stroke(width), a, outside)
                }),
                Simple::Step { .. } => {}
            }
            return out;
        }
        let mut out = Vec::new();
        for (clip, sign, base) in [
            (outside, 1f32, f32::INFINITY),
            (inside, -1., f32::NEG_INFINITY),
        ] {
            out.push(object(fill(rect.clone(), FillRule::NonZero), base, clip));
            for (r, inner) in rings(&level.breaks, sign) {
                out.push(object(stroke(2. * r), sign * 0.5 * (r + inner), clip));
            }
        }
        out
    }
}
