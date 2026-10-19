//! A radial gradient's focus, for a host without one: the focus dropped where that misses
//! within the price, else the gradient as nested discs at its own circles (each `t`'s circle,
//! centred `t` of the way from the focus to the centre), clipped to the object, the outermost
//! first, each painted with its band's colour.

use crate::render::{FillRule, Gradient as Ramp};
use glam::{DVec2, DVec3, Vec2};

use super::Lowering;
use crate::model::{Clip, Curve, Geometry, Gradient, Object, Paint};

/// Band counts tried after the dropped focus, coarsest first.
const BANDS: [usize; 5] = [8, 16, 32, 64, 128];

/// The share of the radius a cubic's handles reach along a quarter circle.
pub(super) const KAPPA: f64 = 0.552_284_749_830_793_4;

pub(super) fn lower(l: &mut Lowering, o: &Object) -> Option<Vec<Object>> {
    let Paint::Gradient(g) = &o.paint else {
        return None;
    };
    let (from, ramp) = g.ramp();
    let Ramp::Radial { focus, .. } = *ramp else {
        return None;
    };
    if focus == Vec2::ZERO {
        return None;
    }
    l.report.foci += 1;
    let mut centred = *ramp;
    if let Ramp::Radial { focus, .. } = &mut centred {
        *focus = Vec2::ZERO;
    }
    let dropped = Object {
        paint: Paint::Gradient(Gradient::new(from, centred)),
        ..o.clone()
    };
    if l.quick {
        return Some(vec![dropped]);
    }
    let (loops, rule) = o.geometry.outline();
    let clip = l.clip(Clip {
        loops: crate::model::loops64(&loops),
        rule,
        parent: o.clip,
    });
    let forms = std::iter::once(vec![dropped]).chain(BANDS.iter().map(|&n| discs(o, g, clip, n)));
    Some(l.coarsest(o, forms))
}

/// The gradient as `n` nested discs over the object painted at its outer colour: band `j`
/// covers `t` in `(j - 1) / n .. j / n` in the colour at its middle. With alpha, each disc's
/// opacity and colour are those that bring the stack to the band's own, which holds where the
/// alpha grows inwards (a glow), and is clamped where it does not.
fn discs(o: &Object, g: &Gradient, clip: usize, n: usize) -> Vec<Object> {
    let crate::model::GradientShape::Radial {
        transform: m,
        focus,
    } = g.shape()
    else {
        return Vec::new();
    };
    let map = |u: DVec2| {
        DVec2::new(
            m[0] * u.x + m[2] * u.y + m[4],
            m[1] * u.x + m[3] * u.y + m[5],
        )
    };
    let disc = |t: f64| -> Vec<Curve> {
        let c = focus * (1. - t);
        let quarter = |a: DVec2, b: DVec2| {
            [
                map(c + a * t),
                map(c + (a + b * KAPPA) * t),
                map(c + (b + a * KAPPA) * t),
                map(c + b * t),
            ]
        };
        let axes = [DVec2::X, DVec2::Y, -DVec2::X, -DVec2::Y];
        (0..4)
            .map(|k| quarter(axes[k], axes[(k + 1) % 4]))
            .collect()
    };
    // The bands' targets from the outside in: premultiplied colour and alpha
    let target = |t: f64| {
        let [r, gr, b, a] = g.color(t);
        let a = a * o.opacity;
        (DVec3::new(r, gr, b) * a, a)
    };
    let mut out = Vec::with_capacity(n + 1);
    let (mut below, mut below_alpha) = (DVec3::ZERO, 0.);
    for j in (0..=n).rev() {
        // The object itself first, at the colour beyond the outer circle
        let (t, geometry, clip) = if j == n {
            (1., o.geometry.clone(), o.clip)
        } else {
            let loops = vec![disc((j + 1) as f64 / n as f64)];
            let fill = Geometry::Fill {
                loops,
                rule: FillRule::NonZero,
            };
            ((j as f64 + 0.5) / n as f64, fill, Some(clip))
        };
        let (want, alpha) = target(t);
        let opacity = if below_alpha >= 1. {
            1.
        } else {
            (1. - (1. - alpha) / (1. - below_alpha)).clamp(0., 1.)
        };
        if opacity <= 0. {
            continue;
        }
        let colour = ((want - below * (1. - opacity)) / opacity).clamp(DVec3::ZERO, DVec3::ONE);
        (below, below_alpha) = (
            below * (1. - opacity) + colour * opacity,
            below_alpha + opacity * (1. - below_alpha),
        );
        out.push(Object {
            geometry,
            paint: Paint::Solid(colour),
            opacity,
            blur: o.blur,
            clip,
        });
    }
    out
}
