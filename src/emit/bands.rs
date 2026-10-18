//! A path gradient in SVG, which has no such paint: the colour at signed distance `d` from a
//! closed curve drawn as strokes along the curve, a stroke of width `2r` with round joins
//! covering exactly `|d| < r`, stacked widest first on each side of the curve and clipped to
//! that side and to the object. The levels, cheapest first: the profile as one blurred step
//! (a terminator) or one blurred stroke (a ridge) where it is one, 4 and 8 bands of equal colour
//! change under a blur of half the narrowest, and 32 exact bands.

use std::fmt::Write;

use crate::model::{Geometry, Object};
use crate::render::Gradient;
use glam::{Vec2, Vec3};

use crate::emit::svg::{curves, hex, loops, rule_attribute};
use crate::lower::bands;
pub use bands::FINEST;
use bands::{Level, Profile, Simple};

/// Writes `o`, painted with the path gradient `gradient` starting from `from`, as the `n`th
/// path gradient at `level`, its definitions in `defs`; `clip` is the object's own clip.
#[allow(clippy::too_many_arguments)]
pub fn write(
    svg: &mut String,
    defs: &mut String,
    n: usize,
    o: &Object,
    from: Vec3,
    gradient: &Gradient,
    level: u8,
    clip: &str,
) {
    let Gradient::Path {
        contour, near, far, ..
    } = gradient
    else {
        return;
    };
    let Geometry::Fill { loops: l, rule } = &o.geometry else {
        return;
    };
    let profile = Profile {
        gradient,
        from,
        near: *near,
        far: *far,
    };
    let cubics: Vec<[glam::DVec2; 4]> = contour.cubics().map(|c| c.map(|p| p.as_dvec2())).collect();
    // What the strokes run along (an open curve with its rays to the box), and the region of
    // negative distance (an open curve closed by its rays and the box)
    let (curve, region) = if contour.is_closed() {
        let c = curves(&cubics, true, 2);
        (c.clone(), c)
    } else {
        let open = curves(&cubics, false, 2);
        let closure = contour.closure();
        let (end, start) = (closure[0], closure[closure.len() - 1]);
        let line: String = closure
            .iter()
            .map(|q| format!("L{:.2} {:.2}", q.x, q.y))
            .collect();
        (
            format!(
                "M{:.2} {:.2}L{}L{:.2} {:.2}",
                start.x,
                start.y,
                &open[1..],
                end.x,
                end.y
            ),
            format!("{open}{line}Z"),
        )
    };
    let points = l.iter().flatten().flatten().map(|p| p.as_vec2());
    let (min, max) = points.fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(a, b), p| {
        (a.min(p), b.max(p))
    });
    let Level {
        breaks,
        simple,
        sigma,
    } = bands::level(&profile, level);
    let pad = 3. * sigma + 2.;
    let (lo, hi) = (min - pad, max + pad);
    let rect = |fill: Vec3| {
        format!(
            "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{}\"/>",
            lo.x,
            lo.y,
            hi.x - lo.x,
            hi.y - lo.y,
            hex(fill.as_dvec3())
        )
    };
    let stroke = |colour: Vec3, width: f32| {
        format!(
            "<use href=\"#pc{n}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{width:.2}\" stroke-linejoin=\"round\"/>",
            hex(colour.as_dvec3())
        )
    };
    let _ = writeln!(defs, "<path id=\"pc{n}\" d=\"{curve}\" fill=\"none\"/>");
    let _ = writeln!(
        defs,
        "<clipPath id=\"po{n}\"{clip}><path d=\"{}\"{}/></clipPath>",
        loops(l, 2),
        rule_attribute("clip", *rule)
    );
    let blur = if sigma > 0. {
        let _ = writeln!(
            defs,
            "<filter id=\"pf{n}\" filterUnits=\"userSpaceOnUse\" x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" color-interpolation-filters=\"sRGB\"><feGaussianBlur stdDeviation=\"{sigma:.3}\"/></filter>",
            lo.x,
            lo.y,
            hi.x - lo.x,
            hi.y - lo.y
        );
        format!(" filter=\"url(#pf{n})\"")
    } else {
        String::new()
    };
    let opacity = if o.opacity < 1. - 0.5 / 255. {
        format!(" opacity=\"{:.3}\"", o.opacity)
    } else {
        String::new()
    };
    // A ramp with alpha: the colours' bands masked by the same bands in the alpha's greys
    let mask = if gradient.is_opaque() {
        String::new()
    } else {
        let mut content = String::new();
        sides(
            &mut content,
            n,
            &breaks,
            &|d| Vec3::splat(profile.alpha(d)),
            &blur,
            &rect,
            &stroke,
        );
        let _ = writeln!(
            defs,
            "<mask id=\"pm{n}\" maskUnits=\"userSpaceOnUse\" x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\">\n{content}</mask>",
            lo.x,
            lo.y,
            hi.x - lo.x,
            hi.y - lo.y
        );
        format!(" mask=\"url(#pm{n})\"")
    };
    let _ = writeln!(svg, "<g clip-path=\"url(#po{n})\"{opacity}{mask}>");
    if let Some(simple) = simple {
        // The inside's colour over the outside's, blurred, the curve moved by a stroke
        let (a, b) = (profile.at(f32::NEG_INFINITY), profile.at(f32::INFINITY));
        let _ = writeln!(svg, "<g{blur}>{}", rect(b));
        let _ = writeln!(
            svg,
            "<path d=\"{region}\" fill=\"{}\" fill-rule=\"evenodd\"/>",
            hex(a.as_dvec3())
        );
        match simple {
            Simple::Step { middle, .. } if middle.abs() > 0.05 => {
                let _ = writeln!(
                    svg,
                    "{}",
                    stroke(if middle > 0. { a } else { b }, 2. * middle.abs())
                );
            }
            Simple::Ridge { colour, width, .. } => {
                let _ = writeln!(svg, "{}", stroke(colour, width));
            }
            Simple::Step { .. } => {}
        }
        svg.push_str("</g>\n</g>\n");
        return;
    }
    let _ = writeln!(
        defs,
        "<clipPath id=\"pi{n}\"><path d=\"{region}\" clip-rule=\"evenodd\"/></clipPath>"
    );
    sides(svg, n, &breaks, &|d| profile.at(d), &blur, &rect, &stroke);
    svg.push_str("</g>\n");
}

/// Each side of the curve: its far value over the box, then a stroke per breakpoint on it,
/// widest first, each in the value of the band inside it. The outside is drawn whole and the
/// inside over it clipped (`pi{n}`), so the curve's anti-aliasing lets nothing through.
#[allow(clippy::too_many_arguments)]
fn sides(
    svg: &mut String,
    n: usize,
    breaks: &[f32],
    value: &dyn Fn(f32) -> Vec3,
    blur: &str,
    rect: &dyn Fn(Vec3) -> String,
    stroke: &dyn Fn(Vec3, f32) -> String,
) {
    for (clipped, sign, base) in [(false, 1f32, f32::INFINITY), (true, -1., f32::NEG_INFINITY)] {
        let mut radii: Vec<f32> = breaks
            .iter()
            .map(|s| sign * s)
            .filter(|&r| r > 0.)
            .collect();
        radii.sort_by(|a, b| b.total_cmp(a));
        let clip = if clipped {
            format!(" clip-path=\"url(#pi{n})\"")
        } else {
            String::new()
        };
        let _ = writeln!(svg, "<g{clip}><g{blur}>{}", rect(value(base)));
        for (k, &r) in radii.iter().enumerate() {
            let inner = radii.get(k + 1).copied().unwrap_or(0.);
            let colour = value(sign * 0.5 * (r + inner));
            let _ = writeln!(svg, "{}", stroke(colour, 2. * r));
        }
        svg.push_str("</g></g>\n");
    }
}
