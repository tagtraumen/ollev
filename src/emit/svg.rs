//! SVG with filters and clips: one `<path>` per object, a layer's runs as groups, gradients
//! and blur filters in the definitions after them. A coordinate's decimals grow with the
//! object's encoding level.

use std::fmt::Write;

use crate::model::{Curve, Geometry, Instance, Item, Model, Object, Paint};
use crate::render::FillRule;
use crate::stroke::{Cap, Join};
use glam::{DVec3, Vec2, Vec3};
use std::collections::HashMap;

/// Levels above the first an object's coordinates may step to (three and four decimals).
pub const FINEST: u8 = 2;

/// Decimals of a coordinate at encoding level `level`.
fn decimals(level: u8) -> usize {
    2 + level as usize
}

pub(crate) fn hex(c: DVec3) -> String {
    let q = |v: f64| (v.clamp(0., 1.) * 255.).round() as u8;
    format!("#{:02x}{:02x}{:02x}", q(c.x), q(c.y), q(c.z))
}

/// The `fill-rule` or `clip-rule` attribute, none for SVG's default (nonzero).
pub(crate) fn rule_attribute(paint: &str, rule: FillRule) -> String {
    match rule {
        FillRule::EvenOdd => format!(" {paint}-rule=\"evenodd\""),
        FillRule::NonZero => String::new(),
    }
}

/// The `clip-path` attribute of clip `clip`, if any.
fn clip_attribute(clip: Option<usize>) -> String {
    clip.map_or(String::new(), |c| format!(" clip-path=\"url(#c{c})\""))
}

/// The cubics as path data from their start, `Z` at the end when closed.
pub(crate) fn curves(cubics: &[Curve], close: bool, p: usize) -> String {
    let mut d = String::new();
    if let Some(f) = cubics.first() {
        let _ = write!(d, "M{:.*} {:.*}", p, f[0].x, p, f[0].y);
    }
    for c in cubics {
        let _ = write!(
            d,
            "C{:.*} {:.*} {:.*} {:.*} {:.*} {:.*}",
            p, c[1].x, p, c[1].y, p, c[2].x, p, c[2].y, p, c[3].x, p, c[3].y
        );
    }
    if close {
        d.push('Z');
    }
    d
}

/// Closed loops as path data, empty loops left out.
pub(crate) fn loops(loops: &[Vec<Curve>], p: usize) -> String {
    loops
        .iter()
        .filter(|l| !l.is_empty())
        .map(|l| curves(l, true, p))
        .collect()
}

/// The dash offset as written, at `decimals`: one landing on a dash's end as written moves a
/// written step into the gap, since a renderer draws that dash's zero-length rest with its caps.
pub(crate) fn dash_offset(array: &[f64], offset: f64, decimals: usize) -> f64 {
    let scale = 10f64.powi(decimals as i32);
    let round = |v: f64| (v * scale).round() / scale;
    let lengths: Vec<f64> = array.iter().map(|&v| round(v)).collect();
    // An odd array repeats once more, so its dashes and gaps alternate over twice its length
    let repeats = if lengths.len() % 2 == 1 { 2 } else { 1 };
    let period: f64 = lengths.iter().sum::<f64>() * repeats as f64;
    let written = round(offset);
    if period <= 0. {
        return written;
    }
    let phase = round(written.rem_euclid(period));
    let mut end = 0.;
    for (k, &len) in lengths
        .iter()
        .cycle()
        .take(lengths.len() * repeats)
        .enumerate()
    {
        end = round(end + len);
        if k % 2 == 0 && len > 0. && (phase - end).abs() < 0.5 / scale {
            return written + 1. / scale;
        }
    }
    written
}

/// An instance: its definition under its matrix.
fn use_of(svg: &mut String, i: &Instance) {
    let m = i.transform;
    let _ = writeln!(
        svg,
        "<use href=\"#d{}\" transform=\"matrix({} {} {} {} {} {})\"/>",
        i.definition, m[0], m[1], m[2], m[3], m[4], m[5]
    );
}

struct Writer<'a> {
    levels: &'a [u8],
    /// The next object of [`Model::objects`].
    next: usize,
    defs: String,
    gradients: usize,
    filters: usize,
    /// Path gradients written, which number their definitions.
    bands: usize,
    groups: HashMap<usize, usize>,
}

impl Writer<'_> {
    fn items(&mut self, svg: &mut String, items: &[Item], level: Option<u8>) {
        for item in items {
            match item {
                Item::Object(o) => {
                    let l = level.unwrap_or_else(|| {
                        self.next += 1;
                        self.levels.get(self.next - 1).copied().unwrap_or(0)
                    });
                    self.object(svg, o, l);
                }
                Item::Group { layer, items } => {
                    let count = self.groups.entry(*layer).or_default();
                    *count += 1;
                    match *count {
                        1 => writeln!(svg, "<g id=\"layer-{layer}\">"),
                        c => writeln!(svg, "<g id=\"layer-{layer}-{c}\">"),
                    }
                    .unwrap();
                    self.items(svg, items, level);
                    svg.push_str("</g>\n");
                }
                Item::Instance(i) => use_of(svg, i),
                Item::Repeat {
                    structure,
                    instances,
                } => {
                    // A repeat names its structure and parameters for Graphite's repeat nodes
                    let (name, parameters) = structure.parameters();
                    let parameters: Vec<String> =
                        parameters.iter().map(|v| format!("{v:.4}")).collect();
                    let _ = writeln!(
                        svg,
                        "<g data-repeat=\"{name}\" data-count=\"{}\" data-parameters=\"{}\">",
                        instances.len(),
                        parameters.join(" ")
                    );
                    for i in instances {
                        use_of(svg, i);
                    }
                    svg.push_str("</g>\n");
                }
            }
        }
    }

    fn object(&mut self, svg: &mut String, o: &Object, level: u8) {
        if let Paint::Gradient(g) = &o.paint
            && let (from, gradient @ crate::render::Gradient::Path { .. }) = g.ramp()
        {
            let clip = clip_attribute(o.clip);
            let n = self.bands;
            self.bands += 1;
            return crate::emit::bands::write(
                svg,
                &mut self.defs,
                n,
                o,
                from,
                gradient,
                level,
                &clip,
            );
        }
        let p = decimals(level);
        let paint = match &o.paint {
            Paint::Solid(c) => hex(*c),
            Paint::Gradient(g) => {
                let id = self.gradients;
                self.gradients += 1;
                let color = |c: [f32; 3]| hex(Vec3::from(c).as_dvec3());
                let (from, gradient) = g.ramp();
                let _ = writeln!(self.defs, "{}", gradient.svg(id, from.to_array(), color));
                format!("url(#g{id})")
            }
        };
        // The paint's opacity, where it does not cover
        let opacity = |paint: &str| {
            let o = o.opacity as f32;
            if o < 1. - 0.5 / 255. {
                format!(" {paint}-opacity=\"{o:.3}\"")
            } else {
                String::new()
            }
        };
        let clip = clip_attribute(o.clip);
        match &o.geometry {
            Geometry::Stroke(s) => {
                let cap = match s.cap {
                    Cap::Butt => "butt",
                    Cap::Round => "round",
                    Cap::Square => "square",
                };
                let join = match s.join {
                    Join::Miter => "miter",
                    Join::Round => "round",
                    Join::Bevel => "bevel",
                };
                // the pattern at the path's own precision, finer where the verify loop steps it
                let dashes = s.dashes.as_ref().map_or(String::new(), |(array, offset)| {
                    let offset = dash_offset(array, *offset, p);
                    let array: Vec<String> = array.iter().map(|v| format!("{v:.p$}")).collect();
                    format!(
                        " stroke-dasharray=\"{}\" stroke-dashoffset=\"{offset:.p$}\"",
                        array.join(" ")
                    )
                });
                let _ = writeln!(
                    svg,
                    "<path d=\"{}\" fill=\"none\" stroke=\"{paint}\" stroke-width=\"{:.2}\" stroke-linecap=\"{cap}\" stroke-linejoin=\"{join}\"{dashes}{}{clip}/>",
                    curves(&s.centerline, s.closed, p),
                    s.width,
                    opacity("stroke"),
                );
            }
            Geometry::Fill { loops: l, rule } => {
                // A blur is a filter over the box its kernel reaches, applied before the clip
                let filter = if o.blur > 0. {
                    let id = self.filters;
                    self.filters += 1;
                    let sigma = o.blur as f32;
                    let reach = Vec2::splat(crate::render::blur::reach(sigma) as f32);
                    let (min, max) = crate::model::bounds(l);
                    let (min, max) = (min - reach, max + reach);
                    let _ = writeln!(
                        self.defs,
                        "<filter id=\"b{id}\" filterUnits=\"userSpaceOnUse\" x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" color-interpolation-filters=\"sRGB\"><feGaussianBlur stdDeviation=\"{sigma:.3}\"/></filter>",
                        min.x,
                        min.y,
                        max.x - min.x,
                        max.y - min.y,
                    );
                    format!(" filter=\"url(#b{id})\"")
                } else {
                    String::new()
                };
                let _ = writeln!(
                    svg,
                    "<path d=\"{}\" fill=\"{paint}\"{}{}{filter}{clip}/>",
                    loops(l, p),
                    rule_attribute("fill", *rule),
                    opacity("fill"),
                );
            }
        }
    }
}

/// The model as an SVG document, object `k` of [`Model::objects`] at level `levels[k]` (0
/// where none is given).
#[must_use]
pub fn encode(model: &Model, levels: &[u8]) -> String {
    let (w, h) = (model.width, model.height);
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">\n"
    );
    let mut writer = Writer {
        levels,
        next: 0,
        defs: String::new(),
        gradients: 0,
        filters: 0,
        bands: 0,
        groups: HashMap::new(),
    };
    for (c, clip) in model.clips.iter().enumerate() {
        let _ = writeln!(
            svg,
            "<clipPath id=\"c{c}\"{}><path d=\"{}\"{}/></clipPath>",
            clip_attribute(clip.parent),
            loops(&clip.loops, 2),
            rule_attribute("clip", clip.rule)
        );
    }
    // A transparent picture's opaque part clips the whole drawing
    if let Some(c) = model.clip {
        let _ = writeln!(svg, "<g clip-path=\"url(#c{c})\">");
    }
    writer.items(&mut svg, &model.items, None);
    if model.clip.is_some() {
        svg.push_str("</g>\n");
    }
    let mut defs = std::mem::take(&mut writer.defs);
    for (n, d) in model.definitions.iter().enumerate() {
        let mut group = format!("<g id=\"d{n}\">\n");
        writer.items(&mut group, d, Some(0));
        group.push_str("</g>\n");
        defs += &writer.defs;
        writer.defs.clear();
        defs += &group;
    }
    if !defs.is_empty() {
        svg.push_str("<defs>\n");
        svg.push_str(&defs);
        svg.push_str("</defs>\n");
    }
    svg.push_str("</svg>\n");
    svg
}

/// The SVG target, rendered by resvg.
pub struct Svg;

impl super::Emitter for Svg {
    fn encode(&self, model: &Model, levels: &[u8]) -> Vec<u8> {
        encode(model, levels).into_bytes()
    }

    fn finest(&self, o: &Object) -> u8 {
        match &o.paint {
            Paint::Gradient(g) if matches!(g.ramp().1, crate::render::Gradient::Path { .. }) => {
                crate::emit::bands::FINEST
            }
            _ => FINEST,
        }
    }

    fn render(&self, bytes: &[u8], width: usize, height: usize) -> Option<Vec<Vec3>> {
        super::verify::render_svg(std::str::from_utf8(bytes).ok()?, width, height)
    }
}
