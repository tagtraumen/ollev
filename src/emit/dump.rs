//! The model as JSON, every number at full precision: the native form a Graphite importer
//! could read. It has no renderer here, so the verify loop checks nothing.

use std::fmt::Write;

use crate::model::{Clip, Curve, Geometry, GradientShape, Item, Model, Object, Paint, Structure};
use glam::{DVec2, DVec3};

fn rule(r: crate::model::FillRule) -> &'static str {
    match r {
        crate::model::FillRule::NonZero => "nonzero",
        crate::model::FillRule::EvenOdd => "evenodd",
    }
}

fn rgb(c: DVec3) -> String {
    format!("[{},{},{}]", c.x, c.y, c.z)
}

/// A straight RGBA colour, its alpha written only where it is below one.
fn rgba(c: [f64; 4]) -> String {
    match c[3] < 1. {
        true => format!("[{},{},{},{}]", c[0], c[1], c[2], c[3]),
        false => format!("[{},{},{}]", c[0], c[1], c[2]),
    }
}

fn point(p: DVec2) -> String {
    format!("[{},{}]", p.x, p.y)
}

fn curves(cubics: &[Curve]) -> String {
    let c: Vec<String> = cubics
        .iter()
        .map(|c| {
            let p: Vec<String> = c.iter().map(|p| format!("[{},{}]", p.x, p.y)).collect();
            format!("[{}]", p.join(","))
        })
        .collect();
    format!("[{}]", c.join(","))
}

fn loops(loops: &[Vec<Curve>]) -> String {
    let l: Vec<String> = loops.iter().map(|l| curves(l)).collect();
    format!("[{}]", l.join(","))
}

fn opt(v: Option<usize>) -> String {
    v.map_or("null".into(), |v| v.to_string())
}

fn paint(p: &Paint) -> String {
    match p {
        Paint::Solid(c) => format!("{{\"solid\":{}}}", rgb(*c)),
        Paint::Gradient(g) => {
            let stops: Vec<String> = g
                .stops()
                .iter()
                .map(|s| format!("[{},{}]", s.offset, rgba(s.color)))
                .collect();
            let geometry = match g.shape() {
                GradientShape::Linear { start, end } => format!(
                    "\"linear\":{{\"start\":{},\"end\":{}}}",
                    point(start),
                    point(end)
                ),
                GradientShape::Radial { transform, focus } => format!(
                    "\"radial\":{{\"transform\":{transform:?},\"focus\":{}}}",
                    point(focus)
                ),
                GradientShape::Path {
                    curve,
                    closed,
                    closure,
                    near,
                    far,
                } => {
                    let closure: Vec<String> = closure.into_iter().map(point).collect();
                    format!(
                        "\"path\":{{\"curve\":{},\"closed\":{closed},\"closure\":[{}],\"near\":{near},\"far\":{far}}}",
                        curves(&curve),
                        closure.join(",")
                    )
                }
            };
            format!("{{{geometry},\"stops\":[{}]}}", stops.join(","))
        }
    }
}

fn structure(s: &Structure) -> String {
    match s {
        Structure::List => "\"list\"".into(),
        Structure::Linear { step } => format!("{{\"linear\":{{\"step\":{}}}}}", point(*step)),
        Structure::Grid { a, b, columns } => format!(
            "{{\"grid\":{{\"a\":{},\"b\":{},\"columns\":{columns}}}}}",
            point(*a),
            point(*b)
        ),
        Structure::Circular { center, step } => format!(
            "{{\"circular\":{{\"center\":{},\"step\":{step}}}}}",
            point(*center)
        ),
    }
}

fn object(o: &Object) -> String {
    let geometry = match &o.geometry {
        Geometry::Fill { loops: l, rule: r } => {
            format!(
                "{{\"fill\":{{\"rule\":\"{}\",\"loops\":{}}}}}",
                rule(*r),
                loops(l)
            )
        }
        Geometry::Stroke(s) => format!(
            "{{\"stroke\":{{\"centerline\":{},\"closed\":{},\"width\":{},\"cap\":\"{:?}\",\"join\":\"{:?}\",\"dashes\":{}}}}}",
            curves(&s.centerline),
            s.closed,
            s.width,
            s.cap,
            s.join,
            s.dashes.as_ref().map_or("null".into(), |(a, o)| format!(
                "{{\"array\":{a:?},\"offset\":{o}}}"
            ))
        ),
    };
    format!(
        "{{\"object\":{{\"geometry\":{geometry},\"paint\":{},\"opacity\":{},\"blur\":{},\"clip\":{}}}}}",
        paint(&o.paint),
        o.opacity,
        o.blur,
        opt(o.clip)
    )
}

fn instance(i: &crate::model::Instance) -> String {
    format!(
        "{{\"definition\":{},\"transform\":{:?}}}",
        i.definition, i.transform
    )
}

fn items(list: &[Item]) -> String {
    let i: Vec<String> = list
        .iter()
        .map(|i| match i {
            Item::Object(o) => object(o),
            Item::Group { layer, items: g } => {
                format!("{{\"group\":{{\"layer\":{layer},\"items\":{}}}}}", items(g))
            }
            Item::Instance(i) => format!("{{\"instance\":{}}}", instance(i)),
            Item::Repeat {
                structure: st,
                instances,
            } => {
                let list: Vec<String> = instances.iter().map(instance).collect();
                format!(
                    "{{\"repeat\":{{\"structure\":{},\"instances\":[{}]}}}}",
                    structure(st),
                    list.join(",")
                )
            }
        })
        .collect();
    format!("[\n{}\n]", i.join(",\n"))
}

fn clip(c: &Clip) -> String {
    format!(
        "{{\"rule\":\"{}\",\"parent\":{},\"loops\":{}}}",
        rule(c.rule),
        opt(c.parent),
        loops(&c.loops)
    )
}

/// The model as a JSON document.
pub fn encode(model: &Model) -> String {
    let clips: Vec<String> = model.clips.iter().map(clip).collect();
    let definitions: Vec<String> = model.definitions.iter().map(|d| items(d)).collect();
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{{\"width\":{},\"height\":{},\"clip\":{},\"items\":{},\"clips\":[{}],\"definitions\":[{}]}}",
        model.width,
        model.height,
        opt(model.clip),
        items(&model.items),
        clips.join(",\n"),
        definitions.join(",\n")
    );
    s
}

/// The model dump target.
pub struct ModelDump;

impl super::Emitter for ModelDump {
    fn encode(&self, model: &Model, _: &[u8]) -> Vec<u8> {
        encode(model).into_bytes()
    }

    fn finest(&self, _: &Object) -> u8 {
        0
    }

    fn render(&self, _: &[u8], _: usize, _: usize) -> Option<Vec<glam::Vec3>> {
        None
    }
}
