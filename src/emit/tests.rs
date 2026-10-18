use crate::model::{Fidelity, Object};
use crate::{Fill, Scene, Shape};
use glam::{Vec2, Vec3};

use super::*;

fn square(a: Vec2, b: Vec2) -> Vec<Vec<[Vec2; 4]>> {
    let corners = [a, Vec2::new(b.x, a.y), b, Vec2::new(a.x, b.y)];
    vec![
        (0..4)
            .map(|k| {
                let (p, q) = (corners[k], corners[(k + 1) % 4]);
                [p, p.lerp(q, 1. / 3.), p.lerp(q, 2. / 3.), q]
            })
            .collect(),
    ]
}

/// A layer's shapes and marks are written as its group: without the group's two tags, the
/// SVG is the one written with no layer.
#[test]
fn a_layer_is_written_as_its_group() {
    let mut scene = Scene::with_background(8, 8, Vec3::ONE);
    scene.push(Shape::new(square(Vec2::ZERO, Vec2::splat(4.)), Vec3::X));
    let layer = Fill {
        opacity: 0.4,
        ..Fill::solid(Vec3::Y)
    };
    scene.push(Shape {
        layer: Some(0),
        ..Shape::new(square(Vec2::splat(2.), Vec2::splat(8.)), layer)
    });
    let tint = Shape::new(square(Vec2::splat(3.), Vec2::splat(6.)), Vec3::ZERO);
    scene.marks = vec![Object {
        opacity: 0.5,
        ..Object::of_shape(&tint, None)
    }];
    scene.marks_layer = Some(0);
    let grouped = svg_of(&scene);
    let mut flat = scene.clone();
    flat.shapes.iter_mut().for_each(|s| s.layer = None);
    flat.marks_layer = None;
    let flat = svg_of(&flat);
    let lines: Vec<&str> = grouped.lines().collect();
    let open = lines.iter().position(|l| *l == "<g id=\"layer-0\">");
    let close = lines.iter().position(|l| *l == "</g>");
    assert!(
        matches!((open, close), (Some(o), Some(c)) if c == o + 3),
        "{grouped}"
    );
    assert!(lines[open.unwrap() + 2].contains("fill-opacity=\"0.500\""));
    let mut a: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|l| !l.starts_with("<g id=") && *l != "</g>")
        .collect();
    let mut b: Vec<&str> = flat.lines().collect();
    assert_eq!(a.len(), b.len());
    a.sort();
    b.sort();
    assert_eq!(a, b);
    assert!(!flat.contains("<g "));
}

/// A blurred clipped shape is written as a filter before its clip, and the SVG renders as
/// the model does: the verify loop finds nothing to step.
#[test]
fn a_blurred_clipped_shape_verifies() {
    let mut scene = Scene::with_background(24, 24, Vec3::ONE);
    let parent = scene.push(Shape::new(
        square(Vec2::splat(4.), Vec2::splat(20.)),
        Vec3::ZERO,
    ));
    let child = Shape::new(square(Vec2::new(2., 2.), Vec2::new(12., 22.)), Vec3::X);
    scene.push(Shape {
        clip: Some(parent),
        ..child.with_blur(2.)
    });
    scene.fidelity = Some(Fidelity {
        tolerance: None,
        segment_price: 1e-3,
    });
    let out = compile(&scene.model(), &Svg);
    let svg = String::from_utf8(out.bytes).unwrap();
    let line = svg
        .lines()
        .find(|l| l.contains("filter="))
        .expect("the blurred path");
    assert!(line.contains("clip-path=\"url(#c0)\""), "{line}");
    assert!(svg.contains("feGaussianBlur stdDeviation=\"2.000\""));
    assert!(out.report.checked);
    assert_eq!(out.report.left, 0, "{}", out.report);
}

/// SVG that loses the square (the model's second object) at the first level.
struct Lossy;

impl Emitter for Lossy {
    fn encode(&self, model: &crate::model::Model, levels: &[u8]) -> Vec<u8> {
        let mut model = model.clone();
        if levels.get(1) == Some(&0)
            && let crate::model::Item::Object(o) = &mut model.items[1]
        {
            o.opacity = 0.;
        }
        svg::encode(&model, levels).into_bytes()
    }

    fn finest(&self, _: &Object) -> u8 {
        svg::FINEST
    }

    fn render(&self, bytes: &[u8], width: usize, height: usize) -> Option<Vec<Vec3>> {
        Svg.render(bytes, width, height)
    }
}

/// An object whose first encoding misses steps to a finer one, and only that object.
#[test]
fn a_missing_object_steps_finer() {
    let mut scene = Scene::with_background(16, 16, Vec3::ONE);
    scene.push(Shape::new(
        square(Vec2::splat(4.), Vec2::splat(12.)),
        Vec3::ZERO,
    ));
    scene.fidelity = Some(Fidelity {
        tolerance: None,
        segment_price: 1.,
    });
    let out = compile(&scene.model(), &Lossy);
    assert_eq!(out.levels, vec![0, 1], "{}", out.report);
    assert_eq!((out.report.missed, out.report.left), (1, 0));
    let plain = compile(&scene.model(), &Svg);
    assert_eq!(plain.levels, vec![0, 0], "{}", plain.report);
}

/// The dump holds every object.
#[test]
fn the_dump_holds_every_object() {
    let mut scene = Scene::with_background(8, 8, Vec3::ONE);
    scene.push(Shape::new(square(Vec2::ZERO, Vec2::splat(4.)), Vec3::X));
    let json = dump::encode(&scene.model());
    assert_eq!(json.matches("\"object\"").count(), 2);
    assert!(json.contains("\"solid\":[1,0,0]"));
}

/// A linear repeat is written as one group of uses of its definition, renders as the model
/// does, and counts as one object plus its definition.
#[test]
fn a_repeat_is_one_group_of_uses() {
    use crate::instancing::{Repeat, Structure, Transform};
    let mut scene = Scene::with_background(32, 12, Vec3::ONE);
    let prototype = Shape::new(square(Vec2::splat(-3.), Vec2::splat(3.)), Vec3::X);
    let mut members = Vec::new();
    for k in 0..3 {
        let t = Transform::translation(Vec2::new(6. + 10. * k as f32, 6.));
        let id = scene.push(crate::instancing::transformed(&prototype, t));
        members.push((id, t));
    }
    scene.repeats = vec![Repeat {
        prototype,
        members,
        structure: Structure::Linear {
            step: Vec2::new(10., 0.),
        },
    }];
    scene.fidelity = Some(Fidelity {
        tolerance: None,
        segment_price: 1e-3,
    });
    let model = scene.model();
    assert_eq!(model.count().0, 3);
    let out = compile(&model, &Svg);
    let svg = String::from_utf8(out.bytes).unwrap();
    assert_eq!(svg.matches("<use ").count(), 3, "{svg}");
    assert!(svg.contains("data-repeat=\"linear\""), "{svg}");
    assert!(out.report.checked);
    assert_eq!(out.report.left, 0, "{}", out.report);
}

/// A circle's path gradient, drawn as 32 exact bands, renders as the model does to within the
/// banding; the blurred levels are coarser and smaller.
#[test]
fn a_path_gradient_draws_as_bands() {
    let (c, r) = (Vec2::splat(24.), 10f32);
    let k = 0.552_284_8 * r;
    let p = [Vec2::X, Vec2::Y, -Vec2::X, -Vec2::Y].map(|d| c + d * r);
    let t = [Vec2::Y, -Vec2::X, -Vec2::Y, Vec2::X];
    let circle: Vec<[Vec2; 4]> = (0..4)
        .map(|i| {
            let j = (i + 1) % 4;
            [p[i], p[i] + t[i] * k, p[j] - t[j] * k, p[j]]
        })
        .collect();
    let gradient = crate::render::Gradient::Path {
        contour: crate::render::distance::Contour::new(&circle).unwrap(),
        near: -8.,
        far: 12.,
        to: [0.1, 0.2, 0.6, 1.],
        stops: crate::render::Stops::opaque(&[(0.4, [0.9, 0.8, 0.2])]),
    };
    let mut scene = Scene::with_background(48, 48, Vec3::ONE);
    scene.push(Shape::new(
        square(Vec2::splat(2.), Vec2::splat(46.)),
        Fill {
            color: Vec3::new(0.8, 0.1, 0.1),
            gradient: Some(gradient),
            opacity: 1.,
        },
    ));
    let model = scene.model();
    let (truth, _) = model.render();
    let psnr = |level: u8| {
        let svg = svg::encode(&model, &[0, level]);
        let drawn = verify::render_svg(&svg, 48, 48).unwrap();
        let sse: f32 = (0..48 * 48)
            .map(|i| (drawn[i] - truth[i]).length_squared() / 3.)
            .sum();
        (10. * (48. * 48. / sse).log10(), svg.len())
    };
    let (exact, big) = psnr(3);
    let (coarse, small) = psnr(1);
    assert!(exact > 45., "{exact}");
    assert!(coarse < exact && small < big, "{coarse} {small} {big}");
}

/// Glows fading to transparent over a two-colour ground, an elliptical one with a translucent stop
/// of another colour and a path gradient's, render in SVG as the model does.
#[test]
fn glows_with_transparent_stops() {
    let (c, r) = (Vec2::splat(24.), 10f32);
    let k = 0.552_284_8 * r;
    let p = [Vec2::X, Vec2::Y, -Vec2::X, -Vec2::Y].map(|d| c + d * r);
    let t = [Vec2::Y, -Vec2::X, -Vec2::Y, Vec2::X];
    let circle: Vec<[Vec2; 4]> = (0..4)
        .map(|i| {
            let j = (i + 1) % 4;
            [p[i], p[i] + t[i] * k, p[j] - t[j] * k, p[j]]
        })
        .collect();
    let stops = crate::render::Stops::new(&[(0.5, [0.9, 0.4, 0.2, 0.6])]);
    let radial = crate::render::Gradient::Radial {
        center: c,
        focus: Vec2::ZERO,
        radius: 20.,
        transform: glam::Mat2::from_angle(0.5)
            * glam::Mat2::from_diagonal(glam::Vec2::new(1., 0.6)),
        to: [0.9, 0.4, 0.2, 0.],
        stops,
    };
    let path = crate::render::Gradient::Path {
        contour: crate::render::distance::Contour::new(&circle).unwrap(),
        near: -8.,
        far: 12.,
        to: [0.9, 0.4, 0.2, 0.],
        stops,
    };
    for gradient in [radial, path] {
        let mut scene = Scene::with_background(48, 48, Vec3::new(0.1, 0.1, 0.2));
        scene.push(Shape::new(
            square(Vec2::new(0., 24.), Vec2::new(48., 48.)),
            Vec3::new(0.3, 0.5, 0.3),
        ));
        scene.push(Shape::new(
            square(Vec2::splat(2.), Vec2::splat(46.)),
            Fill {
                color: Vec3::new(1., 0.9, 0.5),
                gradient: Some(gradient),
                opacity: 1.,
            },
        ));
        let model = scene.model();
        let (truth, _) = model.render();
        let svg = svg::encode(&model, &[0, 0, 3]);
        let drawn = verify::render_svg(&svg, 48, 48).unwrap();
        let sse: f32 = (0..48 * 48)
            .map(|i| (drawn[i] - truth[i]).length_squared() / 3.)
            .sum();
        let psnr = 10. * (48. * 48. / sse).log10();
        assert!(psnr > 40., "{psnr}\n{svg}");
    }
}

/// A dash offset written onto a dash's end moves into the gap, so no zero-length rest of the
/// dash is drawn as a dot of its caps; elsewhere, and at a zero-length dash, it is as written.
#[test]
fn a_dash_offset_on_a_dash_end_moves_into_the_gap() {
    let array = [27.45, 11.13, 121.91, 10.91, 21.71, 11.79];
    let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
    assert!(near(svg::dash_offset(&array, 193.112, 2), 193.12));
    assert!(near(svg::dash_offset(&array, 193.11 + 204.9, 2), 398.02));
    assert!(near(svg::dash_offset(&array, 195.481, 2), 195.48));
    assert!(near(svg::dash_offset(&[0., 5.], 0., 2), 0.));
}

/// A dotted stroke of dashes of no length under round caps: the model draws each dot as a
/// disc, and the SVG renders the same dots.
#[test]
fn dashes_of_no_length_are_round_dots() {
    use crate::stroke::{Cap, Dashes, Join, Stroke};
    let (a, b) = (Vec2::new(6., 16.), Vec2::new(42., 16.));
    let stroke = Stroke {
        centerline: vec![[a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.), b]],
        closed: false,
        width: 4.,
        cap: Cap::Round,
        join: Join::Round,
        corners: vec![false],
        dashes: Some(Dashes {
            array: vec![0., 8.],
            offset: 0.,
        }),
    };
    let mut scene = Scene::with_background(48, 32, Vec3::ONE);
    let shape = stroke.shape(Fill::solid(Vec3::new(0.8, 0.1, 0.1)));
    assert_eq!(shape.loops.len(), 5);
    scene.push(shape);
    let model = scene.model();
    let (truth, _) = model.render();
    let svg = svg::encode(&model, &[0, 0]);
    assert!(svg.contains("stroke-dasharray=\"0.00 8.00\""), "{svg}");
    let drawn = verify::render_svg(&svg, 48, 32).unwrap();
    let sse: f32 = (0..48 * 32)
        .map(|i| (drawn[i] - truth[i]).length_squared() / 3.)
        .sum();
    let psnr = 10. * (48. * 32. / sse).log10();
    assert!(psnr > 35., "{psnr}\n{svg}");
}
