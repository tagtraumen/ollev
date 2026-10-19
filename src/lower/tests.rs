use crate::render::{Gradient as Ramp, Stops, distance::Contour};
use glam::{DVec2, DVec3, Mat2, Vec2, Vec3};

use super::*;
use crate::model::{
    Cap, Curve, Fidelity, FillRule, Geometry, Gradient, GradientShape, Join, Paint, StrokeGeometry,
};
use crate::translucency::raster::NOISE;

fn rect(a: (f64, f64), b: (f64, f64)) -> Vec<Curve> {
    let (a, b) = (DVec2::new(a.0, a.1), DVec2::new(b.0, b.1));
    let c = [a, DVec2::new(b.x, a.y), b, DVec2::new(a.x, b.y)];
    (0..4).map(|k| bands::line(c[k], c[(k + 1) % 4])).collect()
}

fn fill(loops: Vec<Vec<Curve>>, paint: Paint) -> Object {
    Object {
        geometry: Geometry::Fill {
            loops,
            rule: FillRule::EvenOdd,
        },
        paint,
        opacity: 1.,
        blur: 0.,
        clip: None,
    }
}

fn solid(r: f64, g: f64, b: f64) -> Paint {
    Paint::Solid(DVec3::new(r, g, b))
}

/// A model with one object of each feature a host may lack, at the price a trace of its size
/// accepts a segment by.
fn model() -> Model {
    let (w, h) = (96, 64);
    let to32 =
        |l: Vec<Curve>| -> Vec<[Vec2; 4]> { l.iter().map(|c| c.map(glam::DVec2::as_vec2)).collect() };
    let radial = |focus: Vec2, to: [f32; 4], stops: Stops, center: Vec2, radius: f32| {
        Paint::Gradient(Gradient::new(
            Vec3::new(1., 0.85, 0.4),
            Ramp::Radial {
                center,
                focus,
                radius,
                transform: Mat2::IDENTITY,
                to,
                stops,
            },
        ))
    };
    let contour = Contour::new(&to32(rect((70., 10.), (88., 26.)))).unwrap();
    let path = Paint::Gradient(Gradient::new(
        Vec3::new(0.9, 0.3, 0.2),
        Ramp::Path {
            contour,
            near: -5.,
            far: 6.,
            to: [0.2, 0.3, 0.9, 1.],
            stops: Stops::opaque(&[(0.5, [0.95, 0.9, 0.3])]),
        },
    ));
    let items = vec![
        fill(vec![rect((0., 0.), (96., 64.))], solid(0.9, 0.9, 0.85)),
        // a ring: a square with a square hole
        fill(
            vec![rect((6., 6.), (26., 26.)), rect((11., 11.), (21., 21.))],
            solid(0.1, 0.15, 0.5),
        ),
        fill(
            vec![rect((34., 6.), (58., 30.))],
            radial(
                Vec2::new(5., -4.),
                [0.1, 0.2, 0.6, 1.],
                Stops::opaque(&[(0.4, [0.8, 0.4, 0.3])]),
                Vec2::new(46., 18.),
                12.,
            ),
        ),
        Object {
            blur: 2.,
            ..fill(vec![rect((8., 36.), (24., 52.))], solid(0.8, 0.1, 0.1))
        },
        // a glow fading to transparent
        fill(
            vec![rect((30., 33.), (62., 62.))],
            radial(
                Vec2::ZERO,
                [1., 0.85, 0.4, 0.],
                Stops::NONE,
                Vec2::new(46., 47.),
                14.,
            ),
        ),
        Object {
            clip: Some(0),
            ..fill(vec![rect((28., 44.), (60., 50.))], solid(0.1, 0.6, 0.2))
        },
        fill(vec![rect((66., 6.), (92., 30.))], path),
        Object {
            geometry: Geometry::Stroke(StrokeGeometry {
                centerline: vec![[
                    DVec2::new(66., 40.),
                    DVec2::new(76., 34.),
                    DVec2::new(84., 62.),
                    DVec2::new(93., 56.),
                ]],
                closed: false,
                width: 2.5,
                cap: Cap::Round,
                join: Join::Round,
                corners: vec![false],
                dashes: Some((vec![5., 3.], 1.)),
            }),
            ..fill(Vec::new(), solid(0.2, 0.2, 0.2))
        },
    ];
    let price = crate::Context::new(&vec![255; 3 * w * h], w, h, 1.).segment_cost;
    Model {
        width: w,
        height: h,
        items: items
            .into_iter()
            .map(|o| Item::Object(Box::new(o)))
            .collect(),
        clips: vec![Clip {
            loops: vec![rect((38., 38.), (54., 56.))],
            rule: FillRule::EvenOdd,
            parent: None,
        }],
        definitions: Vec::new(),
        fidelity: Some(Fidelity {
            tolerance: None,
            segment_price: price,
        }),
        clip: None,
    }
}

/// Whether the model uses a feature the host lacks.
fn uses(model: &Model, host: Capabilities) -> Vec<&'static str> {
    let mut found = Vec::new();
    for o in model.objects() {
        if let Geometry::Fill {
            rule: FillRule::EvenOdd,
            ..
        } = o.geometry
            && !host.even_odd
        {
            found.push("even-odd");
        }
        if let Paint::Gradient(g) = &o.paint {
            match g.shape() {
                GradientShape::Path { .. } if !host.path_gradients => found.push("path gradient"),
                GradientShape::Radial { focus, .. }
                    if focus != DVec2::ZERO && !host.radial_focus =>
                {
                    found.push("focus");
                }
                _ => {}
            }
            if !g.is_opaque() && !host.transparent_stops {
                found.push("transparent stop");
            }
        }
        if o.blur > 0. && !host.blur {
            found.push("blur");
        }
        if o.clip.is_some() && !host.clips {
            found.push("clip");
        }
        if let Geometry::Stroke(s) = &o.geometry
            && s.dashes.is_some()
            && !host.dashes
        {
            found.push("dashes");
        }
    }
    if !host.even_odd && model.clips.iter().any(|c| c.rule == FillRule::EvenOdd) {
        found.push("even-odd clip");
    }
    found
}

/// The verify loop's miss of the whole drawing: per pixel the colour distance beyond the
/// range the model takes over its 3 x 3 neighbourhood and beyond the noise, squared, summed.
fn excess(model: &Model, lowered: &Model) -> f64 {
    let (w, h) = (model.width, model.height);
    let (want, got) = (model.render().0, lowered.render().0);
    let mut sum = 0.;
    for y in 0..h {
        for x in 0..w {
            let (mut lo, mut hi) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    (lo, hi) = (lo.min(want[yy * w + xx]), hi.max(want[yy * w + xx]));
                }
            }
            let e = got[y * w + x];
            let over = ((e.clamp(lo, hi) - e).length() - NOISE as f32).max(0.) as f64;
            sum += over * over;
        }
    }
    sum
}

/// Each capability off, alone and all together: the lowered model uses none of what the host
/// lacks, and renders within a segment's price of the model per object it rewrote.
#[test]
fn each_capability_off_renders_as_the_model() {
    let model = model();
    let price = model.fidelity.as_ref().unwrap().segment_price;
    let off = |f: fn(&mut Capabilities)| {
        let mut c = Capabilities::ALL;
        f(&mut c);
        c
    };
    let hosts = [
        ("even-odd", off(|c| c.even_odd = false)),
        ("path gradients", off(|c| c.path_gradients = false)),
        ("blur", off(|c| c.blur = false)),
        ("radial focus", off(|c| c.radial_focus = false)),
        ("clips", off(|c| c.clips = false)),
        ("dashes", off(|c| c.dashes = false)),
        ("transparent stops", off(|c| c.transparent_stops = false)),
        ("nothing", Capabilities::NONE),
    ];
    for (name, host) in hosts {
        let (lowered, report) = lower(&model, &host);
        assert_eq!(uses(&lowered, host), Vec::<&str>::new(), "{name}");
        let rewritten =
            report.path_gradients + report.foci + report.layered + report.dashed + report.wound;
        assert!(rewritten > 0, "{name}: nothing rewritten");
        let miss = excess(&model, &lowered);
        assert!(
            miss <= price * rewritten as f64,
            "{name}: miss {miss:.4} against {price:.4} for {rewritten} ({report})"
        );
    }
}

/// A host that draws everything gets the model as it is.
#[test]
fn a_full_host_lowers_nothing() {
    let model = model();
    let (lowered, report) = lower(&model, &Capabilities::ALL);
    assert_eq!(format!("{lowered:?}"), format!("{model:?}"));
    assert_eq!(report, Report::default());
}

/// Two squares that cross, filled even-odd, show their overlap empty; no winding does that,
/// so they are traced afresh into loops that do not cross.
#[test]
fn crossing_loops_are_traced_afresh() {
    let mut model = model();
    model.items = vec![Item::Object(Box::new(fill(
        vec![rect((-6., -4.), (40., 40.)), rect((25., 25.), (55., 50.))],
        solid(0.2, 0.3, 0.6),
    )))];
    let host = Capabilities {
        even_odd: false,
        ..Capabilities::ALL
    };
    let (lowered, report) = lower(&model, &host);
    assert_eq!(uses(&lowered, host), Vec::<&str>::new());
    let price = model.fidelity.as_ref().unwrap().segment_price;
    let miss = excess(&model, &lowered);
    assert!(miss <= price, "miss {miss} against {price}: {report}");
}

/// Through a memo, after a partial's quick lowering or the objects kept one by one, the model
/// lowers as [`lower`] lowers it, each object's clips placed after those added before; a quick
/// lowering uses none of what the host lacks either.
#[test]
fn a_memo_lowers_as_lower_does() {
    let model = model();
    let mut reversed = model.clone();
    reversed.items.reverse();
    let graphite = Capabilities {
        even_odd: false,
        path_gradients: false,
        blur: false,
        radial_focus: false,
        ..Capabilities::ALL
    };
    for host in [graphite, Capabilities::NONE] {
        let want = format!("{:?}", lower(&model, &host).0);
        let memo = std::sync::Mutex::new(Memo::default());
        let quick = lower_kept(&model, &host, &mut lock(&memo), true);
        assert_eq!(uses(&quick, host), Vec::<&str>::new());
        crate::progress::finish(keep(&model, &host, &memo));
        for _ in 0..2 {
            let kept = lower_kept(&model, &host, &mut lock(&memo), false);
            assert_eq!(format!("{kept:?}"), want);
        }
        let want = format!("{:?}", lower(&reversed, &host).0);
        for quick in [false, true] {
            let kept = lower_kept(&reversed, &host, &mut lock(&memo), quick);
            assert_eq!(format!("{kept:?}"), want);
        }
    }
}
