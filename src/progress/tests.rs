use std::ops::ControlFlow;

use super::*;
use crate::plan;

/// A picture with something for most stages: a ground ramp, discs, a ring, a thin line and a
/// translucent band over them.
fn picture(size: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(3 * size * size);
    for y in 0..size {
        for x in 0..size {
            let (fx, fy) = (x as f32 / size as f32, y as f32 / size as f32);
            let mut c = [0.2 + 0.6 * fx, 0.5, 0.8 - 0.4 * fy];
            let d = |cx: f32, cy: f32| ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
            if d(0.3, 0.3) < 0.15 {
                c = [0.9, 0.1, 0.1];
            }
            if (0.12..0.2).contains(&d(0.7, 0.65)) {
                c = [0.1, 0.1, 0.4];
            }
            if (fx - fy).abs() < 0.01 {
                c = [0.05, 0.05, 0.05];
            }
            if (0.45..0.7).contains(&fy) {
                c = c.map(|v| 0.6 * v + 0.4 * 0.95);
            }
            rgb.extend(c.map(|v| (v.clamp(0., 1.) * 255.).round() as u8));
        }
    }
    rgb
}

fn trace(rgb: &[u8], size: usize, callback: impl FnMut(&Progress) -> ControlFlow<()>) -> Scene {
    let detail = crate::tracer::auto_detail(rgb, size, size);
    plan::trace(rgb, size, size, detail, plan::Options::default(), callback)
}

/// A trace cancelled at any ask returns, without a panic, and asks nothing more of the host;
/// one never cancelled hears every stage and ends where it would without a callback.
#[test]
fn a_trace_stops_wherever_it_is_cancelled() {
    let size = 72;
    let rgb = picture(size);
    let mut asks = 0usize;
    let mut stages = Vec::new();
    let whole = trace(&rgb, size, |p| {
        asks += 1;
        if stages.last() != Some(&p.stage) {
            stages.push(p.stage);
        }
        ControlFlow::Continue(())
    });
    let plain = trace(&rgb, size, |_| ControlFlow::Continue(()));
    assert_eq!(
        format!("{:?}", whole.model()),
        format!("{:?}", plain.model())
    );
    for stage in ["analyze", "plan", "draw", "shapes", "passes"] {
        assert!(stages.contains(&stage), "{stage} in {stages:?}");
    }
    assert!(asks > 100, "{asks} asks");
    // Every ask early on, then spread over the rest
    let at: Vec<usize> = (1..20).chain((1..40).map(|k| k * asks / 40)).collect();
    for n in at {
        let (mut seen, mut after) = (0usize, 0usize);
        trace(&rgb, size, |_| {
            seen += 1;
            if seen > n {
                after += 1;
            }
            if seen >= n {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        assert_eq!(after, 0, "asked again after cancelling at {n} of {asks}");
    }
}

/// A scope says whether its callback cancelled, and an inner scope leaves the outer one's
/// callback in place.
#[test]
fn scopes_nest() {
    let mut outer = 0;
    let mut cb_outer = |_: &Progress| {
        outer += 1;
        ControlFlow::Continue(())
    };
    let ((), cancelled) = scope(&mut cb_outer, || {
        poll();
        let mut cb_inner = |_: &Progress| ControlFlow::Break(());
        let (inner_stopped, inner_cancelled) = scope(&mut cb_inner, poll);
        assert!(inner_stopped && inner_cancelled);
        assert!(!cancelled());
        poll();
    });
    assert!(!cancelled);
    assert_eq!(outer, 2);
}

/// Each stage switched off still traces, and what it alone draws is gone: no stroke without
/// the strokes, no instance without instancing, no layer group without translucency.
#[test]
fn every_stage_can_be_switched_off() {
    use crate::model::{Geometry, Item};
    let size = 72;
    let rgba: Vec<u8> = picture(size)
        .chunks(3)
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect();
    let image = crate::Image {
        pixels: crate::Pixels::Rgba8(&rgba),
        width: size,
        height: size,
    };
    let all = crate::Stages::ALL;
    for stages in [
        crate::Stages { translucency: false, ..all },
        crate::Stages { strokes: false, ..all },
        crate::Stages { primitives: false, ..all },
        crate::Stages { effects: false, ..all },
        crate::Stages { instancing: false, ..all },
    ] {
        let options = crate::TraceOptions {
            stages,
            ..crate::TraceOptions::default()
        };
        let model = crate::trace(&image, &options).expect("a trace");
        assert!(!model.objects().is_empty(), "{stages:?}");
        let strokes = model
            .objects()
            .iter()
            .any(|o| matches!(o.geometry, Geometry::Stroke(_)));
        assert!(stages.strokes || !strokes, "{stages:?}");
        let instanced = model
            .items
            .iter()
            .any(|i| matches!(i, Item::Instance(_) | Item::Repeat { .. }));
        assert!(stages.instancing || !instanced, "{stages:?}");
        let grouped = model.items.iter().any(|i| matches!(i, Item::Group { .. }));
        assert!(stages.translucency || !grouped, "{stages:?}");
    }
}
