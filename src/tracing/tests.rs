use std::ops::ControlFlow;

use super::*;
use crate::Pixels;

/// A picture with something for most stages: a ground ramp, discs, a ring, a thin line and a
/// translucent band over them, as RGBA.
fn picture(size: usize) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(4 * size * size);
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
            rgba.extend(c.map(|v| (v.clamp(0., 1.) * 255.).round() as u8));
            rgba.push(255);
        }
    }
    rgba
}

fn image(rgba: &[u8], size: usize) -> Image<'_> {
    Image {
        pixels: Pixels::Rgba8(rgba),
        width: size,
        height: size,
    }
}

/// A callback that ends the slice at every `n`th checkpoint, counting them all in `asks`.
fn every(n: usize, asks: &mut usize) -> impl FnMut(&Progress) -> ControlFlow<()> + '_ {
    move |_| {
        *asks += 1;
        if (*asks).is_multiple_of(n) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Steps `tracing` to its end, each slice ended by [`every`] `n`th checkpoint; the model and
/// the number of slices.
fn sliced(mut tracing: Tracing, n: usize) -> (Model, usize) {
    let (mut asks, mut slices) = (0, 0);
    loop {
        slices += 1;
        if let Step::Done(model) = tracing.step(every(n, &mut asks)) {
            return (model, slices);
        }
    }
}

/// A trace resumed slice after slice ends with the model of one uninterrupted trace, however
/// short the slices; a slice ends exactly where its callback breaks.
#[test]
fn a_resumed_trace_ends_as_one_call() {
    let size = 72;
    let rgba = picture(size);
    let options = TraceOptions::default();
    let whole = format!("{:?}", trace(&image(&rgba, size), &options).unwrap());
    let mut asks = 0;
    trace_with(&image(&rgba, size), &options, |_| asks += 1).unwrap();
    assert!(asks > 100, "{asks} checkpoints");
    for n in [1, 7, 97, asks / 3] {
        let tracing = Tracing::new(&image(&rgba, size), &options).unwrap();
        let (model, slices) = sliced(tracing, n);
        assert_eq!(format!("{model:?}"), whole, "slices of {n} checkpoints");
        assert_eq!(slices, asks / n + 1, "slices of {n} checkpoints");
    }
}

/// A trace paused where it shows its scene hands the host that scene, as the model the
/// finished trace would be of it: at the image's size, lowered to the host.
#[test]
fn a_paused_trace_shows_its_scene() {
    let size = 72;
    let rgba = picture(size);
    let options = TraceOptions {
        max_pixels: Some(size * size / 4),
        host: crate::Capabilities::NONE,
        ..TraceOptions::default()
    };
    let mut tracing = Tracing::new(&image(&rgba, size), &options).unwrap();
    let mut shown = 0;
    let shows = |p: &Progress| match p.scene {
        Some(_) => ControlFlow::Break(()),
        None => ControlFlow::Continue(()),
    };
    while let Step::Paused(p) = tracing.step(shows) {
        let model = p.model().expect("the scene it paused at");
        assert_eq!((model.width, model.height), (size, size));
        shown += 1;
    }
    assert!(shown > 3, "{shown} scenes shown");
}

/// A trace dropped between slices is cancelled, wherever it is, and leaves nothing behind
/// that changes the next trace; a callback that breaks makes `trace_with` return the
/// cancellation, asking nothing more.
#[test]
fn a_dropped_trace_is_cancelled() {
    let size = 72;
    let rgba = picture(size);
    let options = TraceOptions::default();
    let whole = format!("{:?}", trace(&image(&rgba, size), &options).unwrap());
    let mut asks = 0;
    trace_with(&image(&rgba, size), &options, |_| asks += 1).unwrap();
    // Every checkpoint early on, then spread over the rest
    for at in (1..20).chain((1..40).map(|k| k * asks / 40)) {
        let mut tracing = Tracing::new(&image(&rgba, size), &options).unwrap();
        let mut seen = 0;
        assert!(matches!(tracing.step(every(at, &mut seen)), Step::Paused(_)));
        drop(tracing);
        let (mut seen, mut after) = (0, 0);
        let cancelled = trace_with(&image(&rgba, size), &options, |_| {
            seen += 1;
            after += usize::from(seen > at);
            if seen >= at {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        assert_eq!(cancelled.err(), Some(TraceError::Cancelled));
        assert_eq!(after, 0, "asked again after cancelling at {at} of {asks}");
    }
    let again = trace(&image(&rgba, size), &options).unwrap();
    assert_eq!(format!("{again:?}"), whole);
}

/// Two traces stepped in turn on one thread end as each does alone, and a trace started by
/// a callback runs as its own, the outer one going on after it.
#[test]
fn traces_interleave_and_nest() {
    let (a, b) = (picture(72), picture(56));
    let options = TraceOptions::default();
    let alone = |rgba: &[u8], size| format!("{:?}", trace(&image(rgba, size), &options).unwrap());
    let (want_a, want_b) = (alone(&a, 72), alone(&b, 56));
    let mut ta = Some(Tracing::new(&image(&a, 72), &options).unwrap());
    let mut tb = Some(Tracing::new(&image(&b, 56), &options).unwrap());
    let (mut got_a, mut got_b) = (None, None);
    let (mut asks_a, mut asks_b) = (0, 0);
    while ta.is_some() || tb.is_some() {
        if let Some(t) = &mut ta
            && let Step::Done(model) = t.step(every(13, &mut asks_a))
        {
            got_a = Some(format!("{model:?}"));
            ta = None;
        }
        if let Some(t) = &mut tb
            && let Step::Done(model) = t.step(every(5, &mut asks_b))
        {
            got_b = Some(format!("{model:?}"));
            tb = None;
        }
    }
    assert_eq!(got_a.as_deref(), Some(want_a.as_str()));
    assert_eq!(got_b.as_deref(), Some(want_b.as_str()));
    let mut inner = None;
    let outer = trace_with(&image(&a, 72), &options, |_| {
        if inner.is_none() {
            inner = Some(alone(&b, 56));
        }
    })
    .unwrap();
    assert_eq!(format!("{outer:?}"), want_a);
    assert_eq!(inner.as_deref(), Some(want_b.as_str()));
}

/// Each stage switched off still traces, and what it alone draws is gone: no stroke without
/// the strokes, no instance without instancing, no layer group without translucency.
#[test]
fn every_stage_can_be_switched_off() {
    use crate::model::{Geometry, Item};
    let size = 72;
    let rgba = picture(size);
    let all = crate::Stages::ALL;
    for stages in [
        crate::Stages { translucency: false, ..all },
        crate::Stages { strokes: false, ..all },
        crate::Stages { primitives: false, ..all },
        crate::Stages { effects: false, ..all },
        crate::Stages { instancing: false, ..all },
    ] {
        let options = TraceOptions {
            stages,
            ..TraceOptions::default()
        };
        let model = trace(&image(&rgba, size), &options).expect("a trace");
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
