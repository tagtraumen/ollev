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
        if let Step::Done(model) = tracing.step(every(n, &mut asks)).unwrap() {
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
    while let Step::Paused(p) = tracing.step(shows).unwrap() {
        let model = p.model().expect("the scene it paused at");
        assert_eq!((model.width, model.height), (size, size));
        shown += 1;
    }
    assert!(shown > 3, "{shown} scenes shown");
}

/// A host asking for the model at every pause gets a first look before the trace's own first
/// scene, then each kept scene lowered to it, and in the end the model of a trace it never
/// asked.
#[test]
fn partial_models_leave_the_result_alone() {
    let size = 128;
    let rgba = picture(size);
    let graphite = crate::Capabilities {
        even_odd: false,
        path_gradients: false,
        blur: false,
        radial_focus: false,
        ..crate::Capabilities::ALL
    };
    for host in [graphite, crate::Capabilities::NONE] {
        let options = TraceOptions {
            host,
            ..TraceOptions::default()
        };
        let whole = format!("{:?}", trace(&image(&rgba, size), &options).unwrap());
        let mut tracing = Tracing::new(&image(&rgba, size), &options).unwrap();
        let (mut asks, mut revision, mut fresh) = (0, 0, false);
        let mut first_look = None;
        let model = loop {
            let step = tracing.step(|p: &Progress| {
                fresh |= p.scene.is_some();
                every(5, &mut asks)(p)
            }).unwrap();
            let p = match step {
                Step::Paused(p) => p,
                Step::Done(model) => break model,
            };
            assert!(p.revision() >= revision, "revisions only grow");
            revision = p.revision();
            if let Some(model) = p.model() {
                first_look.get_or_insert(!fresh);
                assert_eq!((model.width, model.height), (size, size));
                let even_odd = model.objects().iter().any(|o| {
                    matches!(o.geometry, crate::model::Geometry::Fill { rule, .. }
                        if rule == crate::model::FillRule::EvenOdd)
                });
                assert!(!even_odd, "a partial model in the host's form");
            }
        };
        assert_eq!(first_look, Some(true), "a first look before the first scene");
        assert!(revision > 3, "{revision} scenes kept");
        assert_eq!(format!("{model:?}"), whole);
    }
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
        assert!(matches!(tracing.step(every(at, &mut seen)).unwrap(), Step::Paused(_)));
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
            && let Step::Done(model) = t.step(every(13, &mut asks_a)).unwrap()
        {
            got_a = Some(format!("{model:?}"));
            ta = None;
        }
        if let Some(t) = &mut tb
            && let Step::Done(model) = t.step(every(5, &mut asks_b)).unwrap()
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
/// the strokes, no instance without instancing, no layer group without translucency, no
/// gradient along a curve without the path gradients.
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
        crate::Stages { path_gradients: false, ..all },
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
        let along = model.objects().iter().any(|o| {
            matches!(&o.paint, crate::model::Paint::Gradient(g)
                if matches!(g.shape(), crate::model::GradientShape::Path { .. }))
        });
        assert!(stages.path_gradients || !along, "{stages:?}");
    }
}

/// The whole trace's progress never goes back and runs to near the end, with every stage;
/// it moves in many small steps, not only at the stages' starts, and each stage's own
/// fraction stays within it.
#[test]
fn the_whole_trace_progress_never_goes_back() {
    let size = 96;
    let rgba = picture(size);
    for stages in [crate::Stages::DEFAULT, crate::Stages::ALL] {
        let options = TraceOptions {
            stages,
            ..TraceOptions::default()
        };
        let mut seen: Vec<f64> = Vec::new();
        let model = trace_with(&image(&rgba, size), &options, |p: &Progress| {
            assert!((0. ..=1.).contains(&p.fraction), "{} at {}", p.stage, p.fraction);
            seen.push(p.overall);
        })
        .unwrap();
        assert!(!model.objects().is_empty());
        assert!(seen.windows(2).all(|w| w[0] <= w[1]), "{stages:?}: progress went back");
        let last = seen.last().copied().unwrap_or_default();
        assert!(last > 0.9 && last <= 1., "{stages:?}: the trace ended at {last}");
        let mut steps = seen.clone();
        steps.dedup();
        assert!(steps.len() > 100, "{stages:?}: {} distinct points", steps.len());
        let jump = seen.windows(2).map(|w| w[1] - w[0]).fold(0., f64::max);
        assert!(jump < 0.25, "{stages:?}: a jump of {jump}");
    }
}

/// The host is shown the scene as the stages after the drawing change it, not only as
/// shapes are added.
#[test]
fn the_scene_shows_as_it_is_redrawn() {
    let size = 96;
    let rgba = picture(size);
    let mut revision = 0;
    let mut stages = Vec::new();
    trace_with(&image(&rgba, size), &TraceOptions::default(), |p: &Progress| {
        if p.revision() != revision {
            revision = p.revision();
            stages.push(p.stage);
        }
    })
    .unwrap();
    assert!(stages.contains(&"draw"), "{stages:?}");
    assert!(
        stages.iter().any(|s| ["redraw", "shapes", "passes"].contains(s)),
        "{stages:?}"
    );
}

/// With `clip_to_image` every object lies within the picture or is clipped to it; without, a
/// shape may reach past the edge.
#[test]
fn the_model_is_clipped_to_the_image() {
    let size = 96;
    // a disc cut by the picture's corner, which the fit completes past it
    let mut rgba = Vec::new();
    for y in 0..size {
        for x in 0..size {
            let (dx, dy) = (x as f64 - 8., y as f64 - 8.);
            let disc = dx * dx + dy * dy < 30. * 30.;
            rgba.extend(if disc { [200, 40, 40, 255] } else { [240, 240, 230, 255] });
        }
    }
    let within = |model: &Model| {
        model.objects().iter().all(|o| {
            o.clip.is_some()
                || match &o.geometry {
                    crate::model::Geometry::Fill { loops, .. } => {
                        loops.iter().flatten().flatten().all(|p| {
                            (-1e-6..=size as f64 + 1e-6).contains(&p.x)
                                && (-1e-6..=size as f64 + 1e-6).contains(&p.y)
                        })
                    }
                    crate::model::Geometry::Stroke(_) => true,
                }
        })
    };
    let clipped = trace(&image(&rgba, size), &TraceOptions::default()).unwrap();
    assert!(within(&clipped));
    let options = TraceOptions {
        clip_to_image: false,
        ..TraceOptions::default()
    };
    let free = trace(&image(&rgba, size), &options).unwrap();
    assert!(!within(&free), "the disc stays within the picture unclipped");
}
