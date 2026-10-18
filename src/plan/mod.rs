//! The tracer: a dynamic program over the region tree decides which nodes are drawn (in the
//! colour of what they show) and which are left to their nearest drawn ancestor, by colour error
//! plus segment cost; the chosen shapes are then fitted parents first, on top of what is drawn.

use crate::tracer::Cubic;
use glam::{DVec2, DVec3, Vec2, Vec3};

use crate::cleanup::clean_in;
use crate::complete::{Completion, complete};
use crate::hierarchy::{Hierarchy, LinearFill, RadialFill, RampFill, Stats};
use crate::outline::outline_with;
use crate::recolor::recolor;
use crate::refine::{MARGIN, Place, SIMPLIFY, settle, simplify_with};
use crate::timing::Timings;
use crate::{Context, Fill, HashMap, Scene, Shape, Weights, Window};

mod fitting;
mod layers;
mod options;
mod order;
mod planner;
mod window;

use fitting::*;
use layers::*;
pub use options::*;
use order::*;
pub use planner::*;
pub use window::*;

/// Fewest segments any shape has.
pub(crate) const MIN_SEGMENTS: f64 = 2.;
/// Transition width, in pixels, from which a region's boundary is shading rather than an edge:
/// an edge is anti-aliased over one pixel.
const SOFT_EDGE: f32 = 2.;
/// Pixels of perimeter per cubic segment, for estimating what a node's outline would cost before
/// it is traced.
const PIXELS_PER_SEGMENT: f64 = 14.;
/// Pixels of image border per segment: straight runs, which one segment covers.
const FRAME_PER_SEGMENT: f64 = 64.;
/// Least spread of a region's surroundings the contrast is measured against, so flat surroundings
/// do not make every difference infinitely salient (about four 8-bit levels).
const SALIENCE_FLOOR: f64 = 0.03;
/// Rounds of replacing estimated segment counts by traced ones and planning again.
const ROUNDS: usize = 2;

/// The pixels within 1.5 px of a translucent shape's mask edge, plus those of the shapes peeled
/// above (`above`): peeling divides an anti-aliased rim by the opacity, leaving an extreme-colour
/// band that must not be traced as a sliver.
fn peeled_edge(mask: &[bool], width: usize, height: usize, above: Option<&[bool]>) -> Vec<bool> {
    use crate::translucency::raster::{Conn, dilate, length_px};
    let reach = length_px(width, 1.5);
    let outside: Vec<bool> = mask.iter().map(|&m| !m).collect();
    let grown = dilate(mask, width, height, Conn::Eight, reach);
    let shrunk = dilate(&outside, width, height, Conn::Eight, reach);
    grown
        .iter()
        .zip(&shrunk)
        .enumerate()
        .map(|(i, (&a, &b))| (a && b) || above.is_some_and(|e| e[i]))
        .collect()
}

/// Traces an RGB image at a detail level, telling `progress` the stage and the scene as it
/// grows (see [`crate::progress`]). Where `progress` breaks, the scene is what was traced so
/// far, unfinished.
pub fn trace<A: crate::progress::Answer>(
    rgb: &[u8],
    width: usize,
    height: usize,
    detail: f64,
    options: Options,
    progress: impl FnMut(&crate::progress::Progress) -> A,
) -> Scene {
    trace_with_layer(rgb, width, height, detail, options, progress).0
}

/// [`trace`], with the translucent candidate the top level accepted, if any: its region, k
/// and colour before the shape drawn for it was fitted (for `bench/examples/glass_pane.rs`).
pub fn trace_with_layer<A: crate::progress::Answer>(
    rgb: &[u8],
    width: usize,
    height: usize,
    detail: f64,
    options: Options,
    mut progress: impl FnMut(&crate::progress::Progress) -> A,
) -> (Scene, Option<crate::translucency::Candidate>) {
    let mut host = |p: &crate::progress::Progress| progress(p).flow();
    crate::progress::scope(&mut host, || traced(rgb, width, height, detail, options)).0
}

/// [`trace_with_layer`] inside the host's progress scope.
fn traced(
    rgb: &[u8],
    width: usize,
    height: usize,
    detail: f64,
    options: Options,
) -> (Scene, Option<crate::translucency::Candidate>) {
    let mut progress = |scene: &Scene| {
        crate::progress::scene(scene);
    };
    let mut timings = Timings::default();
    let start = crate::timing::Clock::now();
    let (mut ctx, hierarchy) = crate::hierarchy::analyze(rgb, width, height, detail, &mut timings);
    timings.time("uncertainty", || {
        crate::uncertainty::apply(&mut ctx, options.tolerance)
    });
    let target = ctx.clone();
    let (mut scene, layer) = trace_level_layer(
        ctx,
        hierarchy,
        detail,
        options,
        None,
        true,
        &mut timings,
        &mut progress,
    );
    if crate::progress::cancelled() {
        return (scene, layer);
    }
    // The kept layer is the shape with the highest layer index, not the scene's last shape,
    // which the level's own passes may put after it
    let top = layer.as_ref().and_then(|_| {
        scene
            .shapes
            .iter()
            .filter(|s| s.layer.is_some())
            .max_by_key(|s| s.layer)
    });
    let layer_n = top.and_then(|s| s.layer);
    crate::progress::stage("passes");
    if options.stages.strokes {
        timings.time("linework", || linework(&target, &mut scene, EDGE_TERM));
    }
    if options.stages.effects {
        effects(&target, &mut scene, &mut timings);
    }
    // Shapes after the kept layer that overlap it in its own hue join its group in the SVG; a
    // line elsewhere or of another hue (hair crossing niabot's dress) is not part of it. Found
    // by layer index, since the shape put on top by the layer decision keeps no id of its own.
    if let Some(at) = layer_n.and_then(|n| scene.shapes.iter().position(|s| s.layer == Some(n))) {
        let width = scene.width;
        let (below, above) = scene.shapes.split_at_mut(at + 1);
        let layer_shape = &below[at];
        let c = layer_shape.fill.color.as_dvec3();
        // A filled shape joins only if it lies mostly in the layer's region, so a shape spread
        // over the whole figure does not draw its silhouette in the pane
        let region = layer.as_ref().map(|l| l.mask.as_slice());
        let mostly_in_region = |s: &Shape| {
            let Some(region) = region else {
                return true;
            };
            let (mut inside, mut all) = (0usize, 0usize);
            for (i, v) in s.raster().pixels(width) {
                if v > 0.5 && i < region.len() {
                    all += 1;
                    inside += usize::from(region[i]);
                }
            }
            2 * inside > all
        };
        for shape in above.iter_mut().filter(|s| {
            overlaps(layer_shape, s, width)
                && crate::translucency::vocabulary::of_layer_hue(s.fill.color.as_dvec3(), c)
                && (s.stroke.is_some() || mostly_in_region(s))
        }) {
            shape.layer = layer_n;
        }
    }
    if let (Some(candidate), Some(n)) = (layer.as_ref(), layer_n) {
        timings.time("marks", || emit_marks(&target, &mut scene, n, candidate));
        scene.marks_layer = layer_n;
    }
    if options.stages.instancing {
        timings.time("instancing", || {
            crate::instancing::instance(&target, &mut scene)
        });
    }
    // Objects that do not pay their price once grouped, judged on the whole scene
    let economy = timings.time("economy", || {
        crate::economy::economize(&target, &mut scene, Some(EDGE_TERM), &tolerances(&target))
    });
    level_log(format_args!("economy: {economy:?}"));
    // The objects still in error moved jointly down the whole scene's error, last: before the
    // economy its merges retrace polished shapes on a thinner margin (focaccia -1.2 dB)
    let polished = timings.time("polish", || {
        crate::polish::polish(&target, &mut scene, Some(EDGE_TERM))
    });
    level_log(format_args!("polish: {polished:?}"));
    scene.fidelity = Some(crate::model::Fidelity {
        tolerance: target.tolerance.clone(),
        segment_price: target.accept_cost(),
    });
    timings.report(&format!("plan {:.1?}", start.elapsed()));
    progress(&scene);
    (scene, layer)
}

/// Whether two shapes cover a pixel in common, by any coverage (an anti-aliased rim
/// included).
fn overlaps(a: &Shape, b: &Shape, width: usize) -> bool {
    let (ra, rb) = (a.raster(), b.raster());
    let (small, other) = if ra.values.len() <= rb.values.len() {
        (ra, rb)
    } else {
        (rb, ra)
    };
    small
        .pixels(width)
        .any(|(i, v)| v > 0. && other.at(i, width) > 0.)
}

/// Whether a mark of an accepted layer pays: its gain must exceed its cubics plus one segment
/// (for its width, or a lens's opacity and blur) at the price per segment. The one keep rule
/// of the layer's marks (see [`crate::translucency::vocabulary::Keep`]).
pub fn keep_mark(mark: &crate::translucency::vocabulary::Mark, segment_cost: f64) -> bool {
    mark.gain > (mark.cubics.len() as f64 + 1.) * segment_cost
}

/// Adds the target's edges the decided scene does not draw as strokes and cuts (see
/// [`crate::linework::into_scene`]), kept by [`keep_mark`] with `term`'s edge credit: the tree
/// has no region for a thin line on one colour, such as the contour between niabot's legs.
fn linework(ctx: &Context, scene: &mut Scene, term: crate::edges::EdgeTerm) {
    let mut field = crate::edges::EdgeField::of_target(&ctx.target, ctx.width, ctx.height);
    field.tolerance = ctx.tolerance.clone();
    let report = crate::linework::into_scene(ctx, scene, &field, term.lambda, keep_mark);
    if crate::diagnostics::log() || crate::diagnostics::levels() {
        eprintln!(
            "linework: {} missing line px, {} step px; {} candidates, {} paying; kept {} strokes, {} cuts ({} closed), {} below a shape; stand-ins {} dropped, {} refitted",
            report.missing_lines,
            report.missing_steps,
            report.proposed,
            report.paying,
            report.strokes,
            report.cuts,
            report.closed,
            report.below,
            report.dropped,
            report.refitted
        );
    }
}

/// Writes the scene's marks for its kept translucent layer (`layer_n`): lenses, hems and
/// lines (see [`crate::translucency::vocabulary::emit`]) fitted over the rendered scene and kept
/// by [`keep_mark`]. The layer's own shape stays the plan's fitted one.
fn emit_marks(
    ctx: &Context,
    scene: &mut Scene,
    layer_n: usize,
    layer: &crate::translucency::Candidate,
) {
    use crate::translucency::raster::{DVec3, Image};
    use crate::translucency::vocabulary;
    // The layer's shape is the first of its group (its lines follow it), found by layer
    // index since the shape put on top by the layer decision keeps no id of its own
    let Some(shape) = scene.shapes.iter().find(|s| s.layer == Some(layer_n)) else {
        return;
    };
    let c = shape.fill.color.as_dvec3();
    let img = Image::from_target(&ctx.target, ctx.width);
    let full = Window::full(ctx.width, ctx.height);
    let base: Vec<DVec3> = scene
        .render(full, 0..scene.shapes.len(), [1.; 3])
        .iter()
        .map(|v| v.as_dvec3())
        .collect();
    // Under the edge term a mark is kept on its gain with its edge credit (one keep rule
    // sees the credited gain): a hem saves a quarter of its price in error alone
    let mut field = crate::edges::EdgeField::of_target(&ctx.target, ctx.width, ctx.height);
    field.tolerance = ctx.tolerance.clone();
    let edge = Some(std::sync::Arc::new((EDGE_TERM, field)));
    // The lenses are clipped to the layer's shape as it is drawn
    let options = vocabulary::Options {
        edge,
        tolerance: ctx.tolerance.clone(),
        clip: Some(std::sync::Arc::new(shape.coverage(full))),
        ..vocabulary::Options::new(ctx.segment_cost, keep_mark)
    };
    let v = vocabulary::emit(
        &img,
        &base,
        &layer.mask,
        c,
        layer.above.as_deref(),
        &options,
    );
    if crate::diagnostics::levels() {
        let hems = v
            .strokes
            .iter()
            .filter(|t| t.kind == vocabulary::StrokeKind::Hem)
            .count();
        eprintln!(
            "marks: {} lenses, {} tints, {hems} hems, {} lines",
            v.lenses.len(),
            v.tints.len(),
            v.strokes.len() - hems
        );
    }
    scene.marks = vocabulary::objects(&v);
}

/// Traces one image of the analysis: the given one, or the image beneath a translucent shape
/// peeled off it, which this traces again by the same steps with the shape put back on top.
/// Without `levels` no shape is peeled.
#[allow(clippy::too_many_arguments)]
fn trace_level(
    ctx: Context,
    hierarchy: Hierarchy,
    detail: f64,
    options: Options,
    edge: Option<&[bool]>,
    levels: bool,
    timings: &mut Timings,
    progress: &mut dyn FnMut(&Scene),
) -> Scene {
    trace_level_layer(
        ctx, hierarchy, detail, options, edge, levels, timings, progress,
    )
    .0
}

/// [`trace_level`], with the candidate this level's scene keeps as a translucent layer on top
/// (its highest layer), if it keeps one.
#[allow(clippy::too_many_arguments)]
fn trace_level_layer(
    mut ctx: Context,
    hierarchy: Hierarchy,
    detail: f64,
    options: Options,
    edge: Option<&[bool]>,
    levels: bool,
    timings: &mut Timings,
    progress: &mut dyn FnMut(&Scene),
) -> (Scene, Option<crate::translucency::Candidate>) {
    let (width, height) = (ctx.width, ctx.height);
    ctx.pull = crate::refine::HANDLE_PULL;
    // A level's own image (the beneath of a peeled layer) has its own uncertainty
    if ctx.tolerance.is_none() {
        timings.time("uncertainty", || {
            crate::uncertainty::apply(&mut ctx, options.tolerance)
        });
    }
    let hierarchy = &hierarchy;
    let cost = ctx.segment_cost;

    // Plan with estimated segment counts, trace the chosen outlines, and plan again with their
    // real counts
    // The target's edges, for the edge term
    let edge_field = crate::edges::EdgeField::of_target(&ctx.target, width, height);
    let planner = Planner::new(&ctx, hierarchy);
    let mut planner = timings.time("edges", || planner.with_edges(&edge_field, EDGE_TERM));
    // A fit's weights, with each pixel at its edge weight under the weighted edge term
    let edge_weighted = |mut weights: Weights| {
        for (w, i) in weights.values.iter_mut().zip(weights.window.pixels(width)) {
            *w *= edge_field.weight(i, EDGE_TERM.lambda);
        }
        weights
    };
    let background = hierarchy.background(height);
    let fit = &crate::outline::FIT_TURNS;
    let mut outlines: HashMap<u32, Vec<Vec<Cubic>>> = HashMap::default();
    // The sharp corners the trace found on each node's outline
    let mut corners: HashMap<u32, Vec<Vec2>> = HashMap::default();
    let mut plan = Vec::new();
    crate::progress::stage("plan");
    for _ in 0..ROUNDS {
        plan = timings.time("plan", || planner.plan(background));
        // Outlines are traced once per node
        for drawn in &plan[1..] {
            if crate::progress::poll() {
                break;
            }
            let node = drawn.node;
            if let std::collections::hash_map::Entry::Vacant(entry) = outlines.entry(node) {
                let (loops, found) =
                    timings.time("outline", || outline_with(hierarchy, node, height, fit));
                corners.insert(node, found);
                let segments = loops.iter().map(Vec::len).sum::<usize>() as f64;
                planner.segments[node as usize] = segments;
                entry.insert(loops);
            }
        }
    }
    // A cancelled trace ends on its ground
    if crate::progress::cancelled() {
        let ground = Fill::solid(ctx.rgb(plan[0].color));
        return (Scene::with_background(width, height, ground), None);
    }
    // Which shapes go below an earlier sibling
    let under = timings.time("order", || {
        occluders(hierarchy, &plan, &outlines, width, height)
    });

    // Draw the plan, each shape on top of what is drawn. A shape occluded by a sibling is drawn
    // before it, so the sibling is fitted against it and keeps the notch it shows through, and
    // the occluded outline under the sibling is fitted at the later weight
    let order = draw_order(&under);
    let rank = {
        let mut rank = vec![0; plan.len()];
        for (r, &k) in order.iter().enumerate() {
            rank[k] = r;
        }
        rank
    };
    // What each drawn node is painted with, fitted to the pixels it shows
    let fills = timings.time("fills", || fit_fills(&ctx, hierarchy, &plan));
    let mut scene = Scene::with_background(width, height, fills[0]);
    progress(&scene);
    if crate::progress::stage("draw") {
        return (scene, None);
    }
    let mut dropped = 0;
    // The scene id each plan entry got, or that of its nearest drawn ancestor if it was dropped
    let mut ids: Vec<usize> = vec![0; plan.len()];
    // The plan index of each scene id
    let mut planned: HashMap<usize, usize> = HashMap::default();
    planned.insert(0, 0);
    for (step, &k) in order[1..].iter().enumerate() {
        if crate::progress::at(step, order.len() - 1) {
            break;
        }
        let drawn = &plan[k];
        let fill = fills[k];
        // Simplification tolerances
        let tolerances = tolerances(&ctx);
        let reach = MARGIN + tolerances[tolerances.len() - 1];
        // The shape's own drawn descendants cover it later, so it does not follow them
        let masked: Vec<(u32, DVec3)> = plan
            .iter()
            .filter(|d| d.parent == Some(k))
            .map(|d| (d.node, d.color))
            .collect();
        let covered: Vec<u32> = masked.iter().map(|m| m.0).collect();
        let reading = smooth_at(&ctx, hierarchy, drawn.node, &covered);
        let smooth = reading.smooth;
        let allowance = block_allowance(&ctx, hierarchy, drawn.node, drawn.parent, &plan, smooth);
        // The gain, the position (on top of everything) and the shape
        let mut best: Option<(f64, usize, Shape)> = None;
        let top = scene.shapes.len();
        for (position, loops) in [(top, outlines[&drawn.node].clone())] {
            let mut shape = Shape::new(loops, fill).with_corners(corners[&drawn.node].clone());
            shape.smooth = reading.smooth.is_some();
            shape.antialiased = reading.antialiased.is_some();
            shape.node = Some(drawn.node);
            let window = Window::around(shape.min, shape.max, reach, width, height);
            // Shapes drawn later cover their regions, so the fit does not reach for those: a
            // square fitted before the circle it overlaps would otherwise grow over the circle,
            // whose color it is nearer to than the background
            let later: Vec<u32> = order[rank[k] + 1..]
                .iter()
                .map(|&j| plan[j].node)
                .filter(|&n| {
                    let n = &hierarchy.nodes[n as usize];
                    window.intersects(n.min.as_vec2(), n.max.as_vec2())
                })
                .collect();
            let fit = Context {
                weights: Some(edge_weighted(plan_weights(
                    &ctx,
                    hierarchy,
                    &masked,
                    &later,
                    drawn.color,
                    window,
                ))),
                ..ctx.clone()
            };
            let Some(near) = timings.time("place", || {
                Place::new(&fit, &scene, &shape, position, MARGIN)
            }) else {
                continue;
            };
            timings.time("refine", || near.refine(&fit, &mut shape));
            let Some(place) = timings.time("place", || {
                Place::new(&fit, &scene, &shape, position, reach)
            }) else {
                continue;
            };
            let fit_cost = timings.time("costs", || place.cost(&fit, fill));
            let origin = place.window.origin();
            let error = fit_cost.value(&shape.path(origin));
            timings.time("cleanup", || {
                clean_in(&fit, &fit_cost, origin, &mut shape, error)
            });
            timings.time("complete", || {
                complete(
                    &fit,
                    &place,
                    &fit_cost,
                    &mut shape,
                    completion(smooth, allowance, options.stages.primitives),
                )
            });
            let origin = place.window.origin();
            let error = fit_cost.value(&shape.path(origin));
            timings.time("cleanup", || {
                clean_in(&fit, &fit_cost, origin, &mut shape, error)
            });
            timings.time("simplify", || {
                simplify_with(
                    &fit,
                    &place,
                    &fit_cost,
                    &mut shape,
                    &tolerances,
                    crate::outline::CORNER_ANGLE,
                )
            });
            timings.time("complete", || {
                complete(
                    &fit,
                    &place,
                    &fit_cost,
                    &mut shape,
                    completion(smooth, allowance, options.stages.primitives),
                )
            });
            let error = place.measure(&fit, &fit_cost, &shape);
            let segments = shape.cost_segments();
            // Whether the shape is drawn at all is priced at the acceptance's price
            let (accept, path) = (ctx.accept_cost(), ctx.accept_path_cost());
            let gain = (place.before - error) as f64 - segments * accept - path;
            if best.as_ref().is_none_or(|(g, _, _)| gain > *g) {
                best = Some((gain, position, shape));
            }
        }
        let Some((_, position, shape)) = best.take_if(|(gain, _, _)| *gain >= 0.) else {
            if crate::diagnostics::log() {
                let n = &hierarchy.nodes[drawn.node as usize];
                eprintln!(
                    "dropped: node {} of {} pixels at {:?}, gain {:?}",
                    drawn.node,
                    n.stats.n,
                    n.min,
                    best.map(|b| b.0)
                );
            }
            dropped += 1;
            ids[k] = ids[drawn.parent.expect("not the background")];
            continue;
        };
        let id = scene.insert(position, shape);
        ids[k] = id;
        planned.insert(id, k);
        progress(&scene);
    }
    if crate::diagnostics::log() {
        eprintln!("plan: {dropped} planned shapes did not pay once fitted");
    }

    // Now that everything is drawn, every shape's outline is free under the shapes above it, so
    // complete and simplify each once more where they hide it: a circle drawn before the square
    // on it can become a circle only now
    crate::progress::stage("redraw");
    for position in 1..scene.shapes.len() {
        if crate::progress::at(position, scene.shapes.len()) {
            break;
        }
        let k = planned[&scene.shapes[position].id];
        if scene.shapes[position].primitive.is_some() {
            continue;
        }
        // Out of the scene while it is placed, so it is not above itself
        let mut shape = scene.shapes.remove(position);
        // The whole-scene render is cached by the count of changes: out and back in are two
        scene.change();
        let kept = shape.clone();
        let tolerances = tolerances(&ctx);
        let covered: Vec<u32> = plan
            .iter()
            .filter(|d| d.parent == Some(k))
            .map(|d| d.node)
            .collect();
        let smooth = smooth_at(&ctx, hierarchy, plan[k].node, &covered).smooth;
        let allowance =
            block_allowance(&ctx, hierarchy, plan[k].node, plan[k].parent, &plan, smooth);
        let reach = MARGIN + tolerances[tolerances.len() - 1];
        let window = Window::around(shape.min, shape.max, reach, width, height);
        let masked: Vec<(u32, DVec3)> = plan
            .iter()
            .filter(|d| d.parent == Some(k))
            .map(|d| (d.node, d.color))
            .collect();
        let fit = Context {
            weights: Some(edge_weighted(detail_weights(
                &ctx,
                hierarchy,
                &masked,
                plan[k].color,
                window,
            ))),
            ..ctx.clone()
        };
        let Some(place) = timings.time("place", || {
            Place::new(&fit, &scene, &shape, position, reach)
        }) else {
            scene.shapes.insert(position, kept);
            scene.change();
            continue;
        };
        let fit_cost = timings.time("costs", || place.cost(&fit, shape.fill));
        let before = fit_cost.value(&shape.path(place.window.origin()));
        let segments = shape.cost_segments();
        timings.time("complete", || {
            complete(
                &fit,
                &place,
                &fit_cost,
                &mut shape,
                completion(smooth, allowance, options.stages.primitives),
            )
        });
        let origin = place.window.origin();
        let error = fit_cost.value(&shape.path(origin));
        timings.time("cleanup", || {
            clean_in(&fit, &fit_cost, origin, &mut shape, error)
        });
        let error = timings.time("simplify", || {
            simplify_with(
                &fit,
                &place,
                &fit_cost,
                &mut shape,
                &tolerances,
                crate::outline::CORNER_ANGLE,
            )
        });
        let error = timings
            .time("complete", || {
                complete(
                    &fit,
                    &place,
                    &fit_cost,
                    &mut shape,
                    completion(smooth, allowance, options.stages.primitives),
                )
            })
            .min(error);
        if (error - before) as f64 <= (segments - shape.cost_segments()) * cost {
            let change = scene.change();
            scene.shapes.insert(
                position,
                Shape {
                    changed: change,
                    fitted: change,
                    ..shape
                },
            );
        } else {
            scene.shapes.insert(position, kept);
            scene.change();
        }
    }

    if crate::progress::stage("shapes") {
        return (scene, None);
    }
    timings.time("recolor", || recolor(&ctx, &mut scene));
    timings.time("settle", || settle(&ctx, &mut scene));
    timings.time("cleanup", || crate::cleanup::cleanup(&ctx, &mut scene));
    if options.stages.strokes {
        strokes(&ctx, hierarchy, &mut scene, timings);
    }
    if !levels || !options.stages.translucency || crate::progress::stage("families") {
        return (scene, None);
    }
    // No layer is proposed where no contour runs on under a tint: the families and the
    // discovery read nothing else, and pictures without one are most of them
    let lv = timings.time("evidence", || {
        crate::translucency::leaves::Leaves::new(&ctx.target, hierarchy.labels(), width, height)
    });
    if !timings.time("evidence", || {
        crate::translucency::evidence::layer_witnessed(&lv)
    }) {
        level_log(format_args!("no contour runs on under a tint"));
        return (scene, None);
    }
    // A candidate is peeled, the beneath traced by these same steps and the shape put back on
    // top; most likely first, each decided in its window ([`decide_in_window`]), the first kept
    // then whole by the objective. [`crate::diagnostics::Diagnostics::no_peel`] leaves every shape opaque
    let no_peel = crate::diagnostics::no_peel();
    // With `acceptance`, the families past the gate are each decided whole (see
    // [`decide_whole`])
    let mut whole_families = Vec::new();
    if options.acceptance && !no_peel {
        let clock = crate::timing::Clock::now();
        let fams = timings.time("families", || {
            accepted_families(&ctx, &scene, hierarchy.labels())
        });
        level_log(format_args!(
            "{} families past the gate in {:.1?}",
            fams.len(),
            clock.elapsed()
        ));
        whole_families = fams;
    }
    // The levels beneath a peeled shape run without the loop
    let nested = Options {
        acceptance: false,
        ..options
    };
    // the scene with the layer, its objective, the objective without, the layer, and whether
    // it is accepted on its evidence rather than on the objective (see [`decide_whole`])
    let mut chosen: Option<(Scene, f64, f64, crate::translucency::Candidate, bool)> = None;
    // The families decided whole before the discovery's candidates are decided below: neither
    // reads the other, and their results meet only after both
    let whole = (!whole_families.is_empty()).then(|| {
        decide_whole(
            &ctx,
            &scene,
            std::mem::take(&mut whole_families),
            edge,
            detail,
            nested,
            timings,
        )
    });
    // A family is accepted on its gate's evidence, the objective only ordering the families
    // (see [`decide_whole`]); but not where its drawing errs by more than the opaque trace's
    // whole error again, a beneath or a part above that went wrong (plain 1052: 10080 vs 178)
    let without = parts(&ctx, &scene);
    let without_error = without.error;
    let decided: Vec<(Scene, f64, crate::translucency::Candidate)> = whole
        .into_iter()
        .flatten()
        .filter(|(with, _, candidate)| {
            let drawn = parts(&ctx, with).error;
            let refused = drawn > 2.0 * without_error;
            if refused {
                level_log(format_args!(
                    "whole family k {:.3}: refused, drawn at error {drawn:.0} against {without_error:.0} without",
                    candidate.k
                ));
            }
            !refused
        })
        .collect();
    // A family decided on its evidence overrides any candidate of the discovery, which is
    // then not proposed at all
    let candidates = timings.time("peel", || {
        if no_peel || !decided.is_empty() || crate::progress::cancelled() {
            Vec::new()
        } else {
            crate::translucency::candidates(&ctx, &lv)
        }
    });
    let discovered = candidates.len();
    level_log(format_args!(
        "{discovered} discovered{}",
        if decided.is_empty() {
            ""
        } else {
            " (a family is decided)"
        }
    ));
    for (n, candidate) in candidates.into_iter().enumerate() {
        if crate::progress::cancelled() {
            break;
        }
        let Some(overlay) = candidate.overlay(width, height) else {
            continue;
        };
        let what = format!(
            "{} {n} k {:.3} c {:?} {} px{}",
            if n < discovered {
                "discovery"
            } else {
                "family"
            },
            candidate.k,
            (candidate.fill().color * 255.).round().to_array(),
            candidate.mask.iter().filter(|&&m| m).count(),
            truth_iou(&candidate.mask).map_or(String::new(), |v| format!(" IoU {v:.2}"))
        );
        let peeled = timings.time("peel", || candidate.peeled(&ctx, &overlay));
        let clock = crate::timing::Clock::now();
        let window = timings.time("window", || {
            decide_in_window(&ctx, &candidate, &peeled, detail, options, edge)
        });
        let verdict = window.map(|v| v.keeps());
        if let Some(v) = window {
            level_log(format_args!(
                "{what}: window {} with {:.0} (error {:.0}, {} paths) without {:.0} (error {:.0}, {} paths) in {:.1?}",
                if v.keeps() { "keeps" } else { "rejects" },
                v.plain.total() + v.margin(),
                v.with.error,
                v.with.paths,
                v.plain.total(),
                v.plain.error,
                v.plain.paths,
                clock.elapsed()
            ));
        }
        if verdict == Some(false) {
            continue;
        }
        // The image beneath each peeled shape, as `peel-N.ppm` in the peel directory,
        // numbered from the top, for `scripts/translucency.sh`
        #[cfg(feature = "diagnostics")]
        if let Some(dir) = crate::diagnostics::peel_dir() {
            let level = std::fs::read_dir(&dir)
                .map(|d| d.filter(|e| e.is_ok()).count())
                .unwrap_or(0);
            let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
            ppm.extend_from_slice(&peeled);
            let _ = std::fs::write(dir.join(format!("peel-{level}.ppm")), ppm);
        }
        level_log(format_args!("{what}: traced whole"));
        let clock = crate::timing::Clock::now();
        LEVEL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let with = beneath_with(
            &ctx, &scene, &candidate, edge, overlay, &peeled, detail, nested, true, timings,
            progress,
        );
        LEVEL.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // Under a layer cost the margin takes the layer's terms (see
        // [`crate::translucency::cost`]); the plain objective otherwise
        let (with_parts, without_parts) = (parts(&ctx, &with), parts(&ctx, &scene));
        let terms = layer_terms(&ctx, &with, &scene, &candidate);
        let worse = without_parts.total();
        let better = worse
            + margin(
                &with_parts,
                &without_parts,
                shading_discount(&ctx, &candidate),
                &terms,
                ctx.segment_cost,
            );
        level_log(format_args!(
            "{what}: terms k {:.3}, error under {:.4} with / {:.4} without",
            terms.k, terms.under_with, terms.under_without,
        ));
        if crate::diagnostics::levels() {
            let (w, o) = (parts(&ctx, &with), parts(&ctx, &scene));
            level_log(format_args!(
                "{what}: {} with {better:.0} (error {:.0}, {:.0} segments, {} paths) without {worse:.0} (error {:.0}, {:.0} segments, {} paths) in {:.1?}",
                if better < worse {
                    "ACCEPTED"
                } else {
                    "rejected"
                },
                w.error,
                w.segments,
                w.paths,
                o.error,
                o.segments,
                o.paths,
                clock.elapsed()
            ));
        }
        // One whole trace per level, for the first candidate its window does not reject: each
        // whole trace recurses with candidates of its own
        chosen = Some((with, better, worse, candidate, false));
        break;
    }
    let worse = without.total();
    for (with, better, candidate) in decided {
        let (best, on_evidence) = chosen
            .as_ref()
            .map_or((worse, false), |c| (c.1.min(c.2), c.4));
        if !on_evidence || better < best {
            chosen = Some((with, better, worse, candidate, true));
        }
    }
    match chosen {
        Some((with, better, worse, candidate, on_evidence)) if on_evidence || better < worse => {
            // The kept layer's region as `layer-N.pgm` in the mask directory, numbered from
            // the top, with its k and c in `layer-N.txt`
            #[cfg(feature = "diagnostics")]
            if let Some(dir) = crate::diagnostics::mask_dir() {
                let n = std::fs::read_dir(&dir)
                    .map(|d| {
                        d.filter(|e| {
                            e.as_ref().is_ok_and(|e| {
                                e.file_name().to_string_lossy().ends_with(".pgm")
                                    && e.file_name().to_string_lossy().starts_with("layer-")
                            })
                        })
                        .count()
                    })
                    .unwrap_or(0);
                let mut pgm = format!("P5\n{width} {height}\n255\n").into_bytes();
                pgm.extend(candidate.mask.iter().map(|m| if *m { 255 } else { 0 }));
                let _ = std::fs::write(dir.join(format!("layer-{n}.pgm")), pgm);
                let c = (candidate.fill().color * 255.).round().to_array();
                let _ = std::fs::write(
                    dir.join(format!("layer-{n}.txt")),
                    format!("{:.3} {} {} {}\n", candidate.k, c[0], c[1], c[2]),
                );
                // and what lies above it inside its region as `holes-N.pgm`
                if let Some(holes) = &candidate.holes {
                    let mut pgm = format!("P5\n{width} {height}\n255\n").into_bytes();
                    pgm.extend(holes.iter().map(|m| if *m { 255 } else { 0 }));
                    let _ = std::fs::write(dir.join(format!("holes-{n}.pgm")), pgm);
                }
            }
            (with, Some(candidate))
        }
        _ => (scene, None),
    }
}

/// The effects: runs of bands the picture shades across drawn as one shape with a gradient,
/// grounds drawn flat under glows, and shapes blurred where the picture shows their boundary as a
/// ramp.
fn effects(target: &Context, scene: &mut Scene, timings: &mut Timings) {
    let softened = timings.time("soft", || crate::soft::soften_scene(target, scene));
    level_log(format_args!("soft: {softened} runs replaced"));
    // Grounds drawn flat with glows fading to transparent over them, where that pays
    let glowed = timings.time("glow", || crate::glow::glow_scene(target, scene));
    level_log(format_args!("glow: {glowed} grounds"));
    // The decided scene's shapes blurred where the picture shows their boundary as a ramp
    let blurred = timings.time("blur", || crate::blur::blur_scene(target, scene));
    level_log(format_args!("blur: {blurred} shapes blurred"));
}

/// The strokes: bands recognized as strokes, strokes continued across gaps, dashed and dotted
/// lines.
fn strokes(ctx: &Context, hierarchy: &Hierarchy, scene: &mut Scene, timings: &mut Timings) {
    timings.time("strokes", || crate::stroke::recognize(ctx, hierarchy, scene));
    // The strokes that continue one another across a gap joined into one (the wall's
    // grid lines under glassware's glass, see [`crate::stroke::continue_through`])
    timings.time("continue", || {
        crate::stroke::continue_through(ctx, scene)
    });
    timings.time("dashes", || crate::dashes::group(ctx, scene));
    // The dashes and dots the trace left undrawn, read off the segmentation's leaves
    timings.time("dashes", || {
        crate::dashes::from_leaves(
            ctx,
            hierarchy,
            scene,
            &tolerances(ctx),
            Some(EDGE_TERM),
        )
    });
}
