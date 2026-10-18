//! A proposal fitted in the scene: each cap, the join, the thick remainder.

use super::*;

/// What [`fit_at`] fits: the error, the strokes with their fills, the thick remainder of a
/// network, and the bridges its lines were joined across.
pub(super) type Fitted = (f32, Vec<(Stroke, Fill)>, Option<Shape>, Vec<Shape>);

/// The strokes `shape` is, fitted in `place`: one for a band in the shape's fill, or one per
/// line of a network (see [`network`]), each fitted on its own in its colour from the image
/// ([`line_fill`]), with the region's thick parts left as a fill ([`thick_remainder`]).
pub(super) fn fit_at(
    ctx: &Context,
    hierarchy: &Hierarchy,
    place: &Place,
    cost: &CoverageCost,
    shape: &Shape,
) -> Option<Fitted> {
    let edges = Edges::new(ctx, place, shape.fill);
    // A network by its skeleton, whatever the one-pen rule and the proposals left of it: a
    // line that fails either leaves its pixels to the remainder
    let (network, skeletons) = match skeletons_of_shape(shape, Some(hierarchy)) {
        Some(Skeletons::One(one)) => (false, vec![one]),
        Some(Skeletons::Network(lines)) => (true, lines),
        None => (false, Vec::new()),
    };
    let proposals: Vec<Proposal> = skeletons
        .into_iter()
        .filter_map(|skeleton| propose(skeleton, &edges))
        .collect();
    if proposals.is_empty() {
        return None;
    }
    let origin = place.window.origin();
    let fitted: Vec<(f32, Stroke, Fill)> = proposals
        .into_iter()
        .filter_map(|p| {
            let fill = if network {
                line_fill(ctx, place, &p.stroke, shape.fill)
            } else {
                shape.fill
            };
            let own_cost = (fill != shape.fill).then(|| place.cost(ctx, fill));
            fit_proposal(ctx, own_cost.as_ref().unwrap_or(cost), origin, p)
                .map(|(e, s)| (e, s, fill))
        })
        .collect();
    // A network stays a network however many of its lines fitted, so one fitted line never
    // stands for the whole shape; what the lines left is the remainder's
    match (network, fitted.len()) {
        (_, 0) => None,
        (false, _) => fitted
            .into_iter()
            .next()
            .map(|(e, s, f)| (e, vec![(s, f)], None, Vec::new())),
        (true, _) => {
            let mut lines: Vec<(Stroke, Fill)> =
                fitted.into_iter().map(|(_, s, f)| (s, f)).collect();
            // One pen, read again on the fitted widths: a line the fit widened past the pen's
            // slack goes to the remainder (as glassware's grid line where completion smeared
            // it under the condenser tube)
            let mut widths: Vec<f32> = lines.iter().map(|(s, _)| s.width).collect();
            widths.sort_by(f32::total_cmp);
            let pen = widths[widths.len() / 2];
            lines.retain(|(s, _)| (s.width - pen).abs() <= (WIDTH_SLACK * pen).max(WIDTH_QUANTUM));
            // A line that draws beyond the shape ([`draws_beyond`]) goes to the remainder too,
            // and the network stands on the rest, so one spilling line (as a piece of
            // glassware's condenser outline in the grid's region) does not throw out the grid
            lines.retain(|(s, _)| !draws_beyond(hierarchy, shape, s, &[], ctx.width, ctx.height));
            if lines.is_empty() {
                return None;
            }
            let (lines, bridges) = join_lines(ctx, place, shape, lines);
            let mut shapes: Vec<Shape> = lines.iter().map(|(s, f)| s.shape(*f)).collect();
            let remainder = thick_remainder(ctx, hierarchy, place, shape, &shapes);
            shapes.extend(remainder.clone());
            let error = place.error_of(ctx, &shapes);
            Some((error, lines, remainder, bridges))
        }
    }
}

/// What a network's lines leave of the shape (as the deep band in one region with glassware's
/// grid): the region's pixels no line touches, as a fill in their own colour over the lines,
/// traced and refitted in place; `None` where the lines cover the region.
pub(super) fn thick_remainder(
    ctx: &Context,
    hierarchy: &Hierarchy,
    place: &Place,
    shape: &Shape,
    lines: &[Shape],
) -> Option<Shape> {
    let (mask, w, h, top_left) = mask_of(shape, Some(hierarchy));
    let rasters: Vec<&Raster> = lines.iter().map(|l| l.raster()).collect();
    let touched = |x: usize, y: usize| {
        rasters.iter().any(|r| {
            x >= r.x0
                && y >= r.y0
                && x < r.x0 + r.w
                && y < r.y0 + r.h
                && r.values[(y - r.y0) * r.w + x - r.x0] > 0.
        })
    };
    let left = |i: usize| -> bool {
        let (x, y) = (i % ctx.width, i / ctx.width);
        let (bx, by) = (
            x as i64 - top_left.x as i64 + 1,
            y as i64 - top_left.y as i64 + 1,
        );
        bx >= 0
            && by >= 0
            && (bx as usize) < w
            && (by as usize) < h
            && mask[by as usize * w + bx as usize]
            && !touched(x, y)
    };
    let min = top_left;
    let max = top_left + IVec2::new(w as i32 - 2, h as i32 - 2);
    let loops =
        crate::outline::outline_of(min, max, ctx.width, ctx.height, &crate::outline::FIT, left);
    if loops.is_empty() {
        return None;
    }
    let pixels = (min.y.max(0) as usize..(max.y.max(0) as usize).min(ctx.height))
        .flat_map(|y| {
            (min.x.max(0) as usize..(max.x.max(0) as usize).min(ctx.width))
                .map(move |x| y * ctx.width + x)
        })
        .filter(|&i| left(i))
        .map(|i| (i, 1.));
    let fill = fill_over(ctx, place, pixels, shape.fill);
    let mut remainder = Shape {
        smooth: shape.smooth,
        antialiased: shape.antialiased,
        node: shape.node,
        layer: shape.layer,
        ..Shape::new(loops, fill)
    };
    place.refine(ctx, &mut remainder);
    Some(remainder)
}

/// The colour of one line of a network, read from the image over the pixels the proposed
/// stroke covers (see [`fill_over`]), since the lines of one region need not share its mean.
pub(super) fn line_fill(ctx: &Context, place: &Place, stroke: &Stroke, fill: Fill) -> Fill {
    let read = |s: &Stroke| {
        let drawn = s.shape(fill);
        let raster = drawn.raster();
        let pixels = (raster.y0..(raster.y0 + raster.h).min(ctx.height)).flat_map(|y| {
            (raster.x0..(raster.x0 + raster.w).min(ctx.width)).map(move |x| {
                (
                    y * ctx.width + x,
                    raster.values[(y - raster.y0) * raster.w + x - raster.x0],
                )
            })
        });
        fill_over(ctx, place, pixels, fill)
    };
    let whole = read(stroke);
    // Also read on the core, an anti-aliasing scale in from either side, which a proposal too
    // wide keeps on its line (wingchair's coils, pale over their rims) and one off it does not
    // (trellis' vine): the reading that screens better is kept
    let core = Stroke {
        width: stroke.width - 2. * crate::edges::SIGMA as f32,
        ..stroke.clone()
    };
    if core.width < 1. {
        return whole;
    }
    let inner = read(&core);
    // readings within the noise are one colour
    if (inner.color - whole.color).abs().max_element() <= crate::translucency::NOISE {
        return whole;
    }
    let origin = place.window.origin();
    let screened = |f: Fill| {
        let mut fitting = Fitting::new(stroke, origin, 1.);
        fitting.run(&place.cost(ctx, f), SCREEN_STEPS);
        fitting.error()
    };
    if screened(inner) < screened(whole) {
        inner
    } else {
        whole
    }
}

/// `fill` with its colour the image's mean over the shown, counted `pixels` (image index,
/// coverage) covered whole, or by half where none is whole; `fill` itself where nothing can
/// be read, or where it is a gradient.
pub(super) fn fill_over(
    ctx: &Context,
    place: &Place,
    pixels: impl Iterator<Item = (usize, f32)>,
    fill: Fill,
) -> Fill {
    if fill.gradient.is_some() {
        return fill;
    }
    let (mut whole, mut half) = ((Vec3::ZERO, 0usize), (Vec3::ZERO, 0usize));
    for (i, c) in pixels {
        if c < 0.5 {
            continue;
        }
        let Some(k) = place.window.index(i, ctx.width) else {
            continue;
        };
        let shown = place
            .above
            .as_ref()
            .is_none_or(|a| a.transmittance[k].min_element() > 0.99);
        let (weight, target) = ctx.weighted_target(i);
        if !shown || weight <= 0. {
            continue;
        }
        half = (half.0 + target, half.1 + 1);
        if c > 0.99 {
            whole = (whole.0 + target, whole.1 + 1);
        }
    }
    let (sum, n) = if whole.1 > 0 { whole } else { half };
    if n == 0 {
        return fill;
    }
    Fill {
        color: sum / n as f32,
        ..fill
    }
}

/// One proposed stroke fitted against `cost`, with its error: each cap fitted, the join picked
/// on the fitted geometry, and the centerline simplified while that pays.
pub(super) fn fit_proposal(
    ctx: &Context,
    cost: &CoverageCost,
    origin: Vec2,
    proposal: Proposal,
) -> Option<(f32, Stroke)> {
    let Proposal {
        stroke: proposal,
        spec,
        ends_at_extent,
        from_pairs,
    } = proposal;
    // Each kind of cap is fitted, since the ends move with it; the join is chosen on
    // the fitted geometry, where the kinds differ only at the corners
    let caps = if proposal.closed {
        vec![Cap::Butt]
    } else {
        vec![Cap::Butt, Cap::Round, Cap::Square]
    };
    // The end moves by what the cap reaches beyond it, from the band's extent, or out to a
    // butt end from where the skeleton stopped
    let half = proposal.width / 2.;
    let moves = |cap: Cap| {
        let by = |at_extent: bool| match (cap, at_extent, proposal.closed) {
            (_, _, true) => 0.,
            (Cap::Square, _, _) if from_pairs => -half,
            (_, _, _) if from_pairs => 0.,
            (Cap::Butt, false, _) => half,
            (Cap::Butt, true, _) => 0.,
            (_, true, _) => -half,
            (_, false, _) => 0.,
        };
        (by(ends_at_extent[0]), by(ends_at_extent[1]))
    };
    let candidates: Vec<(Stroke, f32)> = caps
        .iter()
        .filter_map(|&cap| {
            let (a, b) = moves(cap);
            let mut candidate = extended(&proposal, a, b);
            candidate.cap = cap;
            (!candidate.outline().is_empty()).then_some((candidate, 1.))
        })
        .collect();
    let (mut error, mut stroke) = fit_screened(cost, origin, candidates, |e, _| e as f64)?;
    let cap_moves = moves(stroke.cap);
    if stroke.corners.iter().any(|&c| c) {
        for join in [Join::Round, Join::Bevel] {
            let mut candidate = stroke.clone();
            candidate.join = join;
            let e = cost.value(&candidate.path(origin));
            if e < error {
                (error, stroke) = (e, candidate);
            }
        }
        // The geometry was fitted under miter joins, whose spikes it pulled the corners in
        // to meet, so it is fitted again under the join chosen
        if stroke.join != Join::Miter {
            error = fit(cost, &mut stroke, origin, 1.);
            if !error.is_finite() {
                return None;
            }
        }
    }
    // The centerline is simplified like an outline, kept while the error rises by less than
    // the segments saved cost, from two starts per tolerance since the optimizer's basin is
    // narrow: a fresh fit of the crossings, and the current centerline simplified and polished
    let total = |e: f32, segments: usize| e as f64 + segments as f64 * ctx.segment_cost;
    for tolerance in SIMPLIFY_TOLERANCES {
        let mut candidates: Vec<Stroke> = Vec::new();
        if let Some((centerline, corners)) = spec.build(tolerance) {
            let mut candidate = Stroke {
                centerline,
                corners,
                ..stroke.clone()
            };
            if !candidate.closed {
                candidate = extended(&candidate, cap_moves.0, cap_moves.1);
            }
            candidates.push(candidate);
        }
        let simpler_centerline = crate::tracer::beautify::beautify_chain_at(
            &stroke.centerline,
            &stroke.corners,
            tolerance,
        );
        let corners: Vec<bool> = simpler_centerline
            .iter()
            .map(|c| {
                stroke
                    .centerline
                    .iter()
                    .zip(&stroke.corners)
                    .any(|(o, &flag)| flag && o[0].distance(c[0]) < 1e-3)
            })
            .collect();
        candidates.push(Stroke {
            centerline: polish(&simpler_centerline, &stroke.centerline, stroke.closed),
            corners,
            ..stroke.clone()
        });
        let stroke_len = stroke.centerline.len();
        let off_by = tolerance / CENTERLINE_TOLERANCE;
        let candidates: Vec<(Stroke, f32)> = candidates
            .into_iter()
            .filter(|c| c.centerline.len() < stroke_len)
            .map(|c| (c, off_by))
            .collect();
        let rank = |e: f32, s: &Stroke| total(e, s.centerline.len());
        if candidates.is_empty() {
            continue;
        }
        // none fitting ends the simplification, as a simpler one fitting worse does
        let Some((e, simpler)) = fit_screened(cost, origin, candidates, rank) else {
            break;
        };
        if total(e, simpler.centerline.len()) < total(error, stroke.centerline.len()) {
            (error, stroke) = (e, simpler);
        } else {
            break;
        }
    }
    Some((error, stroke))
}
