//! Every shape that is a band replaced by the stroke it draws, where that pays.

use super::*;

/// How many positions down the stack a stroke is tried at, below the shapes it overlaps.
pub(super) const MAX_POSITIONS: usize = 2;

/// The best reading of a band so far: its gain, position, strokes, remainder and bridges.
type Best = (f64, usize, Vec<(Stroke, Fill)>, Option<Shape>, Vec<Shape>);

/// Whether the boxes of `a` and `b` meet.
fn boxes_meet(a: &Shape, b: &Shape) -> bool {
    a.min.cmple(b.max).all() && b.min.cmple(a.max).all()
}

/// Whether a stroke covers by half a pixel more than a pixel from everything the shape, its
/// region and the network's `bridges` ([`join_lines`]) cover by half, since a stroke stands
/// for its band (a round cap may spill past it).
pub(super) fn draws_beyond(
    hierarchy: &Hierarchy,
    shape: &Shape,
    stroke: &Stroke,
    bridges: &[Shape],
    width: usize,
    height: usize,
) -> bool {
    let drawn = stroke.shape(shape.fill);
    let (rs, ro) = (drawn.raster(), shape.raster());
    let bridged: Vec<&Raster> = bridges.iter().map(Shape::raster).collect();
    let own = |x: usize, y: usize| -> bool {
        let i = y * width + x;
        ro.at(i, width) >= 0.5
            || shape.node.is_some_and(|n| hierarchy.contains(n, i))
            || bridged.iter().any(|b| b.at(i, width) >= 0.5)
    };
    for y in rs.y0..(rs.y0 + rs.h).min(height) {
        for x in rs.x0..(rs.x0 + rs.w).min(width) {
            if rs.values[(y - rs.y0) * rs.w + x - rs.x0] < 0.5 {
                continue;
            }
            let near = (y.saturating_sub(1)..(y + 2).min(height))
                .any(|yy| (x.saturating_sub(1)..(x + 2).min(width)).any(|xx| own(xx, yy)));
            if !near {
                return true;
            }
        }
    }
    false
}

/// The band `shape` (out of `scene`, at `k`) read as the visible part of a fill below the
/// shapes from `j` up, which hide its inner side: bridged there and refitted, the shapes over
/// it refitted. The shapes for `j..k` with it, where they cost less there than `chosen`.
fn fill_under(
    ctx: &Context,
    scene: &Scene,
    shape: &Shape,
    j: usize,
    k: usize,
    chosen: &[Shape],
) -> Option<Vec<Shape>> {
    use crate::complete::{Completion, PRIMITIVE_TOLERANCE, complete};
    if scene.shapes[j..k].iter().any(|s| s.clip.is_some()) {
        return None;
    }
    // Over the shapes it goes under too, so a bridge hidden by them is measured
    let mut framed = shape.clone();
    for s in &scene.shapes[j..k] {
        if boxes_meet(s, shape) {
            framed.min = framed.min.min(s.min);
            framed.max = framed.max.max(s.max);
        }
    }
    let place = Place::new(ctx, scene, &framed, j, MARGIN)?;
    let cost = place.cost(ctx, shape.fill);
    let mut filled = shape.clone();
    let what = Completion {
        primitives: false,
        primitive_tolerance: PRIMITIVE_TOLERANCE,
        primitive_allowance: 0.,
    };
    complete(ctx, &place, &cost, &mut filled, what);
    if filled.loops == shape.loops {
        return None;
    }
    place.refine(ctx, &mut filled);
    let mut trial = scene.clone();
    trial.shapes.insert(j, filled.clone());
    trial.change();
    // What the band hid of the shapes over it now shows, so they are fitted again
    for i in j + 1..=k {
        let s = &trial.shapes[i];
        if !boxes_meet(s, &filled) || s.blur > 0. {
            continue;
        }
        let s = trial.shapes.remove(i);
        trial.change();
        let mut refitted = s.clone();
        let kept = match Place::new(ctx, &trial, &s, i, MARGIN) {
            Some(p) if p.refine(ctx, &mut refitted) < p.error(ctx, &s) => refitted,
            _ => s,
        };
        trial.shapes.insert(i, kept);
        trial.change();
    }
    let under: Vec<Shape> = trial.shapes[j..=k].to_vec();
    // Both readings measured over one window with what lies above them
    let mut base = scene.clone();
    base.shapes.drain(j..k);
    base.change();
    let mut framed = shape.clone();
    for s in under.iter().chain(chosen) {
        framed.min = framed.min.min(s.paint_min());
        framed.max = framed.max.max(s.paint_max());
    }
    let window = Place::new(ctx, &base, &framed, j, MARGIN)?;
    let total = |shapes: &[Shape]| {
        window.error_of(ctx, shapes) as f64
            + shapes.iter().map(Shape::cost_segments).sum::<f64>() * ctx.segment_cost
    };
    let (under_total, chosen_total) = (total(&under), total(chosen));
    (under_total < chosen_total).then_some(under)
}

/// Turns the scene's bands into strokes where that costs less, each tried where the shape is
/// and below the shapes it overlaps (where a notching shape leaves its edges free). A stroke
/// that draws beyond the shape it replaces is not taken ([`draws_beyond`]).
pub async fn recognize(ctx: &Context, hierarchy: &Hierarchy, scene: &mut Scene) {
    let mut k = 1;
    while k < scene.shapes.len() {
        crate::progress::poll().await;
        // A stroke interpolates edges, which only anti-aliased edges ask for (pixel art stays
        // blocks, a blurred shape soft); a shape anti-aliased but not smooth (of more than one
        // colour) is tried only where its skeleton branches, as a network
        let candidate = &scene.shapes[k];
        if candidate.clip.is_some()
            || candidate.blur > 0.
            || !(candidate.smooth || candidate.antialiased)
        {
            k += 1;
            continue;
        }
        // Nor where no skeleton line has a stroke's width and length ([`first_width`]),
        // read before the shape is placed anywhere
        let could = match skeletons_of_shape(candidate, Some(hierarchy)) {
            Some(Skeletons::One(one)) => candidate.smooth && first_width(&one).is_some(),
            Some(Skeletons::Network(lines)) => lines.iter().any(|l| first_width(l).is_some()),
            None => false,
        };
        if !could {
            k += 1;
            continue;
        }
        // The shape is taken out, so what lies above it is only the shapes above; every edit
        // of the stack is noted, since the scene caches its full render by its change count
        let shape = scene.shapes.remove(k);
        scene.change();
        let mut positions = vec![k];
        let mut lower = Vec::new();
        for j in (1..k).rev() {
            if positions.len() > MAX_POSITIONS {
                break;
            }
            let other = &scene.shapes[j];
            let overlaps = boxes_meet(other, &shape);
            let holds = other.min.cmple(shape.min).all() && shape.max.cmple(other.max).all();
            if overlaps && !holds {
                lower.push(j);
                positions.push(j);
            }
        }
        let total = |error: f32, segments: f64| error as f64 + segments * ctx.segment_cost;
        // the shape's own error as the cost reads it, and as the context measures it
        // rendered (a network's lines in their own colours are measured rendered)
        let mut free: Option<(f32, f32)> = None;
        let mut best: Option<(Best, Place)> = None;
        // the single stroke fitted where the shape is, which a lower position refits
        let mut own: Option<Stroke> = None;
        // The gain over the shape left as it is (`free_error`, as its cost reads it or rendered):
        // lines and what they leave are each a path, priced beyond the one path the shape was
        let gain = |free_error: f32, fitted: &Fitted| {
            let (error, strokes, remainder, _) = fitted;
            let paths = strokes.len() + usize::from(remainder.is_some());
            let t = total(
                *error,
                strokes.iter().map(|(s, _)| s.segments()).sum::<usize>() as f64
                    + remainder.as_ref().map_or(0., Shape::cost_segments),
            ) + (paths - 1) as f64 * ctx.path_cost;
            total(free_error, shape.cost_segments()) - t
        };
        for position in positions {
            crate::progress::poll().await;
            // The window covers the region's box too, since the shape's simplified outline can
            // fall short of the band
            let mut framed = shape.clone();
            if let Some(node) = shape.node {
                let n = &hierarchy.nodes[node as usize];
                framed.min = framed.min.min(n.min.as_vec2());
                framed.max = framed.max.max(n.max.as_vec2());
            }
            let Some(place) = Place::new(ctx, scene, &framed, position, MARGIN) else {
                continue;
            };
            crate::progress::poll().await;
            let cost = place.cost(ctx, shape.fill);
            if position == k {
                free = Some((
                    cost.value(&shape.path(place.window.origin())),
                    place.error(ctx, &shape),
                ));
            }
            // Below, the stroke fitted where the shape is is refitted against what lies above
            // there, its cap, join and centerline kept; a network is fitted again whole
            let fitted = match &own {
                Some(stroke) if position != k => {
                    let mut stroke = stroke.clone();
                    let error = fit(&cost, &mut stroke, place.window.origin(), 1.);
                    error
                        .is_finite()
                        .then(|| (error, vec![(stroke, shape.fill)], None, Vec::new()))
                }
                _ => fit_at(ctx, hierarchy, &place, &cost, &shape).await,
            };
            let (Some(fitted), Some(free)) = (fitted, free) else {
                continue;
            };
            if position == k && fitted.1.len() == 1 && fitted.2.is_none() {
                own = Some(fitted.1[0].0.clone());
            }
            // a network's lines in their own colours against the shape rendered, one line
            // against the shape as its cost reads it
            let g = gain(if fitted.1.len() > 1 { free.1 } else { free.0 }, &fitted);
            if best.as_ref().is_none_or(|(b, _)| g > b.0) {
                let (_, strokes, remainder, bridges) = fitted;
                best = Some(((g, position, strokes, remainder, bridges), place));
            }
        }
        let (Some(free), Some((mut best, place))) = (free, best) else {
            scene.shapes.insert(k, shape);
            scene.change();
            k += 1;
            continue;
        };
        // A band's excess over its stroke and caps stays a fill over it, set against the shape
        // as its cost reads it, as the stroke alone is; tried only where the two, drawn
        // exactly, could gain more than the stroke alone
        let bound = |b: &Best| {
            let segments = b.2.iter().map(|(s, _)| s.segments()).sum::<usize>() as f64;
            total(free.0, shape.cost_segments()) - total(0., segments) - ctx.path_cost
        };
        if best.2.len() == 1 && best.3.is_none() && bound(&best) > best.0.max(0.) {
            let (stroke, fill) = &best.2[0];
            let drawn = stroke.shape(*fill);
            if let Some(rest) =
                thick_remainder(ctx, hierarchy, &place, &shape, std::slice::from_ref(&drawn))
            {
                let error = place.error_of(ctx, &[drawn, rest.clone()]);
                let with = (error, best.2.clone(), Some(rest), best.4.clone());
                let g = gain(free.0, &with);
                if g > best.0 {
                    best = (g, best.1, with.1, with.2, with.3);
                }
            }
        }
        let (gain, position, strokes, remainder, bridges) = best;
        let beyond = strokes
            .iter()
            .any(|(s, _)| draws_beyond(hierarchy, &shape, s, &bridges, ctx.width, ctx.height));
        let accepted = gain > 0. && !beyond;
        if !accepted {
            scene.shapes.insert(k, shape);
            scene.change();
            k += 1;
            continue;
        }
        // A band around shapes drawn just below it may be a fill under them instead
        let single = strokes.len() == 1 && remainder.is_none();
        let under = lower
            .iter()
            .filter(|&&j| single && j <= position)
            .find_map(|&j| {
                let mut chosen: Vec<Shape> = scene.shapes[j..k].to_vec();
                chosen.insert(position - j, strokes[0].0.shape(strokes[0].1));
                fill_under(ctx, scene, &shape, j, k, &chosen).map(|u| (j, u))
            });
        if let Some((j, under)) = under {
            let change = scene.change();
            let under = under.into_iter().map(|s| Shape {
                changed: change,
                fitted: change,
                ..s
            });
            scene.shapes.splice(j..k, under);
            scene.change();
            k += 1;
            continue;
        }
        // The first stroke keeps the shape's id; a network's further lines are new shapes
        // beside it, in the same place, and what they leave of the shape goes over them
        let count = strokes.len() + usize::from(remainder.is_some());
        let mut strokes = strokes.into_iter();
        let (first, fill) = strokes.next().expect("a stroke");
        let mut replaced = Shape {
            id: shape.id,
            clip: shape.clip,
            ..first.shape(fill)
        };
        let change = scene.change();
        replaced.changed = change;
        replaced.fitted = change;
        scene.shapes.insert(position, replaced);
        scene.change();
        let rest = strokes.map(|(stroke, fill)| stroke.shape(fill));
        for (at, s) in (position + 1..).zip(rest.chain(remainder)) {
            scene.insert(
                at,
                Shape {
                    clip: shape.clip,
                    ..s
                },
            );
        }
        // the lines joined across the shape's gaps draw over what the trace made of
        // those gaps: the patches that no longer pay go; the shapes between the lines
        // and the next are seen again, since the removals moved them
        let moved = drop_patches(ctx, scene, &bridges, position);
        k = moved + count;
    }
}

#[cfg(test)]
mod dash_tests {
    use super::*;

    #[test]
    fn dashes_cover_their_spans_only() {
        let stroke = Stroke {
            centerline: vec![line(Vec2::new(10., 20.), Vec2::new(110., 20.))],
            closed: false,
            width: 10.,
            cap: Cap::Butt,
            join: Join::Miter,
            corners: vec![false],
            dashes: Some(Dashes {
                array: vec![20., 10.],
                offset: 0.,
            }),
        };
        let ends = stroke.arc_lengths();
        let spans = stroke.dash_spans(&ends);
        assert_eq!(spans.len(), 4, "{spans:?}");
        assert!(
            (spans[0].0 - 0.).abs() < 0.01 && (spans[0].1 - 20.).abs() < 0.01,
            "{spans:?}"
        );
        assert!(
            (spans[3].0 - 90.).abs() < 0.01 && (spans[3].1 - 100.).abs() < 0.01,
            "{spans:?}"
        );
        let sub = stroke.between(spans[1].0, spans[1].1, &ends).0;
        let (a, b) = (
            sub.centerline[0][0],
            sub.centerline[sub.centerline.len() - 1][3],
        );
        assert!(
            a.distance(Vec2::new(40., 20.)) < 0.5 && b.distance(Vec2::new(60., 20.)) < 0.5,
            "{a:?} {b:?}"
        );
        let plain = Stroke {
            dashes: None,
            ..stroke.clone()
        };
        let plain_covered: f32 = plain
            .shape(Fill::solid(Vec3::ZERO))
            .raster()
            .values
            .iter()
            .sum();
        let one = stroke.between(spans[1].0, spans[1].1, &ends).0;
        let one_covered: f32 = one
            .shape(Fill::solid(Vec3::ZERO))
            .raster()
            .values
            .iter()
            .sum();
        let shape = stroke.shape(Fill::solid(Vec3::ZERO));
        let covered: f32 = shape.raster().values.iter().sum();
        assert!(
            (covered - 700.).abs() < 20.,
            "covered {covered}, plain {plain_covered}, one dash {one_covered}, loops {}",
            shape.loops.len()
        );
    }
}

#[cfg(test)]
mod dot_tests {
    use super::*;

    /// A dotted stroke, round caps on dashes of half a pixel, paints a disc of its width at
    /// every dash: the two caps of a dash that short meet in a disc.
    #[test]
    fn dots_paint_discs() {
        let (a, b) = (Vec2::new(10., 20.), Vec2::new(110., 20.));
        let width = 10.;
        let length = 0.5;
        let stroke = Stroke {
            centerline: vec![line(a, b)],
            closed: false,
            width,
            cap: Cap::Round,
            join: Join::Round,
            corners: vec![false],
            dashes: Some(Dashes {
                array: vec![length, 20. - length],
                offset: 0.,
            }),
        };
        let ends = stroke.arc_lengths();
        let spans = stroke.dash_spans(&ends);
        // At 0, 20, 40, 60 and 80, and the one at 100 has no length left
        assert_eq!(spans.len(), 5, "{spans:?}");
        let shape = stroke.shape(Fill::solid(Vec3::ZERO));
        let covered: f32 = shape.raster().values.iter().sum();
        let disc = std::f32::consts::PI * width * width / 4. + length * width;
        assert!(
            (covered - 5. * disc).abs() < 0.03 * 5. * disc,
            "covered {covered}, five discs {}",
            5. * disc
        );
        // Each dot is round: as wide along the path as across it, by the half pixel
        let one = stroke.between(spans[2].0, spans[2].1, &ends).0;
        let dot = one.shape(Fill::solid(Vec3::ZERO));
        let extent = dot.max - dot.min;
        assert!(
            (extent.x - width - length).abs() < 0.1 && (extent.y - width).abs() < 0.1,
            "{extent:?}"
        );
    }
}

#[cfg(test)]
mod network_tests {
    use super::*;

    /// A mask of `w` by `h` with the given 2 px lines drawn (a pixel of margin around, as
    /// [`mask_of`] has it).
    fn grid(w: usize, h: usize, verticals: &[usize], horizontals: &[usize]) -> Vec<bool> {
        let (bw, bh) = (w + 2, h + 2);
        let mut mask = vec![false; bw * bh];
        for y in 0..h {
            for x in 0..w {
                let on = verticals.iter().any(|&v| x == v || x == v + 1)
                    || horizontals.iter().any(|&u| y == u || y == u + 1);
                mask[(y + 1) * bw + x + 1] = on;
            }
        }
        mask
    }

    fn straight(line: &Skeleton) -> bool {
        let a = line.chain[0];
        line.chain
            .iter()
            .all(|p| (p.x - a.x).abs() <= 1 || (p.y - a.y).abs() <= 1)
    }

    /// Two lines crossing are two strokes through the junction, each straight from edge
    /// to edge.
    #[test]
    fn a_cross_is_two_lines() {
        let (w, h) = (60, 50);
        let mask = grid(w, h, &[30], &[24]);
        let Some(Skeletons::Network(lines)) = skeletons_of(&mask, w + 2, h + 2, IVec2::ZERO) else {
            panic!("a network");
        };
        let lengths: Vec<usize> = lines.iter().map(|l| l.chain.len()).collect();
        assert_eq!(lines.len(), 2, "{lengths:?}");
        for line in &lines {
            let (a, b) = (line.chain[0], line.chain[line.chain.len() - 1]);
            assert!(
                straight(line),
                "line from {a:?} to {b:?} wanders: {:?}",
                line.chain
            );
            let span = (w.min(h) - 6) as i32;
            assert!(a.distance_squared(b) > span * span, "{a:?} {b:?}");
        }
    }

    /// A grid of three by two lines is five strokes, each the whole way across.
    #[test]
    fn a_grid_is_its_lines() {
        let (w, h) = (90, 70);
        let mask = grid(w, h, &[20, 45, 70], &[22, 48]);
        let Some(Skeletons::Network(lines)) = skeletons_of(&mask, w + 2, h + 2, IVec2::ZERO) else {
            panic!("a network");
        };
        let mut lengths: Vec<usize> = lines.iter().map(|l| l.chain.len()).collect();
        lengths.sort_unstable();
        assert_eq!(lines.len(), 5, "{lengths:?}");
        for line in &lines {
            let (a, b) = (line.chain[0], line.chain[line.chain.len() - 1]);
            assert!(
                straight(line),
                "line from {a:?} to {b:?} wanders: {:?}",
                line.chain
            );
        }
    }
}
