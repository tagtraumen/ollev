//! A dashed stroke from its members: the centerline, the spans and the dash pattern.

use super::*;

/// The stroke a shape is: its own, or for a rectangle or stadium primitive (a straight dash
/// the stroke stage kept as a primitive) the band along its long axis with butt or round caps.
pub(super) fn stroke_of(shape: &Shape) -> Option<Stroke> {
    if let Some(stroke) = &shape.stroke {
        return Some(stroke.clone());
    }
    let (
        _,
        crate::tracer::primitives::Primitive::RoundedRect {
            center,
            half,
            radius,
            angle,
        },
    ) = shape.primitive?
    else {
        return None;
    };
    let (long, short) = if half.x >= half.y {
        (half.x, half.y)
    } else {
        (half.y, half.x)
    };
    if long <= short {
        return None;
    }
    let cap = if radius <= 0.5 {
        Cap::Butt
    } else if (radius - short).abs() <= 0.5 {
        Cap::Round
    } else {
        return None;
    };
    let reach = if cap == Cap::Round { short } else { 0. };
    let axis = if half.x >= half.y {
        Vec2::from_angle(angle)
    } else {
        Vec2::from_angle(angle).perp()
    };
    let (a, b) = (
        center - axis * (long - reach),
        center + axis * (long - reach),
    );
    if a.distance(b) < 1. {
        return None;
    }
    Some(Stroke {
        centerline: vec![line(a, b)],
        closed: false,
        width: 2. * short,
        cap,
        join: Join::Miter,
        corners: vec![false],
        dashes: None,
    })
}

/// One stroke through the member strokes (in order along the path, each with whether it runs
/// reversed): the centerline fitted through theirs, their median width, a dash at each.
pub(super) fn dashed_stroke(members: &[(&Stroke, bool)], closed: bool) -> Option<Stroke> {
    let strokes: Vec<&Stroke> = members.iter().map(|m| m.0).collect();
    // Samples along every member, about a pixel apart, each member the chain's way round
    let mut samples: Vec<Vec2> = Vec::new();
    let mut member_ranges: Vec<(usize, usize)> = Vec::new();
    for &(stroke, reversed) in members {
        let from = samples.len();
        let mut own: Vec<Vec2> = Vec::new();
        for c in &stroke.centerline {
            let length = hull_length(c);
            let steps = (length.ceil() as usize).max(1);
            for i in 0..steps {
                own.push(eval(c, i as f32 / steps as f32));
            }
        }
        let last = &stroke.centerline[stroke.centerline.len() - 1];
        own.push(last[3]);
        if reversed {
            own.reverse();
        }
        samples.extend(own);
        member_ranges.push((from, samples.len() - 1));
    }
    if samples.len() < 4 {
        return None;
    }
    let centerline = fit_centerline(&samples, closed, CENTERLINE_TOLERANCE)?;
    let mut widths: Vec<f32> = strokes.iter().map(|s| s.width).collect();
    widths.sort_by(f32::total_cmp);
    let m = centerline.len();
    let cap = [Cap::Butt, Cap::Round, Cap::Square]
        .into_iter()
        .max_by_key(|c| strokes.iter().filter(|s| s.cap == *c).count())
        .unwrap_or(Cap::Butt);
    let mut stroke = Stroke {
        centerline,
        closed,
        width: widths[widths.len() / 2],
        cap,
        join: strokes[0].join,
        corners: vec![false; m],
        dashes: None,
    };
    // Where each member lies along the fitted centerline: the arc length of the nearest
    // flattened point to its start and to its end
    let ends = stroke.arc_lengths();
    let total = ends[ends.len() - 1];
    let along = arc_table(&stroke);
    let arc_at = |q: Vec2| nearest_arc(&along, q);
    // A member with a round or square cap paints half a width past its centerline's ends, so
    // its dash reaches that much farther; the chain's own cap's reach is then taken back
    let half = stroke.width / 2.;
    let reach = |c: Cap| {
        if c == Cap::Butt { 0. } else { half }
    };
    let mut spans: Vec<(f32, f32)> = member_ranges
        .iter()
        .zip(&strokes)
        .map(|(&(from, to), member)| {
            let r = reach(member.cap);
            (arc_at(samples[from]) - r, arc_at(samples[to]) + r)
        })
        .collect();
    extend_to_spans(&mut stroke, &mut spans, total);
    let own_reach = reach(stroke.cap);
    for span in &mut spans {
        span.0 += own_reach;
        span.1 -= own_reach;
    }
    stroke.dashes = Some(pattern_of_spans(&mut spans, closed, total));
    Some(stroke)
}

/// The centerline through `samples` in order, as the stroke stage fits a skeleton's chain:
/// corners where it turns by a kink, within `tolerance` of the samples, then refitted to them
/// by least squares.
pub(super) fn fit_centerline(
    samples: &[Vec2],
    closed: bool,
    tolerance: f32,
) -> Option<Vec<crate::tracer::Cubic>> {
    let params = crate::tracer::FitParams {
        corner_angle: KINK,
        tolerance,
        ..crate::outline::FIT
    };
    let chain: Vec<Vec2> = if closed {
        [samples, &samples[..1]].concat()
    } else {
        samples.to_vec()
    };
    let centerline = refit(
        &crate::tracer::fit::fit_at_corners(&chain, &[], closed, &params),
        &chain,
    );
    (!centerline.is_empty()).then_some(centerline)
}

/// Points along the stroke's centerline with their arc length, about a pixel apart (not a
/// fixed count per cubic, which is too coarse on a cubic spanning several dashes).
pub(super) fn arc_table(stroke: &Stroke) -> Vec<(f32, Vec2)> {
    let mut along: Vec<(f32, Vec2)> = Vec::new();
    let mut s = 0f32;
    for c in &stroke.centerline {
        let length = hull_length(c);
        let steps = (length.ceil() as usize).max(1);
        let mut from = c[0];
        along.push((s, from));
        for i in 1..=steps {
            let to = eval(c, i as f32 / steps as f32);
            s += from.distance(to);
            along.push((s, to));
            from = to;
        }
    }
    along
}

/// The arc length of the point of `along` (see [`arc_table`]) nearest to `q`.
pub(super) fn nearest_arc(along: &[(f32, Vec2)], q: Vec2) -> f32 {
    along
        .iter()
        .min_by(|a, b| a.1.distance(q).total_cmp(&b.1.distance(q)))
        .map_or(0., |a| a.0)
}

/// Extends an open stroke's centerline to the spans of paint before its start and past its
/// end (`total` along it), and moves the spans with it: the path itself starts and ends with
/// the paint, so a reach at either end extends it.
pub(super) fn extend_to_spans(stroke: &mut Stroke, spans: &mut [(f32, f32)], total: f32) {
    if stroke.closed {
        return;
    }
    let (lead, trail) = (
        -spans[0].0.min(0.),
        (spans[spans.len() - 1].1 - total).max(0.),
    );
    if lead > 0. || trail > 0. {
        stroke.extend_ends(lead, trail);
        for span in spans.iter_mut() {
            span.0 += lead;
            span.1 += lead;
        }
    }
}

/// Extends an open dashed stroke's ends where they cut a dash to less than [`DOT_LENGTH`]: a
/// dash that short has no direction for its caps, and whether a renderer draws it at all turns
/// on its own arc length.
pub(super) fn whole_end_dashes(stroke: &mut Stroke) {
    let Some(dashes) = stroke.dashes.clone() else {
        return;
    };
    let period: f32 = dashes.array.iter().sum();
    if stroke.closed || period <= 1e-3 || dashes.array.len() % 2 == 1 {
        return;
    }
    // At pattern position `p`: the dash it lies in, as how far into it and its length
    let within = |p: f32| -> Option<(f32, f32)> {
        let mut at = 0f32;
        let p = p.rem_euclid(period);
        for (k, &len) in dashes.array.iter().enumerate() {
            if p < at + len {
                return (k % 2 == 0).then_some((p - at, len));
            }
            at += len;
        }
        None
    };
    let ends = stroke.arc_lengths();
    let total = ends[ends.len() - 1];
    let short = |drawn: f32, len: f32| (len.min(DOT_LENGTH) - drawn).max(0.);
    // the dash at the start shows from `into` on; the one at the end up to `into`
    let lead = within(dashes.offset).map_or(0., |(into, len)| short(len - into, len));
    let trail = within(dashes.offset + total).map_or(0., |(into, len)| short(into, len));
    if lead > 0. || trail > 0. {
        stroke.extend_ends(lead, trail);
        stroke.dashes = Some(Dashes {
            offset: (dashes.offset - lead).rem_euclid(period),
            ..dashes
        });
    }
    if stroke.cap == Cap::Round {
        clear_end_dots(stroke);
    }
}

/// Runs a round-capped stroke on past a dot (a dash under [`DOT_LENGTH`]) lying within an
/// anti-aliasing step of an end, so no renderer's arc length or rounding puts it past the end;
/// a step at most, since stretched farther the end cubics move the dots.
fn clear_end_dots(stroke: &mut Stroke) {
    let Some(dashes) = stroke.dashes.clone() else {
        return;
    };
    let period: f32 = dashes.array.iter().sum();
    if stroke.closed || period <= 1e-3 {
        return;
    }
    let ends = stroke.arc_lengths();
    let total = ends[ends.len() - 1];
    // the dots along the centerline, its ends included
    let mut dots: Vec<f32> = Vec::new();
    let (mut at, mut k) = (-dashes.offset.rem_euclid(period), 0);
    while at <= total && k < 1 << 16 {
        let len = dashes.array[k % dashes.array.len()];
        if k % 2 == 0 && len < DOT_LENGTH && at >= 0. {
            dots.push(at);
        }
        at += len;
        k += 1;
    }
    let margin = crate::edges::SIGMA as f32;
    let lead = dots.first().map_or(0., |&s| (margin - s).max(0.));
    let trail = dots.last().map_or(0., |&s| (margin - (total - s)).max(0.));
    if lead > 0. || trail > 0. {
        stroke.extend_ends(lead, trail);
        stroke.dashes = Some(Dashes {
            offset: (dashes.offset - lead).rem_euclid(period),
            ..dashes
        });
    }
}

/// The pattern of dashes at `spans` of arc length along a centerline of length `total`, in
/// order, see [`pattern`].
pub(super) fn pattern_of_spans(spans: &mut [(f32, f32)], closed: bool, total: f32) -> Dashes {
    // Along a closed path the last member may wrap past the seam
    for span in spans.iter_mut() {
        if span.1 < span.0 {
            span.1 += total;
        }
    }
    let (lengths, gaps) = lengths_and_gaps(spans, closed, total);
    pattern(&lengths, &gaps, spans[0].0)
}

/// Settles the pattern's lengths and offset on the error by coordinate descent, in steps down
/// to where the farthest dash moves by a quarter pixel, since a small error per period adds up
/// over a chain. Returns the error.
pub(super) fn tune_pattern(
    cost: &crate::render::coverage_cost::CoverageCost,
    stroke: &mut Stroke,
    origin: Vec2,
) -> f32 {
    let mut best = cost.value(&stroke.path(origin));
    let Some(mut current) = stroke.dashes.clone() else {
        return best;
    };
    let n = current.array.len();
    let ends = stroke.arc_lengths();
    let period: f32 = current.array.iter().sum();
    let repeats = (ends[ends.len() - 1] / period.max(1e-3)).max(1.);
    let mut steps = vec![1f32, 0.5, 0.25];
    while steps[steps.len() - 1] * repeats > 0.25 {
        let last = steps[steps.len() - 1];
        steps.push(last / 2.);
    }
    for step in steps {
        let mut improved = true;
        while improved {
            improved = false;
            // Each entry of the pattern, and the offset
            for k in 0..=n {
                for sign in [-1., 1.] {
                    let mut trial = current.clone();
                    if k < n {
                        // A dash keeps a length for its caps, a gap may close
                        let least = if k % 2 == 0 { shortest(stroke.cap) } else { 0. };
                        trial.array[k] = (trial.array[k] + sign * step).max(least);
                    } else {
                        trial.offset += sign * step;
                    }
                    stroke.dashes = Some(trial.clone());
                    let e = cost.value(&stroke.path(origin));
                    if e < best {
                        best = e;
                        current = trial;
                        improved = true;
                    }
                }
            }
        }
    }
    stroke.dashes = Some(current);
    // Snapped to the pixel grid where that helps, as the outline cleanup snaps lines to
    // the axes: a dashed line of pixel-exact dashes is drawn on whole pixels, and the fit
    // lands a tenth of a pixel off
    let snapped = |v: f32| (v * 2.).round() / 2.;
    let mut trial = stroke.clone();
    trial.width = snapped(trial.width);
    for c in &mut trial.centerline {
        for p in c.iter_mut() {
            *p = Vec2::new(snapped(p.x), snapped(p.y));
        }
    }
    let e = cost.value(&trial.path(origin));
    if e < best {
        *stroke = trial;
        best = e;
    }
    best
}

/// The dash pattern that lays `lengths` of dashes with `gaps` between them, the first
/// starting at `start` along the path: the median dash and gap where they are all alike
/// within the width slack, else the shortest period that repeats, else the whole sequence.
pub(super) fn pattern(lengths: &[f32], gaps: &[f32], start: f32) -> Dashes {
    let median = |v: &[f32]| -> f32 {
        let mut w = v.to_vec();
        w.sort_by(f32::total_cmp);
        w[w.len() / 2]
    };
    let alike = |v: &[f32], m: f32| {
        v.iter()
            .all(|&x| (x - m).abs() <= (WIDTH_SLACK * m).max(WIDTH_QUANTUM))
    };
    let n = lengths.len();
    // The period is the mean start-to-start distance over the whole chain, so the dashes stay
    // in step to the end; within it the dashes keep their median lengths, the gaps the rest
    let starts: Vec<f32> = {
        let mut at = 0f32;
        (0..n)
            .map(|k| {
                let s = at;
                at += lengths[k] + gaps.get(k).copied().unwrap_or(0.);
                s
            })
            .collect()
    };
    let sequence = |period: usize| -> Vec<f32> {
        let repeats = n / period;
        let span = if repeats > 1 {
            (starts[(repeats - 1) * period] - starts[0]) / (repeats - 1) as f32
        } else {
            (0..period)
                .map(|k| lengths[k] + gaps.get(k).copied().unwrap_or(0.))
                .sum()
        };
        let mut out = Vec::new();
        for k in 0..period {
            out.push(median(
                &(k..n)
                    .step_by(period)
                    .map(|i| lengths[i])
                    .collect::<Vec<_>>(),
            ));
            let g: Vec<f32> = (k..gaps.len()).step_by(period).map(|i| gaps[i]).collect();
            out.push(if g.is_empty() {
                out[out.len() - 1]
            } else {
                median(&g)
            });
        }
        // The gaps scaled so the period comes out as the chain has it
        let dashes_total: f32 = (0..period).map(|k| out[2 * k]).sum();
        let gaps_total: f32 = (0..period).map(|k| out[2 * k + 1]).sum();
        if gaps_total > 1e-3 {
            let scale = ((span - dashes_total) / gaps_total).max(0.);
            for k in 0..period {
                out[2 * k + 1] *= scale;
            }
        }
        out
    };
    let fits = |period: usize| -> bool {
        (0..period).all(|k| {
            let l: Vec<f32> = (k..n).step_by(period).map(|i| lengths[i]).collect();
            let g: Vec<f32> = (k..gaps.len()).step_by(period).map(|i| gaps[i]).collect();
            alike(&l, median(&l)) && (g.is_empty() || alike(&g, median(&g)))
        })
    };
    let array = (1..=MAX_PERIOD.min(n / 2).max(1))
        .find(|&p| fits(p))
        .map_or_else(
            || {
                let last = gaps.last().copied().unwrap_or(0.);
                (lengths.iter().enumerate())
                    .flat_map(|(k, &length)| [length, gaps.get(k).copied().unwrap_or(last)])
                    .collect()
            },
            sequence,
        );
    // The centerline starts at the first dash's start, `start` along it
    let period: f32 = array.iter().sum();
    let offset = if period > 0. {
        (-start).rem_euclid(period)
    } else {
        0.
    };
    Dashes { array, offset }
}
