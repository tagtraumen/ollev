//! Strokes that continue one another across a gap joined into one.

use super::*;

/// The bridge cubic and score where a stroke end at `pa` leaving along `ta` continues as one
/// at `pb` leaving along `tb`, by the rule of [`crate::translucency::completion::pair_ends`]:
/// tangents meeting ahead within a right angle, or collinear within `unit` plus a fifth.
pub(super) fn continuation(
    pa: Vec2,
    ta: Vec2,
    pb: Vec2,
    tb: Vec2,
    unit: f32,
) -> Option<(Cubic, f32)> {
    let arrive = -tb;
    // an end without a tangent continues nothing
    let turn = crate::tracer::angle::turn(ta, arrive)?.abs();
    if turn > std::f32::consts::FRAC_PI_2 {
        return None;
    }
    let ab = pb - pa;
    let dist = ab.length();
    if dist < 1e-6 {
        return None;
    }
    let offset = ta.perp_dot(ab).abs();
    let det = ta.perp_dot(tb);
    let meet = det.abs() > 1e-6 && {
        let s = ab.perp_dot(tb) / det;
        let t = ab.perp_dot(ta) / det;
        s >= 0. && t >= 0.
    };
    let collinear = ta.dot(ab) > 0. && offset <= 1.5 * unit + 0.2 * dist;
    if !meet && !collinear {
        return None;
    }
    let handle = if turn < 1e-3 {
        dist / 3.
    } else {
        let radius = dist / (2. * (turn / 2.).sin());
        (4. / 3. * (turn / 4.).tan() * radius).min(dist)
    };
    Some((
        [pa, pa + ta * handle, pb + tb * handle, pb],
        turn + offset / dist,
    ))
}

/// A stroke's two ends: the point and the tangent out of the stroke.
pub(super) fn ends_of(stroke: &Stroke) -> [(Vec2, Vec2); 2] {
    let first = stroke.centerline[0];
    let last = stroke.centerline[stroke.centerline.len() - 1];
    [
        (
            first[0],
            -crate::complete::tangent_in(&first).normalize_or_zero(),
        ),
        (
            last[3],
            crate::complete::tangent_out(&last).normalize_or_zero(),
        ),
    ]
}

/// A stroke's centerline run the other way, with its corner flags (a flag marks the joint
/// at its cubic's start).
pub(super) fn reversed(stroke: &Stroke) -> (Vec<Cubic>, Vec<bool>) {
    let n = stroke.centerline.len();
    let centerline: Vec<Cubic> = stroke
        .centerline
        .iter()
        .rev()
        .map(|c| [c[3], c[2], c[1], c[0]])
        .collect();
    let corners: Vec<bool> = (0..n)
        .map(|i| i > 0 && stroke.corners.get(n - i).copied().unwrap_or(false))
        .collect();
    (centerline, corners)
}

/// The stroke `a` and `b` make joined at their ends `ea` and `eb` by `bridge`, at their
/// length-weighted width and colour with the longer piece's caps and joins, and the bridge
/// alone as a stroke, for what it covers.
pub(super) fn joined(
    a: (&Stroke, &Fill),
    ea: usize,
    b: (&Stroke, &Fill),
    eb: usize,
    bridge: Cubic,
) -> ((Stroke, Fill), Shape) {
    let ((sa, fa), (sb, fb)) = (a, b);
    let la = sa.arc_lengths().last().copied().unwrap_or(0.).max(1e-3);
    let lb = sb.arc_lengths().last().copied().unwrap_or(0.).max(1e-3);
    let width = (sa.width * la + sb.width * lb) / (la + lb);
    let fill = Fill {
        color: (fa.color * la + fb.color * lb) / (la + lb),
        opacity: (fa.opacity * la + fb.opacity * lb) / (la + lb),
        gradient: None,
    };
    let (mut centerline, mut corners) = if ea == 0 {
        reversed(sa)
    } else {
        (sa.centerline.clone(), sa.corners.clone())
    };
    let (cb, mut kb) = if eb == 1 {
        reversed(sb)
    } else {
        (sb.centerline.clone(), sb.corners.clone())
    };
    centerline.push(bridge);
    corners.push(false);
    if let Some(k) = kb.first_mut() {
        *k = false;
    }
    centerline.extend(cb);
    corners.extend(kb);
    let (cap, join) = if la >= lb {
        (sa.cap, sa.join)
    } else {
        (sb.cap, sb.join)
    };
    let stroke = Stroke {
        centerline,
        closed: false,
        width,
        cap,
        join,
        corners,
        dashes: None,
    };
    let bridge = Stroke {
        centerline: vec![bridge],
        corners: vec![false],
        cap: Cap::Butt,
        ..stroke.clone()
    }
    .shape(fill);
    ((stroke, fill), bridge)
}

/// What a bridge's visible, non-free pixels cost drawn in its colour, and as drawn now under
/// `was` (the shape's fill where it covers them, the render below elsewhere; nothing without
/// `was`). A join pays where the first is under the second plus a path and width segment.
pub(super) fn bridge_cost(
    ctx: &Context,
    bridge: &Shape,
    was: Option<(&Place, &Shape)>,
) -> (f64, f64) {
    let r = bridge.raster();
    let (mut line, mut before) = (0f64, 0f64);
    for y in r.y0..(r.y0 + r.h).min(ctx.height) {
        for x in r.x0..(r.x0 + r.w).min(ctx.width) {
            let c = r.values[(y - r.y0) * r.w + x - r.x0];
            let i = y * ctx.width + x;
            if c <= 0. || ctx.free.as_deref().is_some_and(|f| f[i]) {
                continue;
            }
            let (weight, target) = ctx.weighted_target(i);
            let Some((place, shape)) = was else {
                line += (weight * c * (target - bridge.fill.color).length_squared()) as f64;
                continue;
            };
            let Some(k) = place.window.index(i, ctx.width) else {
                continue;
            };
            if place
                .above
                .as_ref()
                .is_some_and(|a| a.transmittance[k].max_element() < 0.5)
            {
                continue;
            }
            let shown = if shape.raster().at(i, ctx.width) >= 0.5 {
                shape.fill.at_pixel(x, y)
            } else {
                place.below[k]
            };
            line += (weight * c * (target - bridge.fill.color).length_squared()) as f64;
            before += (weight * c * (target - shown).length_squared()) as f64;
        }
    }
    (line, before)
}

/// The relatable pairs of ends among `strokes` ([`continuation`]), best first: the two
/// strokes, their ends, the bridge.
pub(super) fn continuations(
    ctx: &Context,
    strokes: &[&Stroke],
    same: impl Fn(usize, usize) -> bool,
) -> Vec<(usize, usize, usize, usize, Cubic)> {
    let unit = crate::translucency::raster::unit_of(ctx.width) as f32;
    let ends: Vec<[(Vec2, Vec2); 2]> = strokes.iter().map(|s| ends_of(s)).collect();
    let mut candidates: Vec<(f32, usize, usize, usize, usize, Cubic)> = Vec::new();
    for i in 0..strokes.len() {
        for j in i + 1..strokes.len() {
            if !same(i, j) {
                continue;
            }
            for ea in 0..2 {
                for eb in 0..2 {
                    let (pa, ta) = ends[i][ea];
                    let (pb, tb) = ends[j][eb];
                    if let Some((bridge, score)) = continuation(pa, ta, pb, tb, unit) {
                        candidates.push((score, i, ea, j, eb, bridge));
                    }
                }
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    candidates
        .into_iter()
        .map(|(_, i, ea, j, eb, bridge)| (i, ea, j, eb, bridge))
        .collect()
}

/// A network's lines joined, best pairs first and repeatedly, where one continues another
/// across a gap outside the shape and the bridge pays ([`bridge_cost`]), as lines parted by
/// something lying over them. Returns the lines and the bridges.
pub(super) fn join_lines(
    ctx: &Context,
    place: &Place,
    shape: &Shape,
    mut lines: Vec<(Stroke, Fill)>,
) -> (Vec<(Stroke, Fill)>, Vec<Shape>) {
    let mut bridges: Vec<Shape> = Vec::new();
    loop {
        let strokes: Vec<&Stroke> = lines.iter().map(|(s, _)| s).collect();
        let open: Vec<bool> = strokes.iter().map(|s| !s.closed).collect();
        let candidates = continuations(ctx, &strokes, |i, j| open[i] && open[j]);
        let mut made = false;
        for (i, ea, j, eb, bridge) in candidates {
            let ((stroke, fill), bridge) = joined(
                (&lines[i].0, &lines[i].1),
                ea,
                (&lines[j].0, &lines[j].1),
                eb,
                bridge,
            );
            let (cost, was) = bridge_cost(ctx, &bridge, Some((place, shape)));
            let saving = ctx.segment_cost + ctx.path_cost;
            if cost >= was + saving {
                continue;
            }
            lines.remove(j);
            lines.remove(i);
            lines.push((stroke, fill));
            bridges.push(bridge);
            made = true;
            break;
        }
        if !made {
            return (lines, bridges);
        }
    }
}

/// Takes out of `scene` the shapes within a bridge's reach that no longer pay once the joined
/// stroke runs through (the patches the trace made of its gaps). `keep` is the joined stroke's
/// position, which the removals move; its position after is returned.
pub(super) fn drop_patches(
    ctx: &Context,
    scene: &mut Scene,
    bridges: &[Shape],
    keep: usize,
) -> usize {
    let unit = crate::translucency::raster::unit_of(ctx.width) as f32;
    let mut keep = keep;
    for bridge in bridges {
        let reach = bridge.stroke.as_ref().map_or(unit, |s| s.width.max(unit));
        let (bmin, bmax) = (bridge.min - reach, bridge.max + reach);
        let mut k = scene.shapes.len();
        while k > 0 {
            k -= 1;
            if k == keep {
                continue;
            }
            let s = &scene.shapes[k];
            if !(bmin.cmple(s.min).all() && s.max.cmple(bmax).all()) {
                continue;
            }
            let shape = scene.shapes.remove(k);
            scene.change();
            let kept = Place::new(ctx, scene, &shape, k, MARGIN).is_none_or(|place| {
                let with = place.error(ctx, &shape) as f64
                    + shape.cost_segments() * ctx.segment_cost
                    + ctx.path_cost;
                with <= place.before as f64
            });
            if kept {
                scene.shapes.insert(k, shape);
                scene.change();
            } else if k < keep {
                keep -= 1;
            }
        }
    }
    keep
}

/// Joins the scene's strokes that continue one another across a gap ([`continuation`]),
/// best pairs first and repeatedly, kept where error plus price falls, and drops what the
/// joined stroke draws over ([`drop_patches`]), so a line cut in pieces becomes one.
pub fn continue_through(ctx: &Context, scene: &mut Scene) {
    let price = |segments: f64| segments * ctx.segment_cost + ctx.path_cost;
    let mut joins = 0;
    loop {
        // the open strokes of the scene, by position
        let positions: Vec<usize> = scene
            .shapes
            .iter()
            .enumerate()
            .filter_map(|(position, shape)| {
                let stroke = shape.stroke.as_ref()?;
                (shape.clip.is_none()
                    && shape.fill.gradient.is_none()
                    && !stroke.closed
                    && stroke.dashes.is_none())
                .then_some(position)
            })
            .collect();
        let strokes: Vec<&Stroke> = positions
            .iter()
            .map(|&p| scene.shapes[p].stroke.as_ref().expect("a stroke"))
            .collect();
        let candidates = continuations(ctx, &strokes, |i, j| {
            scene.shapes[positions[i]].layer == scene.shapes[positions[j]].layer
        });
        let mut made = false;
        for (i, ea, j, eb, bridge) in candidates {
            let (pi, pj) = (positions[i], positions[j]);
            let (a, b) = (&scene.shapes[pi], &scene.shapes[pj]);
            let ((stroke, fill), bridge) = joined(
                (a.stroke.as_ref().expect("a stroke"), &a.fill),
                ea,
                (b.stroke.as_ref().expect("a stroke"), &b.fill),
                eb,
                bridge,
            );
            let (bound, _) = bridge_cost(ctx, &bridge, None);
            if bound >= ctx.segment_cost + ctx.path_cost {
                continue;
            }
            let (lo, hi) = (pi.min(pj), pi.max(pj));
            let mut framed = stroke.shape(fill);
            framed.min = framed.min.min(a.min).min(b.min);
            framed.max = framed.max.max(a.max).max(b.max);
            let joined_shape = Shape {
                layer: a.layer,
                smooth: a.smooth || b.smooth,
                antialiased: a.antialiased || b.antialiased,
                ..stroke.shape(fill)
            };
            let price_before = price(a.cost_segments()) + price(b.cost_segments());
            let price_after = price(joined_shape.cost_segments());
            // the window as it is, the lower piece taken out and measured back in its
            // place with the higher one still above; then both out and the joined one in
            let shape_lo = scene.shapes.remove(lo);
            scene.change();
            let before = Place::new(ctx, scene, &framed, lo, MARGIN)
                .map(|place| place.error(ctx, &shape_lo));
            let shape_hi = scene.shapes.remove(hi - 1);
            scene.change();
            let after = Place::new(ctx, scene, &framed, lo, MARGIN)
                .map(|place| place.error(ctx, &joined_shape));
            let accepted = match (before, after) {
                (Some(before), Some(after)) => {
                    after as f64 + price_after < before as f64 + price_before
                }
                _ => false,
            };
            if !accepted {
                scene.shapes.insert(lo, shape_lo);
                scene.shapes.insert(hi, shape_hi);
                scene.change();
                continue;
            }
            scene.insert(lo, joined_shape);
            joins += 1;
            drop_patches(ctx, scene, std::slice::from_ref(&bridge), lo);
            made = true;
            break;
        }
        if !made {
            break;
        }
    }
    if crate::diagnostics::log() && joins > 0 {
        eprintln!("continue: {joins} strokes joined through their gaps");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stroke end without a tangent continues nothing.
    #[test]
    fn an_end_without_a_tangent_continues_nothing() {
        let (a, b) = (Vec2::ZERO, Vec2::new(6., 0.));
        assert!(continuation(a, Vec2::X, b, Vec2::ZERO, 1.).is_none());
        assert!(continuation(a, Vec2::ZERO, b, -Vec2::X, 1.).is_none());
        assert!(continuation(a, Vec2::X, b, -Vec2::X, 1.).is_some());
    }
}
