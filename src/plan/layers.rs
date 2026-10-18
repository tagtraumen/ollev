//! The translucency decision: the families accepted whole, the beneath beneath them.

use super::*;

/// The families proposed on a level past the gates at their pin ([`GATE`] materials meeting,
/// see `rank::contact`, and the pin on the head's relation's line), most likely first.
pub(super) async fn accepted_families(
    ctx: &Context,
    scene: &Scene,
    leaves: &[u32],
) -> Vec<crate::translucency::Candidate> {
    crate::translucency::families::ranked_candidates(ctx, scene, leaves, GATE)
        .await
        .into_iter()
        .map(|r| {
            level_log(format_args!(
                "family {} {}: {} materials meeting ({} beneath), main image {:.1} levels off the growth relation, tried",
                r.index,
                describe(&r.candidate),
                r.evidence.contact.meeting,
                r.meeting_beneath,
                r.apart * 255.,
            ));
            r.candidate
        })
        .collect()
}

/// A candidate for the levels log: its k, colour, pixels and overlap with the truth.
pub(super) fn describe(c: &crate::translucency::Candidate) -> String {
    format!(
        "k {:.3} c {:?} {} px{}",
        c.k,
        (c.fill().color * 255.).round().to_array(),
        c.mask.iter().filter(|&&m| m).count(),
        truth_iou(&c.mask).map_or(String::new(), |v| format!(" IoU {v:.2}"))
    )
}

/// Each family decided by the whole level, not its window: the image beneath traced whole, the
/// family put back on top; per family the scene with it and its objective (the opaque
/// `scene`'s plus the margin). The caller accepts a family on its gate's evidence.
#[allow(clippy::too_many_arguments)]
pub(super) async fn decide_whole(
    ctx: &Context,
    scene: &Scene,
    families: Vec<crate::translucency::Candidate>,
    above: Option<&[bool]>,
    detail: f64,
    options: Options,
    timings: &mut Timings,
) -> Vec<(Scene, f64, crate::translucency::Candidate)> {
    let without = parts(ctx, scene);
    let mut done = Vec::new();
    // the traces beneath are a level deeper, for the log
    crate::progress::nest(true);
    for (n, candidate) in families.into_iter().enumerate() {
        crate::progress::stage("beneath").await;
        let (width, height) = (ctx.width, ctx.height);
        let Some(overlay) = candidate.overlay(width, height) else {
            continue;
        };
        let clock = crate::timing::Clock::now();
        let peeled = candidate.peeled(ctx, &overlay);
        // No discovery beneath a family: its beneath is inpainted under the holes, so a
        // relation found there reads the inpainting, not the picture
        let with = beneath_with(
            ctx,
            scene,
            &candidate,
            above,
            overlay,
            &peeled,
            detail,
            options,
            false,
            timings,
            false,
        )
        .await;
        let with_parts = parts(ctx, &with);
        let terms = LayerTerms::measure(ctx, &with, scene, &candidate);
        let better = without.total()
            + margin(
                &with_parts,
                &without,
                shading_discount(ctx, &candidate),
                &terms,
                ctx.segment_cost,
            );
        level_log(format_args!(
            "whole family {n} {}: {} with {better:.0} (error {:.0}, {:.0} segments, {} paths; shared {:.1}, meeting {}, under {:.0} / {:.0}, seen {:.0} / {:.0}) without {:.0} (error {:.0}, {:.0} segments, {} paths) in {:.1?}",
            describe(&candidate),
            "ACCEPTED on the relation",
            with_parts.error,
            with_parts.segments,
            with_parts.paths,
            terms.shared,
            terms.meeting,
            terms.under_with,
            terms.under_without,
            terms.seen_with,
            terms.seen_without,
            without.total(),
            without.error,
            without.segments,
            without.paths,
            clock.elapsed()
        ));
        done.push((with, better, candidate));
    }
    crate::progress::nest(false);
    done
}

/// For the record: a region's IoU against the truth's mask the caller gave (see
/// [`crate::diagnostics::Diagnostics::truth`]), where it is the frame's size.
#[cfg(feature = "diagnostics")]
pub(super) fn truth_iou(mask: &[bool]) -> Option<f64> {
    let truth = crate::diagnostics::truth();
    let t = truth.as_ref().filter(|t| t.len() == mask.len())?;
    let inter = mask
        .iter()
        .zip(t.iter())
        .filter(|(a, b)| **a && **b)
        .count();
    let union = mask
        .iter()
        .zip(t.iter())
        .filter(|(a, b)| **a || **b)
        .count();
    Some(inter as f64 / union.max(1) as f64)
}

/// Without the diagnostics there is no truth to score against.
#[cfg(not(feature = "diagnostics"))]
pub(super) fn truth_iou(_: &[bool]) -> Option<f64> {
    None
}

/// A line of the levels log of the translucency decisions, indented by the level.
pub(super) fn level_log(line: std::fmt::Arguments) {
    if crate::diagnostics::levels() {
        let level = crate::progress::depth();
        eprintln!("level {level}:{} {line}", "  ".repeat(level));
    }
}

/// The image beneath a candidate traced whole, by every step of a level (with `levels`, its
/// own candidates included), with the shape put back on top and fitted against `ctx`'s image.
#[allow(clippy::too_many_arguments)]
pub(super) async fn beneath_with(
    ctx: &Context,
    opaque: &Scene,
    candidate: &crate::translucency::Candidate,
    above: Option<&[bool]>,
    overlay: crate::translucency::Overlay,
    peeled: &[u8],
    detail: f64,
    options: Options,
    levels: bool,
    timings: &mut Timings,
    show: bool,
) -> Scene {
    let (mut with, on_top, covered) = beneath_layers(
        ctx, opaque, candidate, above, peeled, detail, options, levels, timings, show,
    )
    .await;
    timings
        .time("overlay", async || {
            append_overlay(
                ctx,
                &mut with,
                overlay,
                candidate,
                covered.as_deref().map(Vec::as_slice),
                options.stages.primitives,
            );
        })
        .await;
    if let Some(on_top) = on_top {
        let start = with.shapes.len();
        push_all(&mut with, on_top);
        timings
            .time("above", async || refit_above(ctx, &mut with, start).await)
            .await;
    }
    with
}

/// Refits the parts above (shapes from `start` up) over the completed beneath and the layer's
/// shape: lifted from the opaque trace, their rims were fitted against what that trace drew.
async fn refit_above(ctx: &Context, scene: &mut Scene, start: usize) {
    let fit = Context {
        weights: None,
        ..ctx.clone()
    };
    for position in start..scene.shapes.len() {
        crate::progress::poll().await;
        if scene.shapes[position].stroke.is_some() || scene.shapes[position].primitive.is_some() {
            continue;
        }
        let mut shape = scene.shapes.remove(position);
        scene.change();
        if let Some(place) = Place::new(&fit, scene, &shape, position, MARGIN) {
            place.refine(&fit, &mut shape);
        }
        scene.shapes.insert(position, shape);
        scene.change();
    }
}

/// The scene beneath a candidate, the parts above (to go over the layer's shape) and the hole
/// pixels they cover. With holes, what they hold is lifted out of `opaque` ([`parts_above`])
/// and the beneath completed and traced free there; without, the peeled image is traced whole.
#[allow(clippy::too_many_arguments)]
async fn beneath_layers(
    ctx: &Context,
    opaque: &Scene,
    candidate: &crate::translucency::Candidate,
    above: Option<&[bool]>,
    peeled: &[u8],
    detail: f64,
    options: Options,
    levels: bool,
    timings: &mut Timings,
    show: bool,
) -> (Scene, Option<Scene>, Option<std::sync::Arc<Vec<bool>>>) {
    let (width, height) = (ctx.width, ctx.height);
    let holes = candidate.holes.clone();
    // the layer's region beside the completed beneath in the peel directory, as `region-N.pgm`
    #[cfg(feature = "diagnostics")]
    if let (Some(dir), Some(_)) = (crate::diagnostics::peel_dir(), &holes) {
        let level = std::fs::read_dir(&dir)
            .map(|d| d.filter(|e| e.is_ok()).count())
            .unwrap_or(0);
        let mut pgm = format!("P5\n{width} {height}\n255\n").into_bytes();
        pgm.extend(candidate.mask.iter().map(|&h| if h { 255u8 } else { 0 }));
        let _ = std::fs::write(dir.join(format!("region-{level}.pgm")), pgm);
    }
    // The parts above first: the beneath is completed only where they cover, since a hole
    // pixel no lifted shape covers belongs to a shape of the beneath
    let on_top = match holes.as_deref() {
        Some(h) => Some(
            timings
                .time("above", async || parts_above(opaque, h, &candidate.mask))
                .await,
        ),
        None => None,
    };
    // And the holes' blends, the anti-aliased rims that mix a part above with the beneath (see
    // [`crate::translucency::completion::complete_holes_with_blend`]): colours between the
    // hole's and the clear beneath near it, so a gap between two parts above is not a blend
    let holes = holes.zip(on_top.as_ref()).map(|(h, top)| {
        let (mut covered, mut touched) = (vec![false; width * height], vec![false; width * height]);
        for s in &top.shapes {
            let r = s.raster();
            for y in 0..r.h.min(height.saturating_sub(r.y0)) {
                for x in 0..r.w.min(width.saturating_sub(r.x0)) {
                    let v = r.values[y * r.w + x];
                    let i = (y + r.y0) * width + x + r.x0;
                    if v > 0.5 {
                        covered[i] = true;
                    }
                    if v > 0. {
                        touched[i] = true;
                    }
                }
            }
        }
        let core: Vec<bool> = h.iter().zip(&covered).map(|(&a, &b)| a && b).collect();
        let colour = |i: usize| {
            Vec3::new(
                peeled[3 * i] as f32,
                peeled[3 * i + 1] as f32,
                peeled[3 * i + 2] as f32,
            ) / 255.
        };
        let through = crate::translucency::SHOW_THROUGH;
        let reach = crate::translucency::raster::length_px(width, 2.0) as isize;
        let blend: Vec<bool> = (0..width * height)
            .map(|i| {
                if core[i] || !touched[i] {
                    return false;
                }
                let (x, y) = ((i % width) as isize, (i / width) as isize);
                let at = |xx: isize, yy: isize| {
                    (xx >= 0 && yy >= 0 && (xx as usize) < width && (yy as usize) < height)
                        .then(|| yy as usize * width + xx as usize)
                };
                let p = colour(i);
                // the hole's colour beside the pixel
                let Some(a) = (-1..=1)
                    .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
                    .filter_map(|(dx, dy)| at(x + dx, y + dy))
                    .find(|&j| core[j])
                    .map(colour)
                else {
                    return false;
                };
                // the beneath seen clear within the reach: the pixel is a mix of the two
                (-reach..=reach).any(|dy| {
                    (-reach..=reach).any(|dx| {
                        at(x + dx, y + dy).is_some_and(|j| {
                            if touched[j] || core[j] {
                                return false;
                            }
                            let b = colour(j);
                            let ab = b - a;
                            let t = ((p - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0., 1.);
                            (p - b).abs().max_element() > through
                                && (p - (a + t * ab)).abs().max_element() <= through
                        })
                    })
                })
            })
            .collect();
        (std::sync::Arc::new(core), blend)
    });
    let completed = match &holes {
        Some((h, blend)) => Some(
            timings
                .time("completion", async || {
                    completed_beneath(peeled, h, blend, width, height, detail).await
                })
                .await,
        ),
        None => None,
    };
    let holes = holes.map(|(h, blend)| {
        std::sync::Arc::new(
            h.iter()
                .zip(&blend)
                .map(|(&a, &b)| a || b)
                .collect::<Vec<bool>>(),
        )
    });
    let image = completed.as_deref().unwrap_or(peeled);
    let (mut inner_ctx, inner_hierarchy) =
        crate::hierarchy::analyze(image, width, height, detail, timings).await;
    let frame = Window::full(width, height);
    let edge = peeled_edge(&candidate.mask, width, height, above);
    let peel = Some((candidate.mask.as_slice(), candidate.k));
    crate::uncertainty::inherit(&mut inner_ctx, ctx, frame, peel, Some(&edge));
    inner_ctx.free = holes.clone();
    let with = trace_level(
        inner_ctx,
        inner_hierarchy,
        detail,
        options,
        Some(&edge),
        levels,
        timings,
        show,
    )
    .await;
    // the beneath's and the parts above's scenes in the peel directory, as SVGs
    #[cfg(feature = "diagnostics")]
    if let (Some(dir), Some(top)) = (crate::diagnostics::peel_dir(), &on_top) {
        let level = std::fs::read_dir(&dir)
            .map(|d| d.filter(|e| e.is_ok()).count())
            .unwrap_or(0);
        let svg = |s: &Scene| crate::emit::svg::encode(&s.model(), &[]);
        let _ = std::fs::write(dir.join(format!("beneath-{level}.svg")), svg(&with));
        let _ = std::fs::write(dir.join(format!("above-{level}.svg")), svg(top));
    }
    (with, on_top, holes)
}

/// The peeled image with its holes filled by completion over the pieces beneath (the leaves
/// of its own hierarchy), see [`crate::translucency::completion::complete_holes`].
async fn completed_beneath(
    peeled: &[u8],
    holes: &[bool],
    blend: &[bool],
    width: usize,
    height: usize,
    detail: f64,
) -> Vec<u8> {
    use crate::translucency::raster::DVec3;
    let mut timings = Timings::default();
    let (_, hierarchy) =
        crate::hierarchy::analyze(peeled, width, height, detail, &mut timings).await;
    let img: Vec<DVec3> = peeled
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| DVec3::new(c[0] as f64, c[1] as f64, c[2] as f64) / 255.)
        .collect();
    let (filled, _, report) = crate::translucency::completion::complete_holes_with_blend(
        &img,
        hierarchy.labels(),
        holes,
        blend,
        width,
        height,
    );
    level_log(format_args!(
        "completion: {} holes, {} ends, {} pairs joined, {} px filled ({} flat)",
        report.holes, report.ends, report.pairs, report.filled_px, report.flat_px
    ));
    let rgb: Vec<u8> = filled
        .iter()
        .flat_map(|c| [c.x, c.y, c.z].map(|v| (v.clamp(0., 1.) * 255.).round() as u8))
        .collect();
    // next to the peeled images in the peel directory, as `completed-N.ppm`
    #[cfg(feature = "diagnostics")]
    if let Some(dir) = crate::diagnostics::peel_dir() {
        let level = std::fs::read_dir(&dir)
            .map(|d| d.filter(|e| e.is_ok()).count())
            .unwrap_or(0);
        let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
        ppm.extend_from_slice(&rgb);
        let _ = std::fs::write(dir.join(format!("completed-{level}.ppm")), ppm);
        // and the holes, white on black, as `holes-N.pgm`
        let mut pgm = format!("P5\n{width} {height}\n255\n").into_bytes();
        pgm.extend(holes.iter().map(|&h| if h { 255u8 } else { 0 }));
        let _ = std::fs::write(dir.join(format!("holes-{level}.pgm")), pgm);
    }
    rgb
}

/// What the holes hold, as a layer of its own: the shapes of the level's opaque trace whose
/// covered and shown pixels lie mostly in the holes, lifted with their order and fills, so they
/// cost no second trace.
fn parts_above(opaque: &Scene, holes: &[bool], region: &[bool]) -> Scene {
    let (width, height) = (opaque.width, opaque.height);
    let pixels = |s| covered_pixels(s, width, height);
    // Whether most of the pixels a shape covers lie in `mask`
    let mostly = |s, mask: &[bool]| {
        let (inside, all) = pixels(s).fold((0usize, 0usize), |(inside, all), i| {
            (inside + usize::from(mask[i]), all + 1)
        });
        2 * inside > all
    };
    // By what it covers as well as what it shows, so a ground under everything, visible
    // mostly in the holes, is not lifted
    let mut keep: Vec<bool> = opaque
        .shapes
        .iter()
        .zip(mostly_in(opaque, holes))
        .map(|(s, shown)| mostly(s, holes) && shown)
        .collect();
    // The picture's order is kept: a shape drawn over a lifted one where they overlap is lifted
    // with it, else the lifted shape would now cover it
    let mut lifted = vec![false; width * height];
    for (s, k) in opaque.shapes.iter().zip(keep.iter_mut()) {
        let over = pixels(s).any(|i| lifted[i]);
        // except a shape lying mostly in the layer's region, which is what the layer and the
        // beneath draw there and lifted would hide the layer
        *k = *k || (over && !mostly(s, region));
        if *k {
            for i in pixels(s) {
                lifted[i] = true;
            }
        }
    }
    // A shape clipped to one that stays beneath is not lifted: unclipped it would spill past
    // its clip over the beneath. Nor is a shape under one that stays beneath where they
    // overlap: lifted, it would cover what the picture draws over it
    loop {
        let mut changed = false;
        let mut stays = vec![false; width * height];
        for (s, k) in opaque.shapes.iter().zip(keep.iter_mut()).rev() {
            if *k {
                if pixels(s).any(|i| stays[i]) {
                    *k = false;
                    changed = true;
                } else {
                    continue;
                }
            }
            for i in pixels(s) {
                stays[i] = true;
            }
        }
        let kept: crate::HashSet<usize> = opaque
            .shapes
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(s, _)| s.id)
            .collect();
        for (s, k) in opaque.shapes.iter().zip(keep.iter_mut()) {
            if *k && s.clip.is_some_and(|c| !kept.contains(&c)) {
                *k = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    keep_only(opaque, &keep)
}

/// The pixels of a `width` x `height` image that `shape` covers by more than half.
fn covered_pixels(shape: &Shape, width: usize, height: usize) -> impl Iterator<Item = usize> + '_ {
    let r = shape.raster();
    (0..r.h.min(height.saturating_sub(r.y0))).flat_map(move |y| {
        (0..r.w.min(width.saturating_sub(r.x0)))
            .filter(move |&x| r.values[y * r.w + x] > 0.5)
            .map(move |x| (y + r.y0) * width + x + r.x0)
    })
}

/// Per shape of `scene`, whether most of the pixels it shows (its coverage over a half, where
/// no shape above covers it so) lie in `mask`.
fn mostly_in(scene: &Scene, mask: &[bool]) -> Vec<bool> {
    let (w, h) = (scene.width, scene.height);
    let mut owner = vec![usize::MAX; w * h];
    for (k, s) in scene.shapes.iter().enumerate() {
        let r = s.raster();
        let win = Window {
            x: r.x0,
            y: r.y0,
            width: r.w.min(w.saturating_sub(r.x0)),
            height: r.h.min(h.saturating_sub(r.y0)),
        };
        if win.width == 0 || win.height == 0 {
            continue;
        }
        let clip = scene.clip_mask(win, k);
        for y in 0..win.height {
            for x in 0..win.width {
                let a = r.values[y * r.w + x] * clip.as_ref().map_or(1., |c| c[y * win.width + x]);
                if a > 0.5 && s.fill.opacity > 0.5 {
                    owner[(y + r.y0) * w + x + r.x0] = k;
                }
            }
        }
    }
    let mut on = vec![(0usize, 0usize); scene.shapes.len()];
    for (i, &k) in owner.iter().enumerate() {
        if k != usize::MAX {
            on[k].1 += 1;
            on[k].0 += usize::from(mask[i]);
        }
    }
    on.iter().map(|&(a, n)| n > 0 && 2 * a > n).collect()
}

/// The scene with only the shapes `keep` marks, those clipped to a shape left out drawn
/// unclipped.
fn keep_only(scene: &Scene, keep: &[bool]) -> Scene {
    let mut out = scene.clone();
    let mut k = 0;
    out.shapes.retain(|_| {
        k += 1;
        keep[k - 1]
    });
    let ids: crate::HashSet<usize> = out.shapes.iter().map(|s| s.id).collect();
    for s in &mut out.shapes {
        if s.clip.is_some_and(|c| !ids.contains(&c)) {
            s.clip = None;
        }
    }
    out.change();
    out
}

/// Pushes the shapes of `from` on top of `scene` in their order, their clips kept among
/// themselves.
fn push_all(scene: &mut Scene, from: Scene) {
    let mut ids: HashMap<usize, usize> = HashMap::default();
    for mut s in from.shapes {
        let old = s.id;
        s.clip = s.clip.and_then(|c| ids.get(&c).copied());
        let new = scene.push(s);
        ids.insert(old, new);
    }
}
