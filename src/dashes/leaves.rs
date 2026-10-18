//! Dashes and dots the scene leaves undrawn: leaves of the segmentation shaped as a dash or a
//! dot that the scene shows in another colour (potholders' stitches as holes to the ground,
//! origami's folds as the ground between the cells, a pond's dotted rim as a dark band). They
//! are chained as the scene's strokes are and drawn on top, the shapes beneath refitted free
//! under the stroke's paint, and kept where the whole objective at the acceptance price falls.

use glam::Vec3;

use crate::economy::Judge;
use crate::edges::EdgeTerm;
use crate::hierarchy::Hierarchy;

use super::*;

/// How far a leaf's pixel count may stray from the band its second moments give for it to be
/// a dash or a dot: a rectangle fills the band, a disc 1.05 of it; a crescent or a corner less.
const FILLED: (f32, f32) = (0.8, 1.25);

/// A leaf shaped as a dash or a dot: its centre, main axis, length and width (from its second
/// moments) and the image's colour over it.
struct Leaf {
    center: Vec2,
    axis: Vec2,
    length: f32,
    width: f32,
    pixels: f32,
    color: Vec3,
}

/// The dash or dot a leaf's pixel statistics describe, if they describe one: a band of at most
/// a stroke's width and at least two anti-aliasing steps across and along (narrower, a blend of
/// an edge, as niabot's hair fragments), that its pixels fill.
fn shaped(stats: &crate::hierarchy::Stats) -> Option<(Vec2, Vec2, f32, f32)> {
    let n = stats.n;
    if n < 2. {
        return None;
    }
    let c = stats.centroid();
    let p = stats.position_products;
    let (xx, xy, yy) = (
        p[0] / n - c.x * c.x,
        p[1] / n - c.x * c.y,
        p[2] / n - c.y * c.y,
    );
    let half = (xx + yy) / 2.;
    let root = (((xx - yy) / 2.).powi(2) + xy * xy).sqrt();
    let (l1, l2) = (half + root, (half - root).max(0.));
    // a run of k pixels has the variance (k^2 - 1) / 12
    let (length, width) = ((12. * l1 + 1.).sqrt() as f32, (12. * l2 + 1.).sqrt() as f32);
    let fill = n as f32 / (length * width);
    let least = 2. * crate::edges::SIGMA as f32;
    // (a 2 px band reads 2 to rounding, so the bound gives a thousandth)
    if width > MAX_WIDTH || width < least - 1e-3 || fill < FILLED.0 || fill > FILLED.1 {
        return None;
    }
    let axis = if xy.abs() > 1e-9 {
        glam::DVec2::new(xy, l1 - xx).normalize()
    } else if xx >= yy {
        glam::DVec2::X
    } else {
        glam::DVec2::Y
    };
    Some((c.as_vec2(), axis.as_vec2(), length, width))
}

/// The leaves shaped as dashes or dots whose pixels the scene draws in a colour other than
/// their own, by more than two of a paint's leaves may differ.
fn undrawn(ctx: &Context, hierarchy: &Hierarchy, scene: &Scene) -> Vec<Leaf> {
    let shaped: Vec<(u32, (Vec2, Vec2, f32, f32))> = hierarchy
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.children.is_none())
        .filter_map(|(i, n)| Some((i as u32, shaped(&n.stats)?)))
        .collect();
    if shaped.len() < MIN_DASHES {
        return Vec::new();
    }
    let full = Window::full(ctx.width, ctx.height);
    let drawn = scene.render(full, 0..scene.shapes.len(), [1.; 3]);
    // A stroke's pixels are drawn by it, if a little off (galewarning's dashed rings, whose
    // rim leaves differ by a few levels): the leaves are for what the fills left
    let mut stroked = vec![false; ctx.width * ctx.height];
    for s in scene.shapes.iter().filter(|s| s.stroke.is_some()) {
        for (i, v) in s.raster().pixels(ctx.width) {
            if v >= 0.5 && i < stroked.len() {
                stroked[i] = true;
            }
        }
    }
    shaped
        .into_iter()
        .filter_map(|(node, (center, axis, length, width))| {
            let pixels = hierarchy.pixels(node);
            // no evidence where a pixel may show anything (the rim a peel invents, single-09's
            // beneath) or nothing says what it shows
            let open = |i: usize| {
                ctx.tolerance_at(i) >= crate::uncertainty::ANYTHING
                    || ctx.free.as_ref().is_some_and(|f| f[i])
            };
            if pixels.iter().any(|&i| open(i as usize)) {
                return None;
            }
            if 2 * pixels.iter().filter(|&&i| stroked[i as usize]).count() >= pixels.len() {
                return None;
            }
            let k = pixels.len() as f32;
            let (own, shown) = pixels.iter().fold((Vec3::ZERO, Vec3::ZERO), |(a, b), &i| {
                (a + ctx.target[i as usize], b + drawn[i as usize])
            });
            let (color, shown) = (own / k, shown / k);
            (color.distance(shown) > ALIKE_COLOR).then_some(Leaf {
                center,
                axis,
                length,
                width,
                pixels: k,
                color,
            })
        })
        .collect()
}

/// Draws the chains of undrawn dash and dot leaves (see the module) where that lowers the
/// scene's error with the edge term `edge` plus its price at the acceptance price.
pub fn from_leaves(
    ctx: &Context,
    hierarchy: &Hierarchy,
    scene: &mut Scene,
    tolerances: &[f32],
    edge: Option<EdgeTerm>,
) {
    let leaves = undrawn(ctx, hierarchy, scene);
    if leaves.len() < MIN_DASHES {
        return;
    }
    let alike =
        |width: f32, extent: f32| extent - width <= (WIDTH_SLACK * extent).max(WIDTH_QUANTUM);
    let mut dashes = Vec::new();
    let mut dots = Vec::new();
    for (k, leaf) in leaves.iter().enumerate() {
        if alike(leaf.width, leaf.length) {
            // a disc's area gives its diameter, the moments only its proportions
            let disc = leaf.length * leaf.width * std::f32::consts::FRAC_PI_4;
            let scale = (leaf.pixels / disc).sqrt();
            dots.push(Dot {
                position: k,
                center: leaf.center,
                width: leaf.width * scale,
                extent: leaf.length * scale,
                color: Some(leaf.color),
            });
        } else {
            let half = leaf.axis * (leaf.length / 2.);
            let (a, b) = (leaf.center - half, leaf.center + half);
            let stroke = Stroke {
                centerline: vec![[a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.), b]],
                closed: false,
                width: leaf.width,
                cap: Cap::Butt,
                join: crate::stroke::Join::Miter,
                corners: vec![false],
                dashes: None,
            };
            dashes.push(Dash::new(k, stroke, Some(leaf.color)));
        }
    }
    let mut planned: Vec<(Vec<usize>, Stroke, &str)> = Vec::new();
    if dashes.len() >= MIN_DASHES {
        for (members, dashed) in dash_chains(&dashes) {
            planned.push((members, dashed, "dashes"));
        }
    }
    for (members, dotted) in chain_dots(&dots) {
        planned.push((members, dotted, "dots"));
    }
    if planned.is_empty() {
        return;
    }
    let judge = Judge::new(ctx, edge);
    let first = scene.shapes.len();
    for (members, dashed, kind) in planned {
        if crate::progress::poll() {
            break;
        }
        let mut colors: Vec<Vec3> = members.iter().map(|&m| leaves[m].color).collect();
        let color = median_color(&mut colors);
        try_leaf_chain(ctx, &judge, scene, dashed, color, kind);
    }
    if scene.shapes.len() > first {
        refit_beneath(ctx, &judge, scene, first, tolerances);
    }
}

/// The median of each channel.
fn median_color(colors: &mut [Vec3]) -> Vec3 {
    let mut channel = |k: usize| {
        colors.sort_by(|a, b| a[k].total_cmp(&b[k]));
        colors[colors.len() / 2][k]
    };
    Vec3::new(channel(0), channel(1), channel(2))
}

/// The chain's stroke drawn on top of `scene` where the error with the edge term plus the
/// price of what changed falls, with each loop beneath it that no longer pays gone; whether it
/// was.
fn try_leaf_chain(
    ctx: &Context,
    judge: &Judge,
    scene: &mut Scene,
    dashed: Stroke,
    color: Vec3,
    kind: &str,
) -> bool {
    let mut proposed = dashed.shape(crate::Fill::solid(color));
    proposed.smooth = true;
    let top = scene.shapes.len();
    // Fitted over the ground it lies on: what the stroke covers goes from the shapes beneath
    // first (a hole to the ground under potholders' stitches is no ground of theirs)
    let mut ground = scene.clone();
    for k in 1..top {
        if let Some(s) = without_covered(&ground.shapes[k], &proposed) {
            ground.shapes[k] = s;
        }
    }
    ground.change();
    let Some(shape) = fitted(ctx, &ground, top, &proposed, dashed, kind) else {
        return false;
    };
    let seen = shape
        .stroke
        .as_ref()
        .is_some_and(|s| gaps_seen(ctx, s, shape.fill.color));
    if !seen {
        return false;
    }
    let mut trial = scene.clone();
    trial.shapes.push(shape.clone());
    trial.change();
    let (changed, min, max) = drop_under(judge, &mut trial, top);
    let window = Window::around(
        min.min(shape.paint_min()),
        max.max(shape.paint_max()),
        MARGIN,
        ctx.width,
        ctx.height,
    );
    if window.is_empty() {
        return false;
    }
    let price = |s: &Scene| -> f64 { changed.iter().map(|&k| judge.price(&s.shapes[k])).sum() };
    let before = judge.drawn(scene, window) + price(scene);
    let after = judge.drawn(&trial, window) + price(&trial) + judge.price(&shape);
    if crate::diagnostics::log() {
        let s = shape.stroke.as_ref();
        eprintln!(
            "dashes: leaf {kind} at {:.0?}..{:.0?}, width {:.2}, {:?}, {} loops beneath: {before:.3} to {after:.3}",
            shape.min,
            shape.max,
            s.map_or(0., |s| s.width),
            s.and_then(|s| s.dashes.as_ref()).map(|d| d.array.len()),
            changed.len()
        );
    }
    if after >= before {
        return false;
    }
    *scene = trial;
    let change = scene.change();
    for k in changed.into_iter().chain([top]) {
        scene.shapes[k].changed = change;
        scene.shapes[k].fitted = change;
    }
    scene.change();
    true
}

/// Drops each loop (a hole or an island) of the shapes below the stroke at `top` that the
/// stroke paints over and that no longer pays its price, as the economy drops loops; the
/// shapes changed and the box of the loops gone.
fn drop_under(judge: &Judge, scene: &mut Scene, top: usize) -> (Vec<usize>, Vec2, Vec2) {
    let stroke = scene.shapes[top].clone();
    let (paint, width, height) = (stroke.raster(), scene.width, scene.height);
    let painted = |l: &[crate::tracer::Cubic], fill| {
        let own = Shape::new(vec![l.to_vec()], fill);
        let r = own.raster();
        let over = (r.y0..r.y0 + r.h).any(|y| {
            (r.x0..r.x0 + r.w).any(|x| {
                r.values[(y - r.y0) * r.w + x - r.x0] >= 0.5
                    && paint.at(y * width + x, width) >= 0.5
            })
        });
        (over, own.min, own.max)
    };
    let (mut changed, mut min, mut max) = (Vec::new(), Vec2::INFINITY, Vec2::NEG_INFINITY);
    for k in 1..top {
        let shape = &scene.shapes[k];
        if shape.loops.len() < 2 || shape.stroke.is_some() || !overlap(shape, &stroke) {
            continue;
        }
        let mut l = 0;
        while scene.shapes[k].loops.len() > 1 && l < scene.shapes[k].loops.len() {
            let shape = scene.shapes[k].clone();
            let (over, lo, hi) = painted(&shape.loops[l], shape.fill);
            if !over {
                l += 1;
                continue;
            }
            let window = Window::around(lo, hi, MARGIN, width, height);
            let with = judge.drawn(scene, window) + judge.price(&shape);
            let mut without = shape.clone();
            without.loops.remove(l);
            without.primitive = match without.primitive {
                Some((p, _)) if p == l => None,
                Some((p, q)) if p > l => Some((p - 1, q)),
                kept => kept,
            };
            without.update_bounds();
            let price = judge.price(&without);
            scene.shapes[k] = without;
            scene.change();
            if judge.drawn(scene, window) + price < with {
                if !changed.contains(&k) {
                    changed.push(k);
                }
                (min, max) = (min.min(lo), max.max(hi));
                continue;
            }
            scene.shapes[k] = shape;
            scene.change();
            l += 1;
        }
    }
    (changed, min, max)
}

/// Each shape below the strokes from `first` up that one of them overlaps, refitted with their
/// paint free above it (as the redraw frees every outline under the shapes drawn later), kept
/// where the error with the edge term plus its price falls.
fn refit_beneath(
    ctx: &Context,
    judge: &Judge,
    scene: &mut Scene,
    first: usize,
    tolerances: &[f32],
) {
    for k in 1..first {
        if crate::progress::poll() {
            break;
        }
        let over = scene.shapes[first..]
            .iter()
            .any(|s| overlap(&scene.shapes[k], s));
        if !over {
            continue;
        }
        let kept = scene.shapes[k].clone();
        if !refit_under(ctx, scene, k, tolerances) {
            continue;
        }
        let new = scene.shapes[k].clone();
        let window = Window::around(
            kept.paint_min().min(new.paint_min()),
            kept.paint_max().max(new.paint_max()),
            MARGIN,
            ctx.width,
            ctx.height,
        );
        let after = judge.drawn(scene, window) + judge.price(&new);
        scene.shapes[k] = kept.clone();
        scene.change();
        let before = judge.drawn(scene, window) + judge.price(&kept);
        if after < before {
            scene.shapes[k] = new;
            scene.change();
        }
    }
}

/// `shape` without its loops (holes or islands) that `stroke` paints over half of, none where it
/// keeps them all or would keep none.
fn without_covered(shape: &Shape, stroke: &Shape) -> Option<Shape> {
    if shape.loops.len() < 2 || shape.stroke.is_some() || !overlap(shape, stroke) {
        return None;
    }
    let paint = stroke.raster();
    let covered: Vec<bool> = shape
        .loops
        .iter()
        .map(|l| {
            let own = Shape::new(vec![l.clone()], shape.fill);
            let r = own.raster();
            let (mut all, mut under) = (0usize, 0usize);
            for y in r.y0..r.y0 + r.h {
                for x in r.x0..r.x0 + r.w {
                    if r.values[(y - r.y0) * r.w + x - r.x0] >= 0.5 {
                        all += 1;
                        let inside = x >= paint.x0
                            && y >= paint.y0
                            && x < paint.x0 + paint.w
                            && y < paint.y0 + paint.h;
                        if inside && paint.values[(y - paint.y0) * paint.w + x - paint.x0] >= 0.5 {
                            under += 1;
                        }
                    }
                }
            }
            all > 0 && 2 * under >= all
        })
        .collect();
    let kept = covered.iter().filter(|&&c| !c).count();
    if kept == shape.loops.len() || kept == 0 {
        return None;
    }
    let mut out = shape.clone();
    let mut index = 0;
    out.loops.retain(|_| {
        index += 1;
        !covered[index - 1]
    });
    out.primitive = shape.primitive.and_then(|(p, q)| {
        if covered[p] {
            None
        } else {
            Some((p - covered[..p].iter().filter(|&&c| c).count(), q))
        }
    });
    out.update_bounds();
    Some(out)
}

/// Refits the shape at `k` in `scene` against what lies above it, its outline then simplified
/// at `tolerances` (a primitive keeps its outline); whether it changed.
fn refit_under(ctx: &Context, scene: &mut Scene, k: usize, tolerances: &[f32]) -> bool {
    let s = &scene.shapes[k];
    if s.primitive.is_some() || s.stroke.is_some() || s.clip.is_some() || s.blur > 0. {
        return false;
    }
    let shape = scene.shapes.remove(k);
    scene.change();
    let mut refitted = shape.clone();
    let changed = match Place::new(ctx, scene, &shape, k, MARGIN) {
        Some(place) => {
            let before = place.error(ctx, &shape);
            place.refine(ctx, &mut refitted);
            let cost = place.cost(ctx, refitted.fill);
            crate::refine::simplify_with(
                ctx,
                &place,
                &cost,
                &mut refitted,
                tolerances,
                crate::outline::CORNER_ANGLE,
            );
            let after = place.error(ctx, &refitted);
            let saved = (shape.cost_segments() - refitted.cost_segments()) * ctx.segment_cost;
            (after - before) as f64 <= saved
        }
        None => false,
    };
    let change = scene.change();
    scene.shapes.insert(
        k,
        if changed {
            Shape {
                changed: change,
                fitted: change,
                ..refitted
            }
        } else {
            shape
        },
    );
    scene.change();
    changed
}

#[cfg(test)]
mod tests {
    /// Potholders' stitches: a row of cream dashes 2.5 px wide off the pixel grid on a red
    /// square over a cream ground, which the trace leaves as holes to the ground, become one
    /// dashed stroke over the square.
    #[test]
    fn stitches_over_a_square_are_one_dashed_stroke() {
        use crate::plan;
        let (w, h) = (96, 64);
        let overlap = |a: f32, b: f32, lo: f32, hi: f32| (b.min(hi) - a.max(lo)).max(0.);
        let mut rgb = vec![0u8; 3 * w * h];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32, y as f32);
                let square = (16..80).contains(&x) && (12..52).contains(&y);
                let row = overlap(fy, fy + 1., 29.75, 32.25);
                let along: f32 = (0..5)
                    .map(|k| overlap(fx, fx + 1., 22. + 11. * k as f32, 28. + 11. * k as f32))
                    .sum();
                let stitch = row * along;
                let under = if square {
                    [192., 81., 59.]
                } else {
                    [232., 220., 200.]
                };
                for c in 0..3 {
                    let v = under[c] + ([240., 224., 208.][c] - under[c]) * stitch;
                    rgb[3 * (y * w + x) + c] = v.round() as u8;
                }
            }
        }
        let detail = crate::tracer::auto_detail(&rgb, w, h);
        let scene = plan::trace(&rgb, w, h, detail, plan::Options::default(), |_| ());
        let dashed = scene
            .shapes
            .iter()
            .filter(|s| s.stroke.as_ref().is_some_and(|t| t.dashes.is_some()))
            .count();
        assert_eq!(dashed, 1, "{} shapes", scene.shapes.len());
    }
}
