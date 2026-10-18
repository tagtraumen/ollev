//! Merging an object into a shape beneath it, retraced over both (a union: fills within the
//! noise, or a sliver) or over its own pixels less the object's (a difference: the object
//! restores what lies beneath), where the scene's error plus price falls.

use glam::{IVec2, Vec3};

use super::{Judge, window_of};
use crate::refine::{MARGIN, Place, simplify_with};
use crate::{Context, Fill, Scene, Shape, Window};

/// Widest a sliver is on average, in pixels: two anti-aliasing steps, so all of it is the
/// mixed rim of its neighbours.
const SLIVER_WIDTH: f64 = 2. * crate::edges::SIGMA;

/// How a shape beneath takes over an object's pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// Fills within the noise.
    Union,
    /// The object restores what lies beneath the lower shape.
    Difference,
    /// The object is a sliver of another colour, which goes.
    Sliver,
}

/// Merges, from the top of the stack down, each object into the shape beneath it where that
/// lowers the scene's error plus price; returns how many merged.
pub(super) async fn merge_all(
    judge: &Judge<'_>,
    scene: &mut Scene,
    fixed: &crate::HashSet<usize>,
    tolerances: &[f32],
) -> usize {
    let repeated = super::repeated(scene);
    let free = |s: &Shape| {
        !fixed.contains(&s.id)
            && s.stroke.is_none()
            && s.clip.is_none()
            && s.layer.is_none()
            && s.blur == 0.
            && s.fill.opacity == 1.
            && !repeated.contains(&s.id)
    };
    let ids: Vec<usize> = scene
        .shapes
        .iter()
        .rev()
        .filter(|s| free(s))
        .map(|s| s.id)
        .collect();
    let candidates: crate::HashSet<usize> = ids.iter().copied().collect();
    let mut merged = 0;
    // A shape whose retrace lost, which a merge seldom wins back at the acceptance price, is
    // not traced again
    let mut lost = crate::HashSet::default();
    for id in ids {
        crate::progress::poll().await;
        let Some(top) = scene.shapes.iter().position(|s| s.id == id) else {
            continue;
        };
        let mut best: Option<(f64, usize, Shape)> = None;
        for below in (1..top).rev() {
            crate::progress::poll().await;
            let b = &scene.shapes[below];
            if !candidates.contains(&b.id)
                || lost.contains(&b.id)
                || !touches(&scene.shapes[top], b, scene.width, scene.height)
            {
                continue;
            }
            let lower = b.id;
            for kind in kinds(judge, scene, top, below) {
                match try_merge(judge, scene, top, below, kind, tolerances) {
                    Some((gain, shape)) if gain > 0. => {
                        if best.as_ref().is_none_or(|(g, _, _)| gain > *g) {
                            best = Some((gain, below, shape));
                        }
                    }
                    _ => {
                        lost.insert(lower);
                    }
                }
            }
        }
        if let Some((gain, below, shape)) = best {
            if crate::diagnostics::log() {
                eprintln!(
                    "economy: shape {id} merged into {}, gain {gain:.5}",
                    scene.shapes[below].id
                );
            }
            let change = scene.change();
            scene.shapes[below] = Shape {
                changed: change,
                fitted: change,
                ..shape
            };
            scene.shapes.remove(top);
            scene.change();
            merged += 1;
        }
    }
    merged
}

/// Whether two shapes cover or border a common pixel.
fn touches(a: &Shape, b: &Shape, width: usize, height: usize) -> bool {
    if a.min.x > b.max.x + 1.
        || b.min.x > a.max.x + 1.
        || a.min.y > b.max.y + 1.
        || b.min.y > a.max.y + 1.
    {
        return false;
    }
    let (ra, rb) = (a.raster(), b.raster());
    in_image(ra, width, height).any(|(i, v)| {
        v >= 0.5 && {
            let (x, y) = ((i % width) as i64, (i / width) as i64);
            [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .any(|(dx, dy)| {
                    let (xx, yy) = (x + dx, y + dy);
                    xx >= 0
                        && yy >= 0
                        && (xx as usize) < width
                        && (yy as usize) < height
                        && rb.at(yy as usize * width + xx as usize, width) >= 0.5
                })
        }
    })
}

/// The merges worth trying of the object at `top` into the shape at `below`: a union where
/// their fills agree within the noise or the object is a sliver; a difference where what lies
/// beneath the lower shape, over the object, agrees with the object's fill.
fn kinds(judge: &Judge, scene: &Scene, top: usize, below: usize) -> Vec<Kind> {
    let ctx = judge.ctx;
    let (o, b) = (&scene.shapes[top], &scene.shapes[below]);
    let width = scene.width;
    // The noise over the object's interior: the median tolerance of the pixels it covers
    // fully, the 8-bit noise at least (the rims' tolerances hold the edges' uncertainty)
    let mut interior: Vec<f32> = in_image(o.raster(), width, scene.height)
        .filter(|&(_, v)| v >= 1.)
        .map(|(i, _)| ctx.tolerance_at(i))
        .collect();
    if interior.is_empty() {
        return Vec::new();
    }
    let middle = interior.len() / 2;
    let noise = (*interior.select_nth_unstable_by(middle, f32::total_cmp).1 as f64)
        .max(crate::translucency::raster::NOISE);
    let agree = |a: Vec3, c: Vec3| (a.distance(c) as f64) <= noise;
    let solid = |f: &Fill| f.gradient.is_none();
    if !solid(&o.fill) {
        return Vec::new();
    }
    // A merge can pay only where showing the lower shape's colour, or what lies beneath it,
    // over the object's pixels costs less error than the object's price
    let mut kinds = Vec::new();
    let union = if agree(o.fill.color, b.fill.color) {
        Kind::Union
    } else {
        Kind::Sliver
    };
    if solid(&b.fill)
        && (union == Kind::Union || sliver(o))
        && judge.repainted(o, |_| b.fill.color, width, scene.height)
            < if union == Kind::Union {
                judge.describe(o)
            } else {
                judge.price(o)
            }
    {
        kinds.push(union);
    }
    // What the pixels of the object inside the lower shape show without the lower shape
    let window = Window::around(o.min, o.max, 0., scene.width, scene.height);
    let beneath = scene.render(window, 0..below, [0.; 3]);
    let rb = b.raster();
    let (mut sum, mut n) = (Vec3::ZERO, 0usize);
    for (k, i) in window.pixels(width).enumerate() {
        if o.raster().at(i, width) >= 0.5 && rb.at(i, width) >= 0.5 {
            sum += beneath[k];
            n += 1;
        }
    }
    let shown = |i: usize| {
        let (x, y) = (i % width, i / width);
        match window.index(i, width) {
            Some(k) if rb.at(i, width) >= 0.5 => beneath[k],
            _ => o.fill.at_pixel(x, y),
        }
    };
    if n > 0
        && agree(sum / n as f32, o.fill.color)
        && judge.repainted(o, shown, width, scene.height) < judge.describe(o)
    {
        kinds.push(Kind::Difference);
    }
    kinds
}

/// The pixels of a raster's box that lie in the image, with their coverage.
fn in_image(
    r: &crate::Raster,
    width: usize,
    height: usize,
) -> impl Iterator<Item = (usize, f32)> + '_ {
    r.pixels(width)
        .enumerate()
        .filter(move |&(k, _)| r.x0 + k % r.w.max(1) < width && r.y0 + k / r.w.max(1) < height)
        .map(|(_, p)| p)
}

/// Whether a shape is a sliver: its mean width, twice its area over its perimeter, within
/// [`SLIVER_WIDTH`].
fn sliver(s: &Shape) -> bool {
    let area: f64 = s.raster().values.iter().map(|&v| v as f64).sum();
    let perimeter: f64 = s
        .loops
        .iter()
        .flatten()
        .map(|c| c[0].distance(c[3]) as f64)
        .sum();
    perimeter > 0. && 2. * area / perimeter <= SLIVER_WIDTH
}

/// The shape at `below` traced again over the pixels `kind` gives it and fitted in its place,
/// with the object at `top` gone, and what that gains in error plus price (a loss below zero).
fn try_merge(
    judge: &Judge,
    scene: &mut Scene,
    top: usize,
    below: usize,
    kind: Kind,
    tolerances: &[f32],
) -> Option<(f64, Shape)> {
    let ctx = judge.ctx;
    let (width, height) = (scene.width, scene.height);
    let (o, b) = (scene.shapes[top].clone(), scene.shapes[below].clone());
    let window = window_of(scene, &[o.id, b.id]);
    let (ro, rb) = (o.raster().clone(), b.raster().clone());
    let inside = |i: usize| match kind {
        Kind::Union | Kind::Sliver => ro.at(i, width) >= 0.5 || rb.at(i, width) >= 0.5,
        Kind::Difference => rb.at(i, width) >= 0.5 && ro.at(i, width) < 0.5,
    };
    let min = IVec2::new(window.x as i32, window.y as i32);
    let max = min + IVec2::new(window.width as i32, window.height as i32);
    let (loops, corners) = crate::outline::outline_corners(
        min,
        max,
        width,
        height,
        &crate::outline::FIT_TURNS,
        inside,
    );
    if loops.is_empty() {
        return None;
    }
    let mut shape = Shape {
        id: b.id,
        node: b.node,
        smooth: b.smooth,
        antialiased: b.antialiased,
        ..Shape::new(loops, b.fill)
    }
    .with_corners(corners);
    // Fitted in the lower shape's place with both out of the scene
    // at the edge term's weights, as the plan fits its shapes
    let reach = MARGIN + tolerances.last().copied().unwrap_or(0.);
    let fit = Context {
        pull: crate::refine::HANDLE_PULL,
        weights: judge.weights(Window::around(
            shape.min,
            shape.max,
            reach + 1.,
            width,
            height,
        )),
        ..ctx.clone()
    };
    scene.shapes.remove(top);
    let lower = scene.shapes.remove(below);
    scene.change();
    let fitted = (|| {
        let near = Place::new(&fit, scene, &shape, below, MARGIN)?;
        near.refine(&fit, &mut shape);
        let place = Place::new(&fit, scene, &shape, below, reach)?;
        if kind != Kind::Difference {
            shape.fill = refit_fill(ctx, &place, &shape);
        }
        let cost = place.cost(&fit, shape.fill);
        let origin = place.window.origin();
        let error = cost.value(&shape.path(origin));
        crate::cleanup::clean_in(&fit, &cost, origin, &mut shape, error);
        simplify_with(
            &fit,
            &place,
            &cost,
            &mut shape,
            tolerances,
            crate::outline::CORNER_ANGLE,
        );
        Some(())
    })();
    scene.shapes.insert(below, lower);
    scene.shapes.insert(top, o);
    scene.change();
    fitted?;
    // Measured over everything the three shapes paint, the merged one in the lower's place
    let (lo, hi) = (
        window.origin(),
        window.origin() + glam::Vec2::new(window.width as f32, window.height as f32),
    );
    let window = Window::around(
        lo.min(shape.paint_min()),
        hi.max(shape.paint_max()),
        1.,
        width,
        height,
    );
    let shown = scene.render(window, 0..scene.shapes.len(), [0.; 3]);
    let error = judge.error(window, &shown);
    let o = scene.shapes.remove(top);
    let lower = std::mem::replace(&mut scene.shapes[below], shape.clone());
    scene.change();
    let merged = scene.render(window, 0..scene.shapes.len(), [0.; 3]);
    scene.shapes[below] = lower;
    scene.shapes.insert(top, o);
    scene.change();
    // A merge that only redescribes what is drawn (within the noise, away from the old
    // outlines' anti-aliasing) is priced at the detail's price, as a simplification is; one
    // that changes the drawing at the acceptance price
    let near = |i: usize| {
        let (x, y) = ((i % width) as i64, (i / width) as i64);
        (-1..=1).any(|dy| {
            (-1..=1).any(|dx| {
                let (xx, yy) = (x + dx, y + dy);
                xx >= 0 && yy >= 0 && (xx as usize) < width && (yy as usize) < height && {
                    let j = yy as usize * width + xx as usize;
                    let (vo, vb) = (ro.at(j, width), rb.at(j, width));
                    (vo > 0. && vo < 1.) || (vb > 0. && vb < 1.)
                }
            })
        })
    };
    let noise = crate::translucency::raster::NOISE as f32;
    let redescribes = kind != Kind::Sliver
        && window
            .pixels(width)
            .zip(shown.iter().zip(&merged))
            .all(|(i, (a, b))| a.distance(*b) <= noise || near(i));
    let price = |s: &Shape| {
        if redescribes {
            judge.describe(s)
        } else {
            judge.price(s)
        }
    };
    let o = &scene.shapes[top];
    let before = error + price(o) + price(&b);
    let after = judge.error(window, &merged) + price(&shape);
    if crate::diagnostics::debug() {
        eprintln!(
            "economy: {kind:?} of {} into {}: {before:.5} to {after:.5}, {} to {} segments",
            scene.shapes[top].id,
            b.id,
            scene.shapes[top].segments() + b.segments(),
            shape.segments()
        );
    }
    Some((before - after, shape))
}

/// The solid colour that best draws the pixels the merged shape shows fully at its place: the
/// mean of the target over them.
fn refit_fill(ctx: &Context, place: &Place, shape: &Shape) -> Fill {
    let coverage = shape.coverage(place.window);
    let (mut sum, mut n) = (Vec3::ZERO, 0usize);
    for (k, i) in place.window.pixels(ctx.width).enumerate() {
        let shown = place
            .above
            .as_ref()
            .is_none_or(|a| a.transmittance[k].min_element() > 0.99);
        if coverage[k] > 0.99 && shown {
            sum += ctx.target[i];
            n += 1;
        }
    }
    if n == 0 {
        return shape.fill;
    }
    Fill::solid(sum / n as f32)
}
