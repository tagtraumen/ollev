//! The object economy, a pass over the finished scene once its pieces are grouped (dashes,
//! dot chains, networks, repeats): an object stays only where it pays its price at the
//! acceptance price, judged on the whole scene's error with the edge term.

use glam::Vec3;

use crate::edges::{EdgeField, EdgeTerm};
use crate::{Context, Scene, Shape, Window};

mod merge;

/// Segments an object costs beyond its cubics per loop, as the bench and usvg count them: the
/// move that opens the loop and the close that ends it.
const LOOP_SEGMENTS: f64 = 2.;
/// Segments an object's paint costs: a colour, three numbers at a segment per two.
const PAINT_SEGMENTS: f64 = 1.5;

/// The whole-scene objective: per pixel its plain squared error, weighted by the edge term where
/// the target has an edge (under the tolerance a missing outline would cost nothing).
pub(crate) struct Judge<'a> {
    ctx: &'a Context,
    edges: Option<(EdgeField, f64)>,
}

impl<'a> Judge<'a> {
    /// The objective of `ctx`'s picture, with the edge term's field where one is given.
    pub(crate) fn new(ctx: &'a Context, edge: Option<EdgeTerm>) -> Self {
        Judge {
            ctx,
            edges: edge.map(|t| {
                (
                    EdgeField::of_target(&ctx.target, ctx.width, ctx.height),
                    t.lambda,
                )
            }),
        }
    }

    fn error(&self, window: Window, rendered: &[Vec3]) -> f64 {
        window
            .pixels(self.ctx.width)
            .zip(rendered)
            .map(|(i, r)| self.pixel(i, *r))
            .sum()
    }

    /// What showing `color` at each pixel `shape` covers instead of its own fill adds to the
    /// error, a quick screen for merging it away.
    fn repainted(
        &self,
        shape: &Shape,
        color: impl Fn(usize) -> Vec3,
        width: usize,
        height: usize,
    ) -> f64 {
        let r = shape.raster();
        let mut added = 0.;
        for y in r.y0..(r.y0 + r.h).min(height) {
            for x in r.x0..(r.x0 + r.w).min(width) {
                let v = r.values[(y - r.y0) * r.w + x - r.x0];
                if v >= 0.5 {
                    let i = y * width + x;
                    let own = shape.fill.at_pixel(x, y);
                    added += self.pixel(i, color(i)) - self.pixel(i, own);
                }
            }
        }
        added
    }

    /// The edge term's weights over `window`, for fitting a shape at them.
    fn weights(&self, window: Window) -> Option<crate::Weights> {
        let (field, lambda) = self.edges.as_ref()?;
        Some(crate::Weights {
            window,
            values: window
                .pixels(self.ctx.width)
                .map(|i| field.weight(i, *lambda))
                .collect(),
            masked_color: None,
        })
    }

    /// The error of pixel `i` showing `r`.
    fn pixel(&self, i: usize, r: Vec3) -> f64 {
        let w = self
            .edges
            .as_ref()
            .map_or(1., |(f, lambda)| f.weight(i, *lambda) as f64);
        w * self.ctx.target[i].distance_squared(r) as f64
    }

    /// The error of the scene's window as drawn now.
    pub(crate) fn drawn(&self, scene: &Scene, window: Window) -> f64 {
        self.error(
            window,
            &scene.render(window, 0..scene.shapes.len(), [0.; 3]),
        )
    }

    /// What `shape` costs at the acceptance price, whether it is drawn at all.
    pub(crate) fn price(&self, shape: &Shape) -> f64 {
        segments(shape) * self.ctx.accept_cost() + self.ctx.accept_path_cost()
    }

    /// What `shape` costs at the detail's price, how what is drawn is described.
    fn describe(&self, shape: &Shape) -> f64 {
        segments(shape) * self.ctx.segment_cost + self.ctx.path_cost
    }
}

/// The segments an object is charged for: its cubics, a move and a close per loop, and its
/// paint, a colour of three numbers at a segment per two (as a transform's are priced).
fn segments(shape: &Shape) -> f64 {
    let loops = if shape.stroke.is_some() {
        1
    } else {
        shape.loops.iter().filter(|l| !l.is_empty()).count()
    };
    shape.cost_segments()
        + LOOP_SEGMENTS * loops as f64
        + PAINT_SEGMENTS
        + crate::fill::cost_segments(&shape.fill)
}

/// An object of the scene: one shape, or a repeat's members drawn as one.
struct Unit {
    ids: Vec<usize>,
    /// Its price at the acceptance price.
    price: f64,
}

/// What [`economize`] did, for the log.
#[derive(Default, Debug)]
pub struct Report {
    pub dropped: usize,
    pub repeats_dropped: usize,
    pub loops_dropped: usize,
    pub merged: usize,
}

/// Drops every object of `scene` whose part of the picture costs less error than its price,
/// the cheapest first, and every loop of a shape (a hole or an island) that does not pay its
/// own. Translucent layers, what lies above or beneath them, and clips stay.
pub async fn economize(
    ctx: &Context,
    scene: &mut Scene,
    edge: Option<EdgeTerm>,
    tolerances: &[f32],
) -> Report {
    let judge = Judge::new(ctx, edge);
    let mut report = Report::default();
    let fixed = fixed(scene);
    let units = units(&judge, scene, &fixed);
    let log = crate::diagnostics::log();
    // Each unit's worth once, the cheapest tried first and judged again as the scene stands
    let mut order: Vec<(f64, usize)> = Vec::new();
    for (u, unit) in units.iter().enumerate() {
        crate::progress::poll().await;
        let margin = worth(&judge, scene, &unit.ids) - unit.price;
        if margin < 0. {
            order.push((margin, u));
        }
    }
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, u) in order {
        crate::progress::poll().await;
        let unit = &units[u];
        let worth = worth(&judge, scene, &unit.ids);
        if worth >= unit.price {
            continue;
        }
        if log {
            eprintln!(
                "economy: drop {:?}, worth {worth:.4} for {:.4}",
                unit.ids, unit.price
            );
        }
        scene.shapes.retain(|s| !unit.ids.contains(&s.id));
        scene.change();
        if unit.ids.len() > 1 || scene.repeats.iter().any(|r| r.members[0].0 == unit.ids[0]) {
            report.repeats_dropped += 1;
            scene.repeats.retain(|r| r.members[0].0 != unit.ids[0]);
        } else {
            report.dropped += 1;
        }
    }
    report.merged = merge::merge_all(&judge, scene, &fixed, tolerances).await;
    report.loops_dropped = drop_loops(&judge, scene, &fixed).await;
    report
}

/// The ids of the shapes the pass leaves alone: the background, the translucent layers and
/// everything drawn from the first of them up (the parts lifted above it), the shapes beneath
/// a layer's shape, and the shapes others are clipped to.
pub(crate) fn fixed(scene: &Scene) -> crate::HashSet<usize> {
    let mut fixed: crate::HashSet<usize> = scene.shapes.iter().filter_map(|s| s.clip).collect();
    fixed.insert(scene.shapes[0].id);
    if let Some(first) = scene.shapes.iter().position(|s| s.layer.is_some()) {
        let width = scene.width;
        for (k, s) in scene.shapes.iter().enumerate() {
            let beneath = scene.shapes[first..]
                .iter()
                .any(|l| l.layer.is_some() && crate::instancing::overlaps(s, l, width));
            if k >= first || beneath {
                fixed.insert(s.id);
            }
        }
    }
    fixed
}

/// The ids of the shapes drawn as members of a repeat.
pub(crate) fn repeated(scene: &Scene) -> crate::HashSet<usize> {
    scene
        .repeats
        .iter()
        .flat_map(|r| r.members.iter().map(|m| m.0))
        .collect()
}

/// The scene's objects the pass may drop: each repeat as one, each other shape alone.
fn units(judge: &Judge, scene: &Scene, fixed: &crate::HashSet<usize>) -> Vec<Unit> {
    let mut members = crate::HashSet::default();
    let mut units = Vec::new();
    for r in &scene.repeats {
        let ids: Vec<usize> = r.members.iter().map(|m| m.0).collect();
        members.extend(ids.iter().copied());
        if ids.iter().any(|id| fixed.contains(id)) {
            continue;
        }
        let loops = r.prototype.loops.len().max(1) as f64;
        let price = (r.segments() + LOOP_SEGMENTS * loops + PAINT_SEGMENTS)
            * judge.ctx.accept_cost()
            + judge.ctx.accept_path_cost();
        units.push(Unit { ids, price });
    }
    for s in &scene.shapes {
        if !fixed.contains(&s.id) && !members.contains(&s.id) {
            units.push(Unit {
                ids: vec![s.id],
                price: judge.price(s),
            });
        }
    }
    units
}

/// The window the shapes `ids` paint.
fn window_of(scene: &Scene, ids: &[usize]) -> Window {
    let (mut min, mut max) = (glam::Vec2::INFINITY, glam::Vec2::NEG_INFINITY);
    for s in scene.shapes.iter().filter(|s| ids.contains(&s.id)) {
        min = min.min(s.paint_min());
        max = max.max(s.paint_max());
    }
    Window::around(min, max, 1., scene.width, scene.height)
}

/// How much error the shapes `ids` save the scene: its error without them less with them.
fn worth(judge: &Judge, scene: &mut Scene, ids: &[usize]) -> f64 {
    let window = window_of(scene, ids);
    if window.is_empty() {
        return 0.;
    }
    let with = judge.drawn(scene, window);
    let places: Vec<usize> = (0..scene.shapes.len())
        .rev()
        .filter(|&k| ids.contains(&scene.shapes[k].id))
        .collect();
    let removed: Vec<(usize, Shape)> = places
        .into_iter()
        .map(|k| (k, scene.shapes.remove(k)))
        .collect();
    scene.change();
    let without = judge.drawn(scene, window);
    for (k, s) in removed.into_iter().rev() {
        scene.shapes.insert(k, s);
    }
    scene.change();
    without - with
}

/// Drops each loop of a many-looped shape (a hole filled, an island gone) whose part of the
/// picture costs less error than its cubics and its move and close; returns how many went.
async fn drop_loops(judge: &Judge<'_>, scene: &mut Scene, fixed: &crate::HashSet<usize>) -> usize {
    let repeated = repeated(scene);
    let mut dropped = 0;
    for k in 1..scene.shapes.len() {
        crate::progress::poll().await;
        let id = scene.shapes[k].id;
        if fixed.contains(&id) || repeated.contains(&id) || scene.shapes[k].stroke.is_some() {
            continue;
        }
        let mut l = 0;
        while scene.shapes[k].loops.len() > 1 && l < scene.shapes[k].loops.len() {
            let shape = scene.shapes[k].clone();
            let window = window_of(scene, &[id]);
            let with = judge.drawn(scene, window);
            let mut without = shape.clone();
            without.loops.remove(l);
            without.primitive = match without.primitive {
                Some((p, _)) if p == l => None,
                Some((p, q)) if p > l => Some((p - 1, q)),
                kept => kept,
            };
            without.update_bounds();
            let saved = judge.price(&shape) - judge.price(&without);
            scene.shapes[k] = without;
            scene.change();
            let after = judge.drawn(scene, window);
            if after - with < saved {
                if crate::diagnostics::log() {
                    eprintln!(
                        "economy: loop {l} of shape {id} dropped, error {with:.4} to {after:.4} for {saved:.4}"
                    );
                }
                dropped += 1;
                continue;
            }
            scene.shapes[k] = shape;
            scene.change();
            l += 1;
        }
    }
    dropped
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::*;
    use crate::scene::tests::square;

    /// A picture of a dark square on white, and its context under the tolerance.
    fn picture(w: usize, a: Vec2, b: Vec2) -> Context {
        let rgb: Vec<u8> = (0..w * w)
            .flat_map(|i| {
                let p = Vec2::new((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                let inside = p.cmpge(a).all() && p.cmplt(b).all();
                if inside { [40u8, 60, 90] } else { [255; 3] }
            })
            .collect();
        let mut ctx = Context::new(&rgb, w, w, 1.);
        crate::uncertainty::apply(&mut ctx, 1.);
        ctx
    }

    /// A shape under a copy of itself draws nothing and goes; so does a loop of no area.
    #[test]
    fn what_draws_nothing_goes() {
        let (a, b) = (Vec2::splat(8.), Vec2::splat(24.));
        let ctx = picture(32, a, b);
        let dark = Vec3::new(40., 60., 90.) / 255.;
        let mut scene = Scene::with_background(32, 32, Vec3::ONE);
        scene.push(Shape::new(square(a, b), dark));
        scene.push(Shape::new(square(a, b), dark));
        let report = crate::progress::finish(economize(&ctx, &mut scene, None, &[]));
        assert_eq!((report.dropped, scene.shapes.len()), (1, 2));
        let mut loops = square(a, b);
        loops.extend(square(Vec2::splat(2.), Vec2::new(2., 6.)));
        scene.shapes[1] = Shape {
            id: scene.shapes[1].id,
            ..Shape::new(loops, dark)
        };
        scene.change();
        let report = crate::progress::finish(economize(&ctx, &mut scene, None, &[]));
        assert_eq!((report.loops_dropped, scene.shapes[1].loops.len()), (1, 1));
    }

    /// Two halves of one square in its colour become one shape.
    #[test]
    fn halves_of_one_fill_merge() {
        let (a, b) = (Vec2::splat(8.), Vec2::splat(24.));
        let ctx = picture(32, a, b);
        let dark = Vec3::new(40., 60., 90.) / 255.;
        let mut scene = Scene::with_background(32, 32, Vec3::ONE);
        scene.push(Shape::new(square(a, Vec2::new(16., 24.)), dark));
        scene.push(Shape::new(square(Vec2::new(16., 8.), b), dark));
        let report = crate::progress::finish(economize(&ctx, &mut scene, None, &[2., 4., 8.]));
        assert_eq!(report.merged, 1);
        assert_eq!(scene.shapes.len(), 2);
    }
}
