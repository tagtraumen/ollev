//! The dual of a lattice: equal rectangular cells separated by equal gaps that show the ground
//! below are also a ground in the cells' colour with strokes in the ground's along the gaps.
//! Both readings are priced as repeats; the cheaper in error plus price is kept.

use crate::tracer::Cubic;
use crate::tracer::fit::eval;
use glam::{Vec2, Vec3};

use super::find::{COLOUR_MERGE, eligible};
use super::structure::lattice;
use super::{ALIGN, Repeat, Structure, TRANSLATION_SEGMENTS};
use crate::refine::MARGIN;
use crate::stroke::{Cap, Join, MAX_WIDTH, Stroke};
use crate::{Context, Fill, Scene, Shape, Window};

/// Most gaps the ground is sampled at.
const GROUND_SAMPLES: usize = 25;

/// A frame of the lattice: `u` along the first step, `v` across it, from `origin`.
#[derive(Clone, Copy)]
struct Frame {
    origin: Vec2,
    u: Vec2,
    v: Vec2,
}

impl Frame {
    fn local(&self, p: Vec2) -> Vec2 {
        let d = p - self.origin;
        Vec2::new(d.dot(self.u), d.dot(self.v))
    }

    fn image(&self, q: Vec2) -> Vec2 {
        self.origin + q.x * self.u + q.y * self.v
    }

    /// The box of the shape's control points in this frame.
    fn extent(&self, shape: &Shape) -> (Vec2, Vec2) {
        let points = shape
            .loops
            .iter()
            .flatten()
            .flatten()
            .map(|p| self.local(*p));
        let min = points.clone().fold(Vec2::INFINITY, Vec2::min);
        let max = points.fold(Vec2::NEG_INFINITY, Vec2::max);
        (min, max)
    }
}

fn line(a: Vec2, b: Vec2) -> Cubic {
    [a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.), b]
}

/// The absolute area the shape's even-odd fill covers, by its flattened loops (holes inside).
fn area(shape: &Shape) -> f32 {
    let mut areas: Vec<f32> = shape
        .loops
        .iter()
        .map(|l| {
            let points: Vec<Vec2> = l
                .iter()
                .flat_map(|c| (0..8).map(move |i| eval(c, i as f32 / 8.)))
                .collect();
            let n = points.len();
            ((0..n)
                .map(|i| points[i].perp_dot(points[(i + 1) % n]))
                .sum::<f32>()
                / 2.)
                .abs()
        })
        .collect();
    areas.sort_by(|a, b| b.total_cmp(a));
    areas.first().copied().unwrap_or(0.) - areas.iter().skip(1).sum::<f32>()
}

/// The lattice's cells, gaps and frame, for a repeat of translated rectangles in rows (and
/// columns) with gaps between them.
struct Cells {
    frame: Frame,
    /// The period along `u` and `v` (zero across a single row).
    period: Vec2,
    /// The cell about its node, and the gap between neighbouring cells, per direction.
    cell: (Vec2, Vec2),
    gap: Vec2,
    nodes: Vec<Vec2>,
    /// The shape ids of the instances on the lattice.
    ids: Vec<usize>,
}

fn cells(repeat: &Repeat) -> Option<Cells> {
    let proto = &repeat.prototype;
    if proto.stroke.is_some() || proto.blur > 0. {
        return None;
    }
    // the lattice of the translated instances; the turned or scaled ones are cells beside it
    let (ids, offsets): (Vec<usize>, Vec<Vec2>) = repeat
        .members
        .iter()
        .filter(|m| m.1.is_translation())
        .map(|m| (m.0, m.1.offset))
        .unzip();
    let l = lattice(&offsets)?;
    let u = l.a.normalize();
    let v = u.perp();
    // a skew moving a cell's far corner by more than an anti-aliasing step is no grid of lines
    if let Some(b) = l.b
        && b.dot(u).abs() > ALIGN
    {
        return None;
    }
    let frame = Frame {
        origin: l.origin,
        u,
        v,
    };
    let local = Frame {
        origin: Vec2::ZERO,
        u,
        v,
    };
    let (min, max) = local.extent(proto);
    let size = max - min;
    // a rectangle: its area the box's, its corners on the outline (a rounded cell is no cell
    // the gap lines leave)
    let points: Vec<Vec2> = proto
        .loops
        .iter()
        .flatten()
        .flat_map(|c| (0..8).map(move |i| local.local(eval(c, i as f32 / 8.))))
        .collect();
    let cornered = [min, max, Vec2::new(min.x, max.y), Vec2::new(max.x, min.y)]
        .iter()
        .all(|c| points.iter().any(|p| p.distance(*c) <= ALIGN));
    if !cornered || area(proto) < (size.x - ALIGN) * (size.y - ALIGN) {
        return None;
    }
    let period = Vec2::new(l.a.length(), l.b.map_or(0., |b| b.dot(v).abs()));
    let gap = Vec2::new(
        period.x - size.x,
        if l.b.is_some() { period.y - size.y } else { 0. },
    );
    // gaps are lines between the cells: a stroke's width and thinner than the cells they part
    // (gaps as wide as the cells make the cells as much the strokes)
    let thin = |g: f32, cell: f32| g > 0. && g <= MAX_WIDTH && g < cell;
    if !thin(gap.x, size.x) || (l.b.is_some() && !thin(gap.y, size.y)) {
        return None;
    }
    // the steps' sign in the frame, so each node is the frame's own coordinates
    let nodes = l.inliers.iter().map(|&i| frame.local(offsets[i])).collect();
    let ids = l.inliers.iter().map(|&i| ids[i]).collect();
    Some(Cells {
        frame,
        period,
        cell: (min, max),
        gap,
        nodes,
        ids,
    })
}

impl Cells {
    /// The centres of the gaps along axis `axis` (0 for `u`) lying strictly between `from`
    /// and `to`, in the frame.
    fn gaps(&self, axis: usize, from: f32, to: f32) -> Vec<f32> {
        let (p, g) = (self.period[axis], self.gap[axis]);
        if p <= 0. {
            return Vec::new();
        }
        let hi = self.cell.1[axis];
        // the gap after node 0's cell, then every period
        let first = hi + g / 2.;
        let base = self
            .nodes
            .iter()
            .map(|n| n[axis])
            .fold(f32::INFINITY, f32::min);
        let k0 = ((from - base - first) / p).floor() as i32 - 1;
        let k1 = ((to - base - first) / p).ceil() as i32 + 1;
        (k0..=k1)
            .map(|k| base + first + k as f32 * p)
            .filter(|c| c - g / 2. >= from + ALIGN && c + g / 2. <= to - ALIGN)
            .collect()
    }
}

/// Whether the two shapes both cover more than half of some pixel: one draws over the other
/// beyond the anti-aliasing at their meeting edges.
fn draws_over(a: &Shape, b: &Shape, width: usize) -> bool {
    if a.min.x >= b.max.x || b.min.x >= a.max.x || a.min.y >= b.max.y || b.min.y >= a.max.y {
        return false;
    }
    let (ra, rb) = (a.raster(), b.raster());
    ra.pixels(width)
        .any(|(i, v)| v > 0.5 && rb.at(i, width) > 0.5)
}

/// The strokes along one direction's gaps: centres at `first + k * step` for each `k` of
/// `index`, across the whole extent, in one width and colour.
struct Pen {
    frame: Frame,
    extent: (Vec2, Vec2),
    axis: usize,
    first: f32,
    step: f32,
    count: usize,
    index: Vec<i32>,
    width: f32,
    colour: Vec3,
}

/// Rounds of halving the pen's steps when fitting it, and most repeats of a move per round.
const PEN_ROUNDS: usize = 5;
const PEN_REPEATS: usize = 4;

impl Pen {
    fn lines(&self) -> Vec<Shape> {
        self.index
            .iter()
            .map(|&k| {
                let c = self.first + k as f32 * self.step;
                let (mut from, mut to) = self.extent;
                from[self.axis] = c;
                to[self.axis] = c;
                let mut s = Stroke {
                    centerline: vec![line(self.frame.image(from), self.frame.image(to))],
                    closed: false,
                    width: self.width,
                    cap: Cap::Butt,
                    join: Join::Miter,
                    corners: vec![false],
                    dashes: None,
                }
                .shape(Fill::solid(self.colour));
                s.antialiased = true;
                s.smooth = true;
                s
            })
            .collect()
    }

    /// The error of the lines over `below` and under `above` in `window`, with the colour that
    /// draws them best there, least squares.
    fn error(
        &mut self,
        ctx: &Context,
        window: Window,
        below: &[Vec3],
        above: Option<&crate::scene::Layer>,
    ) -> f64 {
        let origin = window.origin();
        let scene = crate::render::Scene {
            width: window.width as u32,
            height: window.height as u32,
            paths: self
                .lines()
                .iter()
                .map(|s| {
                    let mut p = s.path(origin);
                    p.color = [1.; 4];
                    p
                })
                .collect(),
        };
        let cover = crate::render::render_over(&scene, vec![Vec3::ZERO; below.len()]);
        // shown = base + t * a * pen, with what lies above as colour plus transmittance
        let (mut num, mut den) = (Vec3::ZERO, Vec3::ZERO);
        let parts: Vec<(Vec3, Vec3)> = (0..below.len())
            .map(|k| {
                let a = cover[k].x;
                let (c, t) = above.map_or((Vec3::ZERO, Vec3::ONE), |l| {
                    (l.color[k], l.transmittance[k])
                });
                (c + t * (1. - a) * below[k], t * a)
            })
            .collect();
        for (k, i) in window.pixels(ctx.width).enumerate() {
            let (base, ta) = parts[k];
            num += ta * (ctx.target[i] - base);
            den += ta * ta;
        }
        self.colour = (num / den.max(Vec3::splat(1e-9))).clamp(Vec3::ZERO, Vec3::ONE);
        window
            .pixels(ctx.width)
            .zip(&parts)
            .map(|(i, (base, ta))| {
                (*base + *ta * self.colour).distance_squared(ctx.target[i]) as f64
            })
            .sum()
    }

    /// Moves the first line, the last and the width where they draw the image best, each move
    /// repeated while it pays, by halving steps from half an anti-aliasing step. A line within
    /// two pixels keeps the gap's width: there a thinner and darker line draws the same pixels.
    fn fit(
        &mut self,
        ctx: &Context,
        window: Window,
        below: &[Vec3],
        above: Option<&crate::scene::Layer>,
    ) {
        let mut best = self.error(ctx, window, below, above);
        let mut kept = (self.first, self.step, self.width, self.colour);
        let span = self.index.last().copied().unwrap_or(0).max(1) as f32;
        let wide = self.width >= 2. * ALIGN;
        let mut h = ALIGN / 2.;
        for _ in 0..PEN_ROUNDS {
            let mut moves = vec![
                (h, 0., 0.),
                (-h, 0., 0.),
                (0., h / span, 0.),
                (0., -h / span, 0.),
            ];
            if wide {
                moves.extend([(0., 0., h), (0., 0., -h)]);
            }
            for (df, ds, dw) in moves {
                for _ in 0..PEN_REPEATS {
                    (self.first, self.step, self.width) = (kept.0 + df, kept.1 + ds, kept.2 + dw);
                    let e = self.error(ctx, window, below, above);
                    if e >= best {
                        break;
                    }
                    best = e;
                    kept = (self.first, self.step, self.width, self.colour);
                }
            }
            h /= 2.;
        }
        (self.first, self.step, self.width, self.colour) = kept;
    }
}

/// Tries the dual of each cluster of shapes alike that is a lattice of cells on a ground,
/// keeping it where it costs less.
pub(super) fn apply(ctx: &Context, scene: &mut Scene, repeats: &[Repeat]) {
    // the largest lattice first, which a row of its edge cells must not stand in for
    let mut order: Vec<&Repeat> = repeats.iter().collect();
    order.sort_by_key(|r| std::cmp::Reverse(r.members.len()));
    for repeat in order {
        if repeat
            .members
            .iter()
            .all(|m| scene.shapes.iter().any(|s| s.id == m.0))
            && let Some(dual) = dual_of(ctx, scene, repeat)
        {
            *scene = dual;
        }
    }
}

/// The scene with the repeat's lattice read as a ground and gap strokes, if that costs less.
fn dual_of(ctx: &Context, scene: &Scene, repeat: &Repeat) -> Option<Scene> {
    let cells = cells(repeat)?;
    let frame = cells.frame;
    let colour = repeat.prototype.fill.color;
    let ids: crate::HashSet<usize> = cells.ids.iter().copied().collect();
    let flat = |s: &Shape| {
        eligible(s)
            && s.stroke.is_none()
            && s.fill.opacity == 1.
            && s.blur == 0.
            && (s.fill.color - colour).abs().max_element() <= COLOUR_MERGE
    };
    // The cells' extent, grown by the shapes in their colour beside it that lie between gaps;
    // the ground is found under the cells' own
    let mut extent = cells
        .ids
        .iter()
        .map(|&id| frame.extent(scene.get(id)))
        .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |a, b| {
            (a.0.min(b.0), a.1.max(b.1))
        });
    let width = scene.width;
    // The ground: the topmost shape below the cells covering the gaps between neighbouring
    // cells, at the gaps' centres
    let lowest = cells.ids.iter().map(|&id| scene.position(id)).min()?;
    let mut gaps: Vec<Vec2> = Vec::new();
    for (k, a) in cells.nodes.iter().enumerate() {
        for b in &cells.nodes[k + 1..] {
            let d = *b - *a;
            let along = |axis: usize| {
                cells.period[axis] > 0.
                    && (d[axis].abs() - cells.period[axis]).abs() <= ALIGN
                    && d[1 - axis].abs() <= ALIGN
            };
            if along(0) || along(1) {
                gaps.push(frame.image((*a + *b) / 2. + (cells.cell.0 + cells.cell.1) / 2.));
            }
        }
    }
    let every = gaps.len().div_ceil(GROUND_SAMPLES).max(1);
    let samples: Vec<usize> = gaps
        .iter()
        .step_by(every)
        .filter(|p| p.x >= 0. && p.y >= 0.)
        .map(|p| (p.y as usize).min(scene.height - 1) * width + (p.x as usize).min(width - 1))
        .collect();
    if samples.is_empty() {
        return None;
    }
    let ground = (0..lowest).rev().find(|&q| {
        let r = scene.shapes[q].raster();
        scene.shapes[q].clip.is_none() && samples.iter().all(|&i| r.at(i, width) > 0.5)
    })?;
    let ground_colour = scene.shapes[ground].fill.at(gaps[0]);
    let shows_ground = |a: Vec2, b: Vec2| {
        let n = a.distance(b).ceil().max(1.) as usize;
        let near = (0..n)
            .filter(|&k| {
                let p = a.lerp(b, (k as f32 + 0.5) / n as f32);
                let (x, y) = (p.x as usize, p.y as usize);
                if x >= width || y >= scene.height {
                    return false;
                }
                let t = ctx.target[y * width + x];
                t.distance(ground_colour) < t.distance(colour)
            })
            .count();
        2 * near >= n
    };
    // A shape of the cells' colour crosses a gap where the image shows the ground along it
    let straddles = |min: Vec2, max: Vec2| {
        (0..2).any(|axis| {
            cells.gaps(axis, min[axis], max[axis]).into_iter().any(|c| {
                let (mut from, mut to) = (min, max);
                from[axis] = c;
                to[axis] = c;
                shows_ground(frame.image(from), frame.image(to))
            })
        })
    };
    let reach = cells.gap.max_element() + ALIGN;
    let mut grown = true;
    while grown {
        grown = false;
        for s in scene
            .shapes
            .iter()
            .filter(|s| !ids.contains(&s.id) && flat(s))
        {
            let (min, max) = frame.extent(s);
            let mut touches =
                min.cmple(extent.1 + reach).all() && max.cmpge(extent.0 - reach).all();
            // a row grows along itself only, its gaps say nothing across it
            if cells.period.y <= 0. {
                touches &= min.y >= extent.0.y - ALIGN && max.y <= extent.1.y + ALIGN;
            }
            let inside = min.cmpge(extent.0).all() && max.cmple(extent.1).all();
            if touches && !inside && !straddles(min, max) {
                extent = (extent.0.min(min), extent.1.max(max));
                grown = true;
            }
        }
    }
    let corners = [
        extent.0,
        Vec2::new(extent.1.x, extent.0.y),
        extent.1,
        Vec2::new(extent.0.x, extent.1.y),
    ]
    .map(|q| frame.image(q));
    let (bmin, bmax) = (
        corners.iter().copied().fold(Vec2::INFINITY, Vec2::min),
        corners.iter().copied().fold(Vec2::NEG_INFINITY, Vec2::max),
    );
    // What goes: the cells and the shapes of their colour between gaps inside the extent that
    // nothing kept above the ground overlaps; what stays may lie over them. By place in the
    // stack, since ids repeat where a clip names one
    let mut kept: Vec<&Shape> = Vec::new();
    let mut gone: Vec<usize> = Vec::new();
    let mut price = 0.;
    for (q, s) in scene.shapes.iter().enumerate().skip(ground + 1) {
        if !s.max.cmpgt(bmin).all() || !s.min.cmplt(bmax).all() {
            continue;
        }
        let (min, max) = frame.extent(s);
        let candidate = ids.contains(&s.id)
            || (flat(s)
                && min.cmpge(extent.0 - ALIGN).all()
                && max.cmple(extent.1 + ALIGN).all()
                && !straddles(min, max));
        if candidate && !kept.iter().any(|k| draws_over(k, s, width)) {
            gone.push(q);
            price += s.cost_segments();
        } else {
            kept.push(s);
        }
    }
    let before_price = price;
    // The ground in the cells' colour and the strokes along the gaps showing the ground
    let mut rect = Shape::new(
        vec![
            (0..4)
                .map(|k| line(corners[k], corners[(k + 1) % 4]))
                .collect(),
        ],
        Fill::solid(colour),
    );
    rect.antialiased = true;
    // Per direction the gap lines' centres, kept where cells that go meet across them and the
    // image shows the ground along them
    let parted: Vec<(Vec2, Vec2)> = gone
        .iter()
        .map(|&q| frame.extent(&scene.shapes[q]))
        .collect();
    let mut directions: Vec<Pen> = Vec::new();
    for axis in 0..2 {
        let across = 1 - axis;
        let (g, near) = (cells.gap[axis], cells.gap[axis] / 2. + ALIGN);
        let meet = |c: f32| {
            parted.iter().any(|a| {
                (a.1[axis] - (c - g / 2.)).abs() <= near
                    && parted.iter().any(|b| {
                        (b.0[axis] - (c + g / 2.)).abs() <= near
                            && a.0[across] < b.1[across]
                            && b.0[across] < a.1[across]
                    })
            })
        };
        let centres: Vec<f32> = cells
            .gaps(axis, extent.0[axis], extent.1[axis])
            .into_iter()
            .filter(|&c| {
                let (mut from, mut to) = extent;
                from[axis] = c;
                to[axis] = c;
                meet(c) && shows_ground(frame.image(from), frame.image(to))
            })
            .collect();
        if let Some(&first) = centres.first() {
            directions.push(Pen {
                frame,
                extent,
                axis,
                first,
                step: cells.period[axis],
                count: centres.len(),
                index: centres
                    .iter()
                    .map(|c| ((c - first) / cells.period[axis]).round() as i32)
                    .collect(),
                width: cells.gap[axis],
                colour: ground_colour,
            });
        }
    }
    if directions.is_empty() {
        return None;
    }
    let after_price = rect.cost_segments()
        + directions
            .iter()
            .map(|d| {
                let rest = d.count - 1;
                d.lines()[0].cost_segments()
                    + TRANSLATION_SEGMENTS
                    + if rest >= 2 {
                        Structure::Linear { step: Vec2::ZERO }.segments(&[])
                    } else {
                        rest as f64 * TRANSLATION_SEGMENTS
                    }
            })
            .sum::<f64>();
    let mut dual = scene.clone();
    let gone_at: crate::HashSet<usize> = gone.iter().copied().collect();
    dual.shapes = std::mem::take(&mut dual.shapes)
        .into_iter()
        .enumerate()
        .filter(|(q, _)| !gone_at.contains(q))
        .map(|(_, s)| s)
        .collect();
    dual.change();
    let window = Window::around(bmin, bmax, MARGIN + reach, ctx.width, ctx.height);
    dual.insert(ground + 1, rect);
    // Each direction's lines fitted to the image over what lies below and under what stays
    // above, the other direction's lines drawn
    let mut slots = Vec::new();
    let mut at = ground + 2;
    for pen in &directions {
        slots.push(at);
        for s in pen.lines() {
            dual.insert(at, s);
            at += 1;
        }
    }
    for (pen, &slot) in directions.iter_mut().zip(&slots) {
        dual.shapes.drain(slot..slot + pen.count);
        dual.change();
        let below = dual.render(window, 0..slot, [1.; 3]);
        let above = dual.layer(window, slot);
        pen.fit(ctx, window, &below, above.as_ref());
        for (k, s) in pen.lines().into_iter().enumerate() {
            dual.insert(slot + k, s);
        }
    }
    let error = |s: &Scene| ctx.error(window, &s.render(window, 0..s.shapes.len(), [1.; 3])) as f64;
    let before = error(scene) + before_price * ctx.segment_cost;
    let after = error(&dual) + after_price * ctx.segment_cost;
    if crate::diagnostics::log() {
        eprintln!(
            "instancing: lattice of {} cells in {bmin}..{bmax} read as {} strokes, {} shapes gone: {after:.0} against {before:.0}",
            repeat.members.len(),
            dual.shapes.len() + gone.len() - scene.shapes.len() - 1,
            gone.len()
        );
    }
    (after < before).then_some(dual)
}
