//! Snapping shapes alike to one prototype: each instance becomes the prototype placed by a
//! transform refitted to the scene's picture, kept where it changes that picture by less than
//! the price of the geometry it no longer pays for.

use glam::{Vec2, Vec3};

use super::find::{Outline, align};
use super::{ALIGN, Repeat, Structure, Transform, reach, transformed};
use crate::refine::MARGIN;
use crate::scene::Layer;
use crate::translucency::raster::NOISE;
use crate::{Context, Scene, Shape, Window};

/// Most members the prototype is chosen among.
const MEDOID_SAMPLE: usize = 16;
/// Rounds of halving the refit's steps.
const REFIT_ROUNDS: usize = 3;

/// The shape at `position` and what lies around it, for measuring stand-ins for it against the
/// scene as it is drawn now: instancing re-encodes a decided scene, so a stand-in answers to the
/// scene's render, not to the image.
pub(super) struct Site {
    window: Window,
    below: Vec<Vec3>,
    above: Option<Layer>,
    /// The window as the scene draws it with the shape itself.
    drawn: Vec<Vec3>,
}

impl Site {
    /// The window around both the shape at `position` and the box from `min` to `max` (where
    /// its stand-ins go), with the scene's shapes below and above it.
    pub fn new(scene: &Scene, position: usize, min: Vec2, max: Vec2) -> Self {
        let shape = &scene.shapes[position];
        let window = Window::around(
            shape.min.min(min),
            shape.max.max(max),
            MARGIN + 2. * ALIGN,
            scene.width,
            scene.height,
        );
        let below = scene.render(window, 0..position, [1.; 3]);
        let above = scene.layer(window, position + 1);
        let mut site = Self {
            window,
            below,
            above,
            drawn: Vec::new(),
        };
        site.drawn = site.shown(shape);
        site
    }

    /// The window with `shape` drawn in the shape's place.
    fn shown(&self, shape: &Shape) -> Vec<Vec3> {
        let scene = crate::render::Scene {
            width: self.window.width as u32,
            height: self.window.height as u32,
            paths: vec![shape.path(self.window.origin())],
        };
        let mut shown = crate::render::render_over(&scene, self.below.clone());
        if let Some(layer) = &self.above {
            for ((s, c), t) in shown.iter_mut().zip(&layer.color).zip(&layer.transmittance) {
                *s = *c + *t * *s;
            }
        }
        shown
    }

    /// How far drawing `shape` in the shape's place changes the scene's picture: per pixel the
    /// squared colour difference beyond the 8-bit noise.
    pub fn change(&self, shape: &Shape) -> f64 {
        self.shown(shape)
            .iter()
            .zip(&self.drawn)
            .map(|(s, d)| crate::uncertainty::excess(s.distance_squared(*d), NOISE as f32) as f64)
            .sum()
    }
}

/// `prototype` placed at `t` standing in for `member`, which keeps its identity in the scene.
pub(super) fn stand_in(prototype: &Shape, member: &Shape, t: Transform) -> Shape {
    Shape {
        id: member.id,
        node: member.node,
        changed: member.changed,
        fitted: member.fitted,
        ..transformed(prototype, t)
    }
}

/// `t` moved where the stand-in for the shape at `site` changes the scene least, by halving
/// steps of a pixel's half in each of its parameters (the rotation if it has one).
fn refit(site: &Site, prototype: &Shape, member: &Shape, t: Transform) -> (Transform, f64) {
    let reach = reach(prototype);
    let error = |t: Transform| site.change(&stand_in(prototype, member, t));
    let (mut t, mut best) = (t, error(t));
    // an exact repeat changes nothing
    if best <= 0. {
        return (t, best);
    }
    let mut step = ALIGN / 2.;
    for _ in 0..REFIT_ROUNDS {
        let along = |offset: Vec2| Transform { offset, ..t };
        let turn = |angle: f32| Transform { angle, ..t };
        let mut moves = vec![
            along(t.offset + Vec2::X * step),
            along(t.offset - Vec2::X * step),
            along(t.offset + Vec2::Y * step),
            along(t.offset - Vec2::Y * step),
        ];
        if !t.is_translation() {
            let da = step / reach;
            moves.extend([turn(t.angle + da), turn(t.angle - da)]);
        }
        for m in moves {
            let e = error(m);
            if e < best {
                (t, best) = (m, e);
            }
        }
        step /= 2.;
    }
    (t, best)
}

/// The member most others of a sample align to, the first of those.
fn medoid(cluster: &[Outline]) -> Option<&Outline> {
    let n = cluster.len();
    let sample: Vec<usize> = (0..n.min(MEDOID_SAMPLE))
        .map(|k| k * n / n.min(MEDOID_SAMPLE))
        .collect();
    let medoid = *sample.iter().max_by_key(|&&j| {
        let aligned = sample
            .iter()
            .filter(|&&i| i == j || align(&cluster[j], &cluster[i]).is_some())
            .count();
        (aligned, std::cmp::Reverse(j))
    })?;
    Some(&cluster[medoid])
}

/// The cluster as it stands, unsnapped: its medoid as the prototype and every member that
/// aligns to it at its transform, for reading the cells as a lattice (see `dual`).
pub(super) fn aligned(scene: &Scene, cluster: &[Outline]) -> Option<Repeat> {
    let lead = medoid(cluster)?;
    let prototype = transformed(scene.get(lead.id), Transform::translation(-lead.centroid));
    let members: Vec<(usize, Transform)> = cluster
        .iter()
        .filter_map(|o| {
            if o.id == lead.id {
                return Some((o.id, Transform::translation(lead.centroid)));
            }
            align(lead, o).map(|t| (o.id, t))
        })
        .collect();
    (members.len() >= 2).then_some(Repeat {
        prototype,
        members,
        structure: Structure::List,
    })
}

/// The cluster's shapes snapped to the member most others align to, those whose snap pays; a
/// repeat of the prototype if two or more remain.
pub(super) fn snap(ctx: &Context, scene: &mut Scene, cluster: &[Outline]) -> Option<Repeat> {
    let lead = medoid(cluster)?;
    let prototype = transformed(scene.get(lead.id), Transform::translation(-lead.centroid));
    let mut members = Vec::new();
    for outline in cluster {
        if outline.id == lead.id {
            members.push((outline.id, Transform::translation(lead.centroid)));
            continue;
        }
        let Some(t) = align(lead, outline) else {
            continue;
        };
        let position = scene.position(outline.id);
        let member = scene.shapes[position].clone();
        let placed = stand_in(&prototype, &member, t);
        let site = Site::new(scene, position, placed.min, placed.max);
        let (t, change) = refit(&site, &prototype, &member, t);
        let saved = (member.cost_segments() - t.segments()) * ctx.accept_cost();
        if change >= saved {
            continue;
        }
        let change = scene.change();
        scene.shapes[position] = Shape {
            changed: change,
            fitted: change,
            ..stand_in(&prototype, &member, t)
        };
        members.push((outline.id, t));
    }
    (members.len() >= 2).then_some(Repeat {
        prototype,
        members,
        structure: Structure::List,
    })
}
