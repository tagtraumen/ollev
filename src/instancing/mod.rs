//! Instancing, a pass over the finished scene: shapes that repeat become one prototype placed by
//! a transform per instance, or by a linear, grid or circular repeat of few parameters; a lattice
//! of cells on a ground showing in its gaps is also read as a ground with strokes along the gaps.

use glam::Vec2;

use crate::{Context, Scene, Shape};

mod dual;
mod find;
mod snap;
mod structure;

/// How far, in pixels, an instance's outline may lie from the prototype placed on it: one
/// anti-aliasing step, within which the two draw the same pixels.
const ALIGN: f32 = crate::edges::SIGMA as f32;
/// Segments a transform costs, at a segment per two numbers: a translation, and a translation
/// with a rotation and a scale.
const TRANSLATION_SEGMENTS: f64 = 1.;
const SIMILARITY_SEGMENTS: f64 = 2.;

/// A similarity: the prototype's point `p` lands at `offset + scale * rotate(angle, p)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub scale: f32,
    pub angle: f32,
    pub offset: Vec2,
}

impl Transform {
    pub fn translation(offset: Vec2) -> Self {
        Self {
            scale: 1.,
            angle: 0.,
            offset,
        }
    }

    pub fn apply(&self, p: Vec2) -> Vec2 {
        self.offset + self.scale * Vec2::from_angle(self.angle).rotate(p)
    }

    pub fn is_translation(&self) -> bool {
        self.scale == 1. && self.angle == 0.
    }

    /// The SVG matrix `a b c d e f`.
    pub fn matrix(&self) -> [f32; 6] {
        let (s, c) = self.angle.sin_cos();
        let k = self.scale;
        [k * c, k * s, -k * s, k * c, self.offset.x, self.offset.y]
    }

    fn segments(&self) -> f64 {
        if self.is_translation() {
            TRANSLATION_SEGMENTS
        } else {
            SIMILARITY_SEGMENTS
        }
    }
}

/// How the instances' transforms are made: listed one by one, or generated from the first by a
/// step `count` times (Graphite's repeat and circular repeat nodes).
#[derive(Clone, Debug, PartialEq)]
pub enum Structure {
    List,
    /// Instance k is the first moved by `k * step`.
    Linear {
        step: Vec2,
    },
    /// Instance k is the first moved by `(k % columns) * a + (k / columns) * b`.
    Grid {
        a: Vec2,
        b: Vec2,
        columns: usize,
    },
    /// Instance k is the first turned about `center` by `k * step` radians.
    Circular {
        center: Vec2,
        step: f32,
    },
}

impl Structure {
    /// The structure's name and parameters: none for a list, the step, the two steps and the
    /// columns of a grid, or the centre and the turn in radians.
    pub fn parameters(&self) -> (&'static str, Vec<f64>) {
        let v = |v: Vec2| [v.x as f64, v.y as f64];
        match self {
            Structure::List => ("list", Vec::new()),
            Structure::Linear { step } => ("linear", v(*step).to_vec()),
            Structure::Grid { a, b, columns } => (
                "grid",
                [v(*a), v(*b)]
                    .concat()
                    .into_iter()
                    .chain([*columns as f64])
                    .collect(),
            ),
            Structure::Circular { center, step } => (
                "circular",
                v(*center).into_iter().chain([*step as f64]).collect(),
            ),
        }
    }

    /// Segments the structure's parameters cost beyond the first instance's transform, for
    /// `members` instances whose transforms are `transforms`.
    fn segments(&self, transforms: &[Transform]) -> f64 {
        match self {
            Structure::List => transforms[1..].iter().map(Transform::segments).sum(),
            // a count and a step: three numbers
            Structure::Linear { .. } => 1.5,
            // a count per direction and two steps
            Structure::Grid { .. } => 3.,
            // a count, a centre and an angle
            Structure::Circular { .. } => 2.,
        }
    }
}

/// A prototype and its instances in the scene.
#[derive(Clone, Debug)]
pub struct Repeat {
    /// The prototype in its own frame, about its centroid.
    pub prototype: Shape,
    /// Each instance's shape id in the scene and the transform that draws it, in the
    /// structure's order.
    pub members: Vec<(usize, Transform)>,
    pub structure: Structure,
}

impl Repeat {
    /// What the repeat costs in segments: the prototype once, the first transform and the
    /// structure's parameters.
    pub fn segments(&self) -> f64 {
        let transforms: Vec<Transform> = self.members.iter().map(|m| m.1).collect();
        self.prototype.cost_segments()
            + transforms[0].segments()
            + self.structure.segments(&transforms)
    }
}

/// `shape` moved by `t`: its outline, its stroke and its primitive.
pub fn transformed(shape: &Shape, t: Transform) -> Shape {
    let mut out = shape.clone();
    let map = |c: &crate::tracer::Cubic| c.map(|p| t.apply(p));
    out.loops = shape
        .loops
        .iter()
        .map(|l| l.iter().map(map).collect())
        .collect();
    out.corners = shape.corners.iter().map(|&p| t.apply(p)).collect();
    if let Some(stroke) = &shape.stroke {
        let mut s = stroke.clone();
        s.centerline = s.centerline.iter().map(map).collect();
        s.width *= t.scale;
        if let Some(d) = &mut s.dashes {
            d.array.iter_mut().for_each(|v| *v *= t.scale);
            d.offset *= t.scale;
        }
        // the outline of a moved stroke is its outline moved, unless the width changes
        if t.scale != 1. {
            out.loops = s.outline();
        }
        out.stroke = Some(s);
    }
    out.primitive = shape.primitive.map(|(k, p)| (k, primitive_moved(p, t)));
    out.update_bounds();
    out
}

fn primitive_moved(
    p: crate::tracer::primitives::Primitive,
    t: Transform,
) -> crate::tracer::primitives::Primitive {
    use crate::tracer::primitives::Primitive;
    match p {
        Primitive::Circle { center, radius } => Primitive::Circle {
            center: t.apply(center),
            radius: radius * t.scale,
        },
        Primitive::Ellipse {
            center,
            radii,
            angle,
        } => Primitive::Ellipse {
            center: t.apply(center),
            radii: radii * t.scale,
            angle: angle + t.angle,
        },
        Primitive::RoundedRect {
            center,
            half,
            radius,
            angle,
        } => Primitive::RoundedRect {
            center: t.apply(center),
            half: half * t.scale,
            radius: radius * t.scale,
            angle: angle + t.angle,
        },
    }
}

/// Whether two shapes cover a pixel in common, by any coverage.
pub(crate) fn overlaps(a: &Shape, b: &Shape, width: usize) -> bool {
    if a.min.x >= b.max.x || b.min.x >= a.max.x || a.min.y >= b.max.y || b.min.y >= a.max.y {
        return false;
    }
    let (ra, rb) = (a.raster(), b.raster());
    let (small, other) = if ra.values.len() <= rb.values.len() {
        (ra, rb)
    } else {
        (rb, ra)
    };
    small
        .pixels(width)
        .any(|(i, v)| v > 0. && other.at(i, width) > 0.)
}

/// Finds the scene's repeats and keeps them in [`Scene::repeats`], after reading each lattice
/// on a ground as strokes where that is cheaper (see [`dual::apply`]).
pub fn instance(ctx: &Context, scene: &mut Scene) {
    scene.unique_ids();
    // A lattice is read from the shapes alike as they stand, before any is snapped
    let taken = crate::HashSet::default();
    let lattices: Vec<Repeat> = find::clusters(scene, &taken)
        .iter()
        .filter_map(|c| snap::aligned(scene, c))
        .collect();
    dual::apply(ctx, scene, &lattices);
    let repeats = repeats_of(ctx, scene);
    if crate::diagnostics::log() {
        for r in &repeats {
            eprintln!(
                "instancing: {} instances of {} segments as {:?}",
                r.members.len(),
                r.prototype.cost_segments(),
                r.structure
            );
        }
    }
    scene.repeats = repeats;
}

/// The repeats of the scene's shapes: clusters of shapes alike, snapped to one prototype where
/// that pays, their transforms read as a structure where that pays, drawn together.
fn repeats_of(ctx: &Context, scene: &mut Scene) -> Vec<Repeat> {
    // The shapes a cluster's snap leaves out are clustered again among themselves
    let mut repeats: Vec<Repeat> = Vec::new();
    let mut taken = crate::HashSet::default();
    loop {
        let before = repeats.len();
        for cluster in find::clusters(scene, &taken) {
            if let Some(repeat) = snap::snap(ctx, scene, cluster) {
                taken.extend(repeat.members.iter().map(|m| m.0));
                repeats.extend(structure::recognize(ctx, scene, repeat));
            }
        }
        if repeats.len() == before {
            return repeats;
        }
    }
}
