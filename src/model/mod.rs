//! The scene model: what the tracer decided, free of any output target. A host converts it to
//! its own content; the trace has already fitted it to what the host draws
//! ([`crate::Capabilities`]). With the `svg` feature, `emit` compiles it to SVG.
//!
//! # The contract
//!
//! - **Space.** Every coordinate is in pixels of the traced image: x to the right, y down, the
//!   image's top left corner at the origin, the image covering `0..width` by `0..height`.
//!   Every length (stroke widths, dash lengths, blurs, a path gradient's distances) is in
//!   pixels too.
//! - **Colour.** sRGB-encoded channels in 0..1, as SVG reads them, with straight alpha;
//!   objects composite source-over in that encoding (as SVG does), over white.
//! - **Stack.** [`Model::items`] are drawn first to last, bottom to top. An opaque trace's
//!   first object is its background, covering the frame; with transparency kept,
//!   [`Model::clip`] holds the picture's opaque part and clips everything.
//! - **Objects.** An [`Object`] covers its [`Geometry`] (a fill's loops by their rule, or a
//!   stroke's pen), its coverage blurred by a Gaussian of standard deviation [`Object::blur`],
//!   then multiplied by its clip's coverage; it paints that coverage times [`Object::opacity`]
//!   times the paint's alpha. A [`Paint::Gradient`] runs in image space ([`GradientShape`]).
//! - **Clips.** [`Model::clips`] are referred to by place. A clip covers its loops by its rule,
//!   in image space, inside its parent clip if it has one; it draws nothing itself.
//! - **Groups.** An [`Item::Group`] holds a translucent layer's objects, drawn in place: the
//!   group adds no opacity or clip of its own, it only tells which objects form one layer.
//! - **Definitions and instances.** [`Model::definitions`] are lists of items in their own
//!   frame, drawn only through instances. An [`Instance`] draws one under its affine
//!   [`Instance::transform`] (the definition's paints included); an [`Item::Repeat`] draws its
//!   instances in order, its [`Structure`] saying how their transforms follow from the first.
//!   Objects inside a definition carry no clip.
//! - **Fidelity.** [`Model::fidelity`] is what an emitter may miss the model's render by; it
//!   draws nothing.

use std::cell::OnceCell;
use std::sync::Arc;

use glam::{DVec2, DVec3, Vec3};

use crate::{Fill, HashMap, Raster, Scene, Shape};

mod gradient;
mod scale;

pub use crate::render::FillRule;
pub use crate::stroke::{Cap, Join};
pub use gradient::{Gradient, GradientShape, Stop};

/// A cubic Bézier in pixels, the image's top left corner at the origin.
pub type Curve = [DVec2; 4];

/// An affine map `[a, b, c, d, e, f]` taking `(x, y)` to `(a x + c y + e, b x + d y + f)`,
/// SVG's matrix order.
pub type Affine = [f64; 6];

/// A traced picture as vector objects, under the contract of the [module](self).
#[derive(Clone, Debug)]
pub struct Model {
    /// The traced image's width in pixels.
    pub width: usize,
    /// The traced image's height in pixels.
    pub height: usize,
    /// What is drawn, bottom to top.
    pub items: Vec<Item>,
    /// The clips objects and clips refer to, by place in this list.
    pub clips: Vec<Clip>,
    /// The groups instances draw, by place in this list.
    pub definitions: Vec<Vec<Item>>,
    /// What the emitted output may miss by, see [`Fidelity`]; `None` checks nothing.
    pub fidelity: Option<Fidelity>,
    /// The clip over the whole drawing, by place in [`Model::clips`]: a transparent picture's
    /// opaque part, where its transparency is kept.
    pub clip: Option<usize>,
}

/// One entry of a stack of items.
#[derive(Clone, Debug)]
pub enum Item {
    /// A drawn object.
    Object(Box<Object>),
    /// A translucent layer's objects.
    Group {
        /// The layer's number, the layers counted bottom to top.
        layer: usize,
        /// The layer's items, bottom to top.
        items: Vec<Item>,
    },
    /// A definition drawn under a map.
    Instance(Instance),
    /// Instances of one definition drawn together, one object to the bench: their transforms
    /// listed, or made from the first by the structure's few parameters (Graphite's repeat nodes).
    Repeat {
        /// How the instances' transforms follow from the first's.
        structure: Structure,
        /// The instances, in drawing order.
        instances: Vec<Instance>,
    },
}

/// A definition drawn again under an affine map.
#[derive(Clone, Debug)]
pub struct Instance {
    /// The definition by place in [`Model::definitions`].
    pub definition: usize,
    /// The map from the definition's frame into the image.
    pub transform: Affine,
}

/// How a repeat's transforms follow from its first instance's, `k` counting the instances
/// from 0 (Graphite's repeat and circular repeat nodes).
#[derive(Clone, Debug, PartialEq)]
pub enum Structure {
    /// Each instance's transform as listed.
    List,
    /// Instance `k` is the first moved by `k * step`.
    Linear {
        /// The move from one instance to the next.
        step: DVec2,
    },
    /// Instance `k` is the first moved by `(k % columns) * a + (k / columns) * b`.
    Grid {
        /// The move along a row.
        a: DVec2,
        /// The move from one row to the next.
        b: DVec2,
        /// The instances in a row.
        columns: usize,
    },
    /// Instance `k` is the first turned about `center` by `k * step` radians.
    Circular {
        /// The point the instances turn about.
        center: DVec2,
        /// The turn from one instance to the next, in radians.
        step: f64,
    },
}

impl Structure {
    /// The structure's name and parameters (the steps' coordinates, a grid's columns, a
    /// turn's centre and angle), as the SVG emitter names them.
    #[must_use]
    pub fn parameters(&self) -> (&'static str, Vec<f64>) {
        match self {
            Structure::List => ("list", Vec::new()),
            Structure::Linear { step } => ("linear", step.to_array().to_vec()),
            Structure::Grid { a, b, columns } => {
                ("grid", vec![a.x, a.y, b.x, b.y, *columns as f64])
            }
            Structure::Circular { center, step } => ("circular", vec![center.x, center.y, *step]),
        }
    }
}

impl From<&crate::instancing::Structure> for Structure {
    fn from(s: &crate::instancing::Structure) -> Self {
        use crate::instancing::Structure as S;
        match *s {
            S::List => Structure::List,
            S::Linear { step } => Structure::Linear {
                step: step.as_dvec2(),
            },
            S::Grid { a, b, columns } => Structure::Grid {
                a: a.as_dvec2(),
                b: b.as_dvec2(),
                columns,
            },
            S::Circular { center, step } => Structure::Circular {
                center: center.as_dvec2(),
                step: step.into(),
            },
        }
    }
}

/// A drawn object: what its geometry covers, painted.
#[derive(Clone, Debug)]
pub struct Object {
    /// What the object covers.
    pub geometry: Geometry,
    /// What the covered pixels are painted with.
    pub paint: Paint,
    /// How much of the paint shows over what lies below, from nothing to all.
    pub opacity: f64,
    /// The standard deviation in pixels of the Gaussian over the coverage, 0 for a hard edge.
    pub blur: f64,
    /// The clip by place in [`Model::clips`]: only what lies within its coverage shows.
    pub clip: Option<usize>,
}

/// What an object covers.
#[derive(Clone, Debug)]
pub enum Geometry {
    /// Closed loops, filled by their rule.
    Fill {
        /// The loops, each a run of cubics ending where it starts.
        loops: Vec<Vec<Curve>>,
        /// How the loops' windings fill.
        rule: FillRule,
    },
    /// A pen along a centerline.
    Stroke(StrokeGeometry),
}

/// A pen drawn along a centerline.
#[derive(Clone, Debug)]
pub struct StrokeGeometry {
    /// The cubics the pen follows, each starting where the one before ends.
    pub centerline: Vec<Curve>,
    /// Whether the centerline closes on itself.
    pub closed: bool,
    /// The pen's width in pixels.
    pub width: f64,
    /// How the open ends (and each dash) end.
    pub cap: Cap,
    /// How the corners are joined.
    pub join: Join,
    /// Which joints of the centerline are corners, where the join shows.
    pub corners: Vec<bool>,
    /// Lengths drawn and left out in turn along the centerline, and the pattern's start.
    pub dashes: Option<(Vec<f64>, f64)>,
}

/// How an object is painted.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum Paint {
    /// An opaque sRGB-encoded colour.
    Solid(DVec3),
    /// A gradient in image space.
    Gradient(Gradient),
}

/// A clip: the coverage of its loops by its rule, inside its own parent clip if any.
#[derive(Clone, Debug)]
pub struct Clip {
    /// Closed loops, as a fill's.
    pub loops: Vec<Vec<Curve>>,
    /// How the loops' windings cover.
    pub rule: FillRule,
    /// The clip this one lies inside, by place in [`Model::clips`].
    pub parent: Option<usize>,
}

/// What an emitted output may differ from the model's own render by: a pixel's colour
/// distance beyond `tolerance` (the picture's noise) is error, priced against a segment at the
/// price the planner accepts a shape by.
#[derive(Clone, Debug)]
pub struct Fidelity {
    /// Per pixel, the colour distance a render may miss by unnoticed; `None` for none.
    pub tolerance: Option<Arc<Vec<f32>>>,
    /// The squared colour error a segment is worth.
    pub segment_price: f64,
}

fn to64(c: &crate::tracer::Cubic) -> Curve {
    c.map(glam::Vec2::as_dvec2)
}

/// The scene's loops in double precision.
pub(crate) fn loops64(loops: &[Vec<crate::tracer::Cubic>]) -> Vec<Vec<Curve>> {
    loops.iter().map(|l| l.iter().map(to64).collect()).collect()
}

fn to32(c: &Curve) -> crate::tracer::Cubic {
    c.map(glam::DVec2::as_vec2)
}

/// The box of the loops' control points, in single precision as the scene keeps it.
pub(crate) fn bounds(loops: &[Vec<Curve>]) -> (glam::Vec2, glam::Vec2) {
    let points = loops.iter().flatten().flatten().map(|p| p.as_vec2());
    let min = points.clone().fold(glam::Vec2::INFINITY, glam::Vec2::min);
    let max = points.fold(glam::Vec2::NEG_INFINITY, glam::Vec2::max);
    (min, max)
}

/// Loops in the scene's single precision.
fn loops32(loops: &[Vec<Curve>]) -> Vec<Vec<crate::tracer::Cubic>> {
    loops.iter().map(|l| l.iter().map(to32).collect()).collect()
}

impl Paint {
    /// The paint as the scene's fill at `opacity`.
    pub(crate) fn fill(&self, opacity: f64) -> Fill {
        let (color, gradient) = match self {
            Paint::Solid(c) => (c.as_vec3(), None),
            Paint::Gradient(g) => {
                let (from, ramp) = g.ramp();
                (from, Some(*ramp))
            }
        };
        Fill {
            color,
            gradient,
            opacity: opacity as f32,
        }
    }
}

impl Geometry {
    /// The segments as usvg counts them: a move, the cubics and a close per loop.
    #[must_use]
    pub fn segments(&self) -> usize {
        match self {
            Geometry::Fill { loops, .. } => loops
                .iter()
                .filter(|l| !l.is_empty())
                .map(|l| l.len() + 2)
                .sum(),
            Geometry::Stroke(s) => s.centerline.len() + 1 + usize::from(s.closed),
        }
    }

    /// The loops the geometry covers and their rule: a stroke's outline, filled nonzero.
    pub(crate) fn outline(&self) -> (Vec<Vec<crate::tracer::Cubic>>, FillRule) {
        match self {
            Geometry::Fill { loops, rule } => (loops32(loops), *rule),
            Geometry::Stroke(s) => (s.stroke().outline(), FillRule::NonZero),
        }
    }

    /// Maps every point by `f`, a stroke's width and dashes kept.
    pub(crate) fn map(&mut self, f: &impl Fn(DVec2) -> DVec2) {
        let curves: &mut dyn Iterator<Item = &mut Curve> = match self {
            Geometry::Fill { loops, .. } => &mut loops.iter_mut().flatten(),
            Geometry::Stroke(s) => &mut s.centerline.iter_mut(),
        };
        curves.flatten().for_each(|p| *p = f(*p));
    }

    /// The geometry scaled by `s` about the origin, a stroke's width and dashes with it.
    #[must_use]
    pub fn scaled(&self, s: f64) -> Self {
        let mut g = self.clone();
        g.map(&|p| p * s);
        if let Geometry::Stroke(k) = &mut g {
            k.width *= s;
            if let Some((array, offset)) = &mut k.dashes {
                for v in array {
                    *v *= s;
                }
                *offset *= s;
            }
        }
        g
    }
}

impl StrokeGeometry {
    /// The tracer's stroke this describes.
    pub(crate) fn stroke(&self) -> crate::stroke::Stroke {
        crate::stroke::Stroke {
            centerline: self.centerline.iter().map(to32).collect(),
            closed: self.closed,
            width: self.width as f32,
            cap: self.cap,
            join: self.join,
            corners: self.corners.clone(),
            dashes: self.dashes.as_ref().map(|(a, o)| crate::stroke::Dashes {
                array: a.iter().map(|&v| v as f32).collect(),
                offset: *o as f32,
            }),
        }
    }
}

impl Object {
    /// The segments as usvg counts them, and a path gradient's curve counted as one more loop.
    #[must_use]
    pub fn segments(&self) -> usize {
        let curve = if let Paint::Gradient(g) = &self.paint
            && let crate::render::Gradient::Path { contour, .. } = g.ramp().1
        {
            contour.len() + 2
        } else {
            0
        };
        self.geometry.segments() + curve
    }

    /// The object of a scene shape clipped to `clip`.
    pub(crate) fn of_shape(shape: &Shape, clip: Option<usize>) -> Self {
        let geometry = match &shape.stroke {
            Some(s) => Geometry::Stroke(StrokeGeometry {
                centerline: s.centerline.iter().map(to64).collect(),
                closed: s.closed,
                width: s.width as f64,
                cap: s.cap,
                join: s.join,
                corners: s.corners.clone(),
                dashes: s
                    .dashes
                    .as_ref()
                    .map(|d| (d.array.iter().map(|&v| v as f64).collect(), d.offset as f64)),
            }),
            None => Geometry::Fill {
                loops: loops64(&shape.loops),
                rule: shape.fill_rule(),
            },
        };
        let paint = match shape.fill.gradient {
            None => Paint::Solid(shape.fill.color.as_dvec3()),
            Some(gradient) => Paint::Gradient(Gradient::new(shape.fill.color, gradient)),
        };
        Object {
            geometry,
            paint,
            opacity: shape.fill.opacity as f64,
            blur: shape.blur as f64,
            clip,
        }
    }

    /// The box `(x0, y0, w, h)` the object paints, for the lab.
    #[cfg(feature = "diagnostics")]
    #[doc(hidden)]
    pub fn painted_box(&self) -> (usize, usize, usize, usize) {
        let r = self.painted();
        (r.x0, r.y0, r.w, r.h)
    }

    /// What the object paints over the pixels of its box: its coverage, blurred where it is.
    pub(crate) fn painted(&self) -> Raster {
        let (loops, rule) = self.geometry.outline();
        let path = crate::scene::loops_path(&loops);
        if self.blur > 0. {
            crate::scene::blurred_raster(&path, rule, self.blur as f32)
        } else {
            crate::scene::coverage_raster(&path, rule)
        }
    }
}

impl Item {
    fn objects<'a>(&'a self, out: &mut Vec<&'a Object>) {
        match self {
            Item::Object(o) => out.push(o),
            Item::Group { items, .. } => items.iter().for_each(|i| i.objects(out)),
            Item::Instance(_) | Item::Repeat { .. } => {}
        }
    }
}

impl Model {
    /// The drawn objects bottom to top, an instance's not included.
    #[must_use]
    pub fn objects(&self) -> Vec<&Object> {
        let mut out = Vec::new();
        self.items.iter().for_each(|i| i.objects(&mut out));
        out
    }

    /// The objects and their segments, as the bench counts them: every drawn object once, an
    /// instance or a repeat once plus its definition; clips, paints and blurs are attributes.
    #[must_use]
    pub fn count(&self) -> (usize, usize) {
        fn walk<'a>(items: &'a [Item], objects: &mut Vec<&'a Object>, instances: &mut usize) {
            for item in items {
                match item {
                    Item::Object(o) => objects.push(o),
                    Item::Group { items, .. } => walk(items, objects, instances),
                    Item::Instance(_) | Item::Repeat { .. } => *instances += 1,
                }
            }
        }
        let (mut objects, mut instances) = (Vec::new(), 0);
        walk(&self.items, &mut objects, &mut instances);
        for d in &self.definitions {
            walk(d, &mut objects, &mut instances);
        }
        let segments = objects.iter().map(|o| o.segments()).sum();
        (objects.len() + instances, segments)
    }

    /// The objects drawn in stack order, an instance's mapped by its transform, each with
    /// whether it is one of [`Model::objects`] (drawn outside any instance).
    #[must_use]
    pub fn drawn(&self) -> Vec<(bool, Object)> {
        fn walk(
            model: &Model,
            items: &[Item],
            map: Option<&[f64; 6]>,
            out: &mut Vec<(bool, Object)>,
        ) {
            for item in items {
                match item {
                    Item::Object(o) => {
                        let mut o = (**o).clone();
                        if let Some(m) = map {
                            o.geometry.map(&|p| apply(m, p));
                        }
                        out.push((map.is_none(), o));
                    }
                    Item::Group { items, .. } => walk(model, items, map, out),
                    Item::Instance(i) => instance(model, i, map, out),
                    Item::Repeat { instances, .. } => {
                        for i in instances {
                            instance(model, i, map, out);
                        }
                    }
                }
            }
        }
        fn instance(
            model: &Model,
            i: &Instance,
            map: Option<&[f64; 6]>,
            out: &mut Vec<(bool, Object)>,
        ) {
            let t = match map {
                Some(m) => compose(m, &i.transform),
                None => i.transform,
            };
            walk(model, &model.definitions[i.definition], Some(&t), out);
        }
        let mut out = Vec::new();
        walk(self, &self.items, None, &mut out);
        out
    }

    /// The model rendered over white with the scene's compositing, and per pixel the object of
    /// [`Model::objects`] that draws most of it, `u32::MAX` for none.
    #[must_use]
    pub fn render(&self) -> (Vec<Vec3>, Vec<u32>) {
        self.composite([0, 0, self.width, self.height], true)
    }

    /// The model rendered over white in the window `[x0, y0, w, h]` of the image, row by row.
    #[must_use]
    pub fn render_window(&self, window: [usize; 4]) -> Vec<Vec3> {
        self.composite(window, false).0
    }

    /// [`Model::render`] in a window, with the owners where `owners` is asked for.
    fn composite(&self, [x0, y0, ww, wh]: [usize; 4], owners: bool) -> (Vec<Vec3>, Vec<u32>) {
        let (w, h) = (self.width, self.height);
        let (x1, y1) = ((x0 + ww).min(w), (y0 + wh).min(h));
        let mut pixels = vec![Vec3::ONE; ww * wh];
        let clip_rasters: Vec<OnceCell<Raster>> =
            self.clips.iter().map(|_| OnceCell::new()).collect();
        let clip_at = |c: usize, x: usize, y: usize| -> f32 {
            let mut c = Some(c);
            let mut v = 1.;
            while let Some(k) = c {
                let clip = &self.clips[k];
                let r = clip_rasters[k].get_or_init(|| {
                    let path = crate::scene::loops_path(&loops32(&clip.loops));
                    crate::scene::coverage_raster(&path, clip.rule)
                });
                v *= r.at(y * w + x, w);
                c = clip.parent;
            }
            v
        };
        // The owner's share of the pixel's colour, which each object drawn over it scales down
        let n = if owners { ww * wh } else { 0 };
        let (mut owner, mut share) = (vec![u32::MAX; n], vec![0f32; n]);
        let mut k = 0u32;
        for (direct, o) in self.drawn() {
            let raster = o.painted();
            let fill = o.paint.fill(o.opacity);
            for y in raster.y0.max(y0)..(raster.y0 + raster.h).min(y1) {
                for x in raster.x0.max(x0)..(raster.x0 + raster.w).min(x1) {
                    let i = (y - y0) * ww + x - x0;
                    let alpha = raster.values[(y - raster.y0) * raster.w + x - raster.x0]
                        * o.clip.map_or(1., |c| clip_at(c, x, y))
                        * self.clip.map_or(1., |c| clip_at(c, x, y));
                    let p = pixels[i];
                    let (c, a) = fill.paint_pixel(x, y);
                    let a = alpha * a;
                    pixels[i] = p + a * (c - p);
                    if owners {
                        share[i] *= 1. - a;
                        if direct && a > share[i] {
                            (owner[i], share[i]) = (k, a);
                        }
                    }
                }
            }
            k += u32::from(direct);
        }
        (pixels, owner)
    }
}

fn apply(m: &[f64; 6], p: DVec2) -> DVec2 {
    DVec2::new(
        m[0] * p.x + m[2] * p.y + m[4],
        m[1] * p.x + m[3] * p.y + m[5],
    )
}

/// The map `outer` after `inner`.
fn compose(outer: &[f64; 6], inner: &[f64; 6]) -> [f64; 6] {
    let o = |p: DVec2| apply(outer, p);
    let origin = o(apply(inner, DVec2::ZERO));
    let x = o(apply(inner, DVec2::X)) - origin;
    let y = o(apply(inner, DVec2::Y)) - origin;
    [x.x, x.y, y.x, y.y, origin.x, origin.y]
}

impl Scene {
    /// The scene as a model: its shapes in stack order, each layer's runs as groups, the
    /// layer's marks at the end of its first run.
    #[must_use]
    pub fn model(&self) -> Model {
        /// The clip of `shape` inside `parent`.
        fn shape_clip(shape: &Shape, parent: Option<usize>) -> Clip {
            Clip {
                loops: loops64(&shape.loops),
                rule: shape.fill_rule(),
                parent,
            }
        }
        /// The place of the clip to the shape with `id`, added with its parents where new.
        fn clip_of(
            scene: &Scene,
            id: usize,
            clips: &mut Vec<Clip>,
            keys: &mut HashMap<usize, usize>,
        ) -> Option<usize> {
            if let Some(&k) = keys.get(&id) {
                return Some(k);
            }
            let shape = scene.shapes.iter().find(|s| s.id == id)?;
            let parent = shape.clip.and_then(|p| clip_of(scene, p, clips, keys));
            clips.push(shape_clip(shape, parent));
            keys.insert(id, clips.len() - 1);
            Some(clips.len() - 1)
        }
        let mut clips: Vec<Clip> = Vec::new();
        let mut keys: HashMap<usize, usize> = HashMap::default();
        // A clipped mark shows through its layer's shape, the first of the layer's group: by
        // the shape's id where that names it alone, else by its outline
        let layer_shape = self
            .marks_layer
            .and_then(|_| self.shapes.iter().find(|s| s.layer == self.marks_layer));
        let mut marks: Option<Vec<Item>> = (!self.marks.is_empty()).then(|| {
            let clipped = self.marks.iter().any(|o| o.clip.is_some());
            let clip = layer_shape.filter(|_| clipped).map(|s| {
                if self.shapes.iter().filter(|q| q.id == s.id).count() == 1
                    && let Some(k) = clip_of(self, s.id, &mut clips, &mut keys)
                {
                    return k;
                }
                let parent = s.clip.and_then(|p| clip_of(self, p, &mut clips, &mut keys));
                clips.push(shape_clip(s, parent));
                clips.len() - 1
            });
            self.marks
                .iter()
                .map(|o| {
                    Item::Object(Box::new(Object {
                        clip: o.clip.and(clip),
                        ..o.clone()
                    }))
                })
                .collect()
        });
        let mut items = Vec::new();
        let mut run: Option<(usize, Vec<Item>)> = None;
        let layer = self.marks_layer;
        let close = |run: &mut Option<(usize, Vec<Item>)>,
                     items: &mut Vec<Item>,
                     marks: &mut Option<Vec<Item>>| {
            if let Some((n, mut group)) = run.take() {
                if Some(n) == layer
                    && let Some(m) = marks.take()
                {
                    group.extend(m);
                }
                items.push(Item::Group {
                    layer: n,
                    items: group,
                });
            }
        };
        // An instance draws its repeat's prototype, a repeat's run of them in one item
        let members: HashMap<usize, (usize, usize)> = self
            .repeats
            .iter()
            .enumerate()
            .flat_map(|(r, repeat)| {
                repeat
                    .members
                    .iter()
                    .enumerate()
                    .map(move |(k, m)| (m.0, (r, k)))
            })
            .collect();
        for shape in &self.shapes {
            if let Some(&(r, k)) = members.get(&shape.id) {
                let repeat = &self.repeats[r];
                let instance = Instance {
                    definition: r,
                    transform: repeat.members[k].1.matrix().map(f64::from),
                };
                if let Some(Item::Repeat { instances, .. }) = items.last_mut()
                    && instances[0].definition == r
                {
                    instances.push(instance);
                } else {
                    items.push(Item::Repeat {
                        structure: (&repeat.structure).into(),
                        instances: vec![instance],
                    });
                }
                continue;
            }
            let clip = shape
                .clip
                .and_then(|c| clip_of(self, c, &mut clips, &mut keys));
            let object = Item::Object(Box::new(Object::of_shape(shape, clip)));
            if run.as_ref().map(|r| r.0) != shape.layer {
                close(&mut run, &mut items, &mut marks);
                run = shape.layer.map(|n| (n, Vec::new()));
            }
            match &mut run {
                Some((_, group)) => group.push(object),
                None => items.push(object),
            }
        }
        // The marks close their layer's group when no earlier run of it took them
        if let Some(n) = layer
            && marks.is_some()
            && run.as_ref().map(|r| r.0) != Some(n)
        {
            close(&mut run, &mut items, &mut marks);
            run = Some((n, Vec::new()));
        }
        close(&mut run, &mut items, &mut marks);
        if let Some(m) = marks {
            items.extend(m);
        }
        let mut model = Model {
            width: self.width,
            height: self.height,
            items,
            clips,
            definitions: self
                .repeats
                .iter()
                .map(|r| vec![Item::Object(Box::new(Object::of_shape(&r.prototype, None)))])
                .collect(),
            fidelity: self.fidelity.clone(),
            clip: None,
        };
        model.drop_empty_loops();
        model
    }
}

/// Whether a loop covers nothing: no cubic, or every point on one line, as the one cubic of no
/// length the cleanup or a simplification leaves of a small hole.
pub(crate) fn covers_nothing(cubics: &[Curve]) -> bool {
    let mut points = cubics.iter().flatten();
    let Some(&a) = points.next() else {
        return true;
    };
    let Some(&b) = points.clone().find(|p| **p != a) else {
        return true;
    };
    points.all(|p| (b - a).perp_dot(*p - a) == 0.)
}

impl Model {
    /// Leaves out the fill and clip loops that cover nothing (see [`covers_nothing`]) and the
    /// fills left without a loop; clips keep their places, which objects refer to.
    fn drop_empty_loops(&mut self) {
        fn prune(items: &mut Vec<Item>) {
            items.retain_mut(|item| match item {
                Item::Object(o) => match &mut o.geometry {
                    Geometry::Fill { loops, .. } => {
                        loops.retain(|l| !covers_nothing(l));
                        !loops.is_empty()
                    }
                    Geometry::Stroke(_) => true,
                },
                Item::Group { items, .. } => {
                    prune(items);
                    true
                }
                Item::Instance(_) | Item::Repeat { .. } => true,
            });
        }
        prune(&mut self.items);
        self.definitions.iter_mut().for_each(prune);
        for clip in &mut self.clips {
            clip.loops.retain(|l| !covers_nothing(l));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Window, scene::tests::square};
    use glam::Vec2;

    #[test]
    fn loops_of_no_length_or_area_cover_nothing() {
        let p = |x: f64, y: f64| DVec2::new(x, y);
        let point = [p(3., 4.); 4];
        assert!(covers_nothing(&[]));
        assert!(covers_nothing(&[point]));
        assert!(covers_nothing(&[
            [p(0., 0.), p(1., 1.), p(2., 2.), p(3., 3.)],
            [p(3., 3.), p(2., 2.), p(1., 1.), p(0., 0.)]
        ]));
        assert!(!covers_nothing(&[[
            p(0., 0.),
            p(4., 0.),
            p(4., 4.),
            p(0., 0.)
        ]]));
    }

    /// The model of a scene renders as the scene does, clips and blurs included.
    #[test]
    fn the_model_renders_as_the_scene() {
        let mut scene = Scene::with_background(24, 24, Vec3::ONE);
        let parent = scene.push(Shape::new(
            square(Vec2::splat(4.), Vec2::splat(20.)),
            Vec3::ZERO,
        ));
        let child = Shape::new(square(Vec2::new(2., 2.), Vec2::new(12., 22.)), Vec3::X);
        scene.push(Shape {
            clip: Some(parent),
            ..child.with_blur(2.)
        });
        scene.push(Shape {
            layer: Some(0),
            ..Shape::new(
                square(Vec2::splat(8.), Vec2::splat(16.)),
                Fill {
                    opacity: 0.4,
                    ..Fill::solid(Vec3::Y)
                },
            )
        });
        // A glow fading to transparent
        scene.push(Shape::new(
            square(Vec2::splat(1.), Vec2::splat(23.)),
            Fill {
                color: Vec3::new(1., 0.8, 0.3),
                gradient: Some(crate::render::Gradient::Radial {
                    center: Vec2::splat(12.),
                    focus: Vec2::ZERO,
                    radius: 10.,
                    transform: glam::Mat2::IDENTITY,
                    to: [1., 0.8, 0.3, 0.],
                    stops: crate::render::Stops::NONE,
                }),
                opacity: 1.,
            },
        ));
        let want = scene.render(Window::full(24, 24), 0..5, [1.; 3]);
        let (got, owner) = scene.model().render();
        assert_eq!(owner[12 * 24 + 12], 4);
        assert_eq!(owner[0], 0);
        assert_eq!(want, got);
    }
}
