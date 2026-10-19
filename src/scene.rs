//! The stack of shapes being built.

use std::sync::{Arc, OnceLock};

use crate::render::{FillRule, Gradient, Path, Subpath};
use crate::tracer::Cubic;
use glam::{Vec2, Vec3};

use crate::Window;
use kept::Kept;

mod kept;

/// What a shape is painted with: one color, or a gradient from that color at its start through
/// its stops to another at its end, the rasterizer's, held constant beyond the ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fill {
    pub color: Vec3,
    pub gradient: Option<Gradient>,
    /// How much of the fill shows over what lies below, from nothing to all: a translucent
    /// shape composites as `below + opacity * (fill - below)`.
    pub opacity: f32,
}

impl Fill {
    /// One opaque colour.
    pub fn solid(color: Vec3) -> Self {
        Self {
            color,
            gradient: None,
            opacity: 1.,
        }
    }

    /// The color at `p`.
    pub fn at(&self, p: Vec2) -> Vec3 {
        match &self.gradient {
            None => self.color,
            Some(g) => g.color(self.color, g.t(p)),
        }
    }

    /// The color at the center of pixel `(x, y)`.
    pub fn at_pixel(&self, x: usize, y: usize) -> Vec3 {
        self.at(Vec2::new(x as f32 + 0.5, y as f32 + 0.5))
    }

    /// The straight colour at `p` and how much of it shows: the opacity times the ramp's alpha.
    pub fn paint(&self, p: Vec2) -> (Vec3, f32) {
        match &self.gradient {
            Some(g) if !g.is_opaque() => {
                let (color, alpha) = g.rgba(self.color, g.t(p));
                (color, self.opacity * alpha)
            }
            _ => (self.at(p), self.opacity),
        }
    }

    /// [`Fill::paint`] at the center of pixel `(x, y)`.
    pub fn paint_pixel(&self, x: usize, y: usize) -> (Vec3, f32) {
        self.paint(Vec2::new(x as f32 + 0.5, y as f32 + 0.5))
    }

    /// Whether the fill covers what lies below wherever it is drawn.
    #[cfg(test)]
    pub fn is_opaque(&self) -> bool {
        self.opacity >= 1. && self.gradient.is_none_or(|g| g.is_opaque())
    }

    /// Whether the two fills look different: a color differs by more than half a level of 8
    /// bits in some channel, or the gradient does.
    pub fn differs(&self, o: &Self) -> bool {
        let differ = |a: Vec3, b: Vec3| (a - b).abs().max_element() > 0.5 / 255.;
        differ(self.color, o.color)
            || (self.opacity - o.opacity).abs() > 0.5 / 255.
            || match (&self.gradient, &o.gradient) {
                (None, None) => false,
                (Some(a), Some(b)) => a != b,
                _ => true,
            }
    }
}

impl From<Vec3> for Fill {
    fn from(color: Vec3) -> Self {
        Self::solid(color)
    }
}

#[derive(Clone, Debug)]
pub struct Shape {
    /// Closed loops, filled even-odd.
    pub loops: Vec<Vec<Cubic>>,
    pub fill: Fill,
    /// Bounding box of the control points, which contains the curves.
    pub min: Vec2,
    pub max: Vec2,
    /// The id of the shape below this one it is clipped to, so only what lies within that
    /// shows.
    pub clip: Option<usize>,
    /// Which shape this is, whatever its place in the stack; given by the scene.
    pub id: usize,
    /// When the shape last changed and when it was last fitted, in the scene's counter of
    /// changes, so a shape fitted after everything around it last changed needs no refit.
    pub changed: u64,
    pub fitted: u64,
    /// The loop that is an exact circle, ellipse or rounded rectangle, if any, which is what an
    /// artist would draw and so costs less than its segments, see [`Shape::cost_segments`].
    pub primitive: Option<(usize, crate::tracer::primitives::Primitive)>,
    /// The shape's coverage of its bounding box, once something asked for it; cleared whenever
    /// the outline changes through [`Shape::update_bounds`].
    pub raster: OnceLock<Arc<Raster>>,
    /// What a blurred shape paints over its box, its raster blurred; cleared with the raster.
    pub painted: OnceLock<Arc<Raster>>,
    /// A gradient fill's paint over the painted box, the first time a render asks for it; kept
    /// with the fill and box it was read for, so a shape given another fill reads it again.
    pub paint: Kept<Option<Arc<Paint>>>,
    /// Where the trace found sharp corners, kept through every rebuild of the outline so a
    /// corner the fit rounded a little is not simplified away. Empty means the angle alone
    /// decides, see [`Shape::corner_flags`].
    pub corners: Vec<Vec2>,
    /// The stroke this shape is, if it was recognized as one: its loops are then the
    /// stroke's outline, and the SVG writes the stroked centerline.
    pub stroke: Option<crate::stroke::Stroke>,
    /// Whether the shape's edges are anti-aliased in the image, so interpolating them (as a
    /// primitive or a stroke does) is what the image asks for; crisp pixel art is not.
    pub smooth: bool,
    /// Whether the shape's edges are anti-aliased at all, whatever the colours inside, since a
    /// network of strokes may run through a region of more than one colour.
    pub antialiased: bool,
    /// The region of the hierarchy the shape was drawn for, whose pixels are the honest
    /// mask of what it covers.
    pub node: Option<u32>,
    /// The translucent layer the shape belongs to, numbered bottom to top (a layer's own shape
    /// and its lines), so the SVG writes the layer as a group of its own; `None` for the
    /// opaque scene.
    pub layer: Option<usize>,
    /// The standard deviation in pixels of the Gaussian blur over the shape's coverage, 0 for
    /// a hard edge; the blurred coverage is what it paints, and a clip cuts it crisp.
    pub blur: f32,
}

/// A fill's colour and alpha per pixel of a box `[x0, y0, w, h]`, row by row.
#[derive(Debug)]
pub struct Paint {
    fill: Fill,
    at: [usize; 4],
    values: Vec<(Vec3, f32)>,
}

impl Paint {
    /// The paint at image pixel `(x, y)` of the box.
    pub fn at(&self, x: usize, y: usize) -> (Vec3, f32) {
        self.values[(y - self.at[1]) * self.at[2] + x - self.at[0]]
    }
}

/// Points by square cells of a side, so those within a side of a point are found in the nine
/// cells about it.
struct PointGrid {
    side: f32,
    /// (cell, point), sorted
    sorted: Vec<((i32, i32), usize)>,
}

impl PointGrid {
    fn of(points: &[Vec2], side: f32) -> Self {
        // a little wider than asked, past the division's rounding
        let side = 1.001 * side;
        let mut sorted: Vec<((i32, i32), usize)> = points
            .iter()
            .enumerate()
            .map(|(k, p)| (Self::cell_of(*p, side), k))
            .collect();
        sorted.sort_unstable();
        Self { side, sorted }
    }

    fn cell_of(p: Vec2, side: f32) -> (i32, i32) {
        let c = (p / side).floor();
        (c.y as i32, c.x as i32)
    }

    /// The points in the nine cells about `p`, every one within a side of it among them.
    fn near(&self, p: Vec2) -> impl Iterator<Item = usize> + '_ {
        let (cy, cx) = Self::cell_of(p, self.side);
        (-1..=1).flat_map(move |dy| {
            let row = cy.saturating_add(dy);
            let (lo, hi) = ((row, cx.saturating_sub(1)), (row, cx.saturating_add(1)));
            let a = self.sorted.partition_point(|e| e.0 < lo);
            let b = self.sorted.partition_point(|e| e.0 <= hi);
            self.sorted[a..b].iter().map(|e| e.1)
        })
    }
}

/// How far a joint may lie from a remembered corner to be it, in pixels.
pub const CORNER_MATCH: f32 = 1.5;
/// How far a remembered corner follows the nearest joint after a fit before it is forgotten.
const CORNER_FOLLOW: f32 = 3.;
/// Least turn at which a remembered corner still counts as one.
const CORNER_LEAST: f32 = 20f32.to_radians();

/// Share of its segments' cost a primitive loop costs.
pub const PRIMITIVE_SHARE: f64 = 0.5;

impl Shape {
    pub fn new(loops: Vec<Vec<Cubic>>, fill: impl Into<Fill>) -> Self {
        let mut shape = Self {
            loops,
            fill: fill.into(),
            min: Vec2::ZERO,
            max: Vec2::ZERO,
            clip: None,
            id: usize::MAX,
            primitive: None,
            changed: 0,
            fitted: 0,
            raster: OnceLock::new(),
            painted: OnceLock::new(),
            paint: Kept::new(None),
            corners: Vec::new(),
            stroke: None,
            smooth: false,
            antialiased: false,
            node: None,
            layer: None,
            blur: 0.,
        };
        shape.update_bounds();
        shape
    }

    /// The center of the shape's circle or ellipse primitive, where a radial gradient would
    /// start.
    pub fn primitive_center(&self) -> Option<Vec2> {
        use crate::tracer::primitives::Primitive;
        match self.primitive {
            Some((_, Primitive::Circle { center, .. } | Primitive::Ellipse { center, .. })) => {
                Some(center)
            }
            _ => None,
        }
    }

    /// A shape of `loops` taking this one's fill, clip, id, edge flags, node, layer and blur.
    #[must_use]
    pub fn successor(&self, loops: Vec<Vec<Cubic>>) -> Self {
        Shape {
            clip: self.clip,
            id: self.id,
            smooth: self.smooth,
            antialiased: self.antialiased,
            node: self.node,
            layer: self.layer,
            blur: self.blur,
            ..Shape::new(loops, self.fill)
        }
    }

    /// The shape with these remembered corners.
    #[must_use]
    pub fn with_corners(mut self, corners: Vec<Vec2>) -> Self {
        self.corners = corners;
        self.snap_corners();
        self
    }

    /// Whether each joint of loop `l` (the start of its cubic) is a corner: turning by more than
    /// `angle`, or within [`CORNER_MATCH`] of a remembered corner and turning by more than
    /// [`CORNER_LEAST`] (a traced corner the fit flattened was staircase noise).
    pub fn corner_flags(&self, l: usize, angle: f32) -> Vec<bool> {
        let grid = PointGrid::of(&self.corners, CORNER_MATCH);
        let near = |joint: Vec2| {
            grid.near(joint)
                .any(|k| self.corners[k].distance(joint) <= CORNER_MATCH)
        };
        (0..self.loops[l].len())
            .map(|j| self.corner_flag_by(l, j, angle, &near))
            .collect()
    }

    /// [`Shape::corner_flags`] of joint `j` of loop `l` alone.
    pub fn corner_flag(&self, l: usize, j: usize, angle: f32) -> bool {
        let near = |joint: Vec2| {
            self.corners
                .iter()
                .any(|c| c.distance(joint) <= CORNER_MATCH)
        };
        self.corner_flag_by(l, j, angle, &near)
    }

    fn corner_flag_by(&self, l: usize, j: usize, angle: f32, near: &impl Fn(Vec2) -> bool) -> bool {
        let cubics = &self.loops[l];
        let remembered = near(cubics[j][0]);
        crate::complete::joint_turn(cubics, j)
            .is_some_and(|t| t.abs() > angle || (remembered && t.abs() > CORNER_LEAST))
    }

    /// Moves every remembered corner onto the nearest joint of the outline when one lies
    /// within [`CORNER_FOLLOW`], and forgets it otherwise: after a fit the joints have moved,
    /// after a simplification some are gone.
    pub fn snap_corners(&mut self) {
        let joints: Vec<Vec2> = self.loops.iter().flatten().map(|c| c[0]).collect();
        // the nearest joint within the reach, the first of equals, from the joints in the
        // cells about the corner; a joint taken already is the one a snapped corner sits on
        let grid = PointGrid::of(&joints, CORNER_FOLLOW);
        let mut taken = vec![false; joints.len()];
        let mut snapped: Vec<Vec2> = Vec::with_capacity(self.corners.len());
        for corner in &self.corners {
            let nearest = grid
                .near(*corner)
                .map(|k| (joints[k].distance(*corner), k))
                .filter(|(d, _)| *d <= CORNER_FOLLOW)
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                .map(|(_, k)| k);
            if let Some(k) = nearest
                && !grid
                    .near(joints[k])
                    .any(|t| taken[t] && joints[t].distance(joints[k]) < 1e-3)
            {
                taken[k] = true;
                snapped.push(joints[k]);
            }
        }
        self.corners = snapped;
    }

    pub fn segments(&self) -> usize {
        self.loops.iter().map(Vec::len).sum()
    }

    /// The segments this shape is charged for: all of them, those of a primitive loop at a share.
    pub fn cost_segments(&self) -> f64 {
        if let Some(stroke) = &self.stroke {
            return stroke.segments() as f64;
        }
        let discount = self.primitive.map_or(0., |(k, _)| {
            self.loops[k].len() as f64 * (1. - PRIMITIVE_SHARE)
        });
        let blur = if self.blur > 0. {
            crate::blur::BLUR_SEGMENTS
        } else {
            0.
        };
        self.segments() as f64 - discount + blur
    }

    /// Recomputes the bounding box after the outline changed, and forgets the raster.
    pub fn update_bounds(&mut self) {
        let points = self.loops.iter().flatten().flatten();
        self.min = points.clone().fold(Vec2::INFINITY, |a, &p| a.min(p));
        self.max = points.fold(Vec2::NEG_INFINITY, |a, &p| a.max(p));
        self.raster = OnceLock::new();
        self.painted = OnceLock::new();
    }

    /// Moves the shape by `offset` (a window's trace into its frame), what it paints forgotten.
    #[cfg(test)]
    pub fn translate(&mut self, offset: Vec2) {
        self.loops
            .iter_mut()
            .flatten()
            .flatten()
            .for_each(|p| *p += offset);
        self.corners.iter_mut().for_each(|p| *p += offset);
        if let Some(g) = &mut self.fill.gradient {
            g.translate(offset);
        }
        if let Some(s) = &mut self.stroke {
            s.centerline.iter_mut().flatten().for_each(|p| *p += offset);
        }
        if let Some((_, p)) = &mut self.primitive {
            use crate::tracer::primitives::Primitive;
            match p {
                Primitive::Circle { center, .. }
                | Primitive::Ellipse { center, .. }
                | Primitive::RoundedRect { center, .. } => *center += offset,
            }
        }
        self.update_bounds();
        *self.paint.lock() = None;
    }

    /// The shape with its blur set to `sigma` pixels, what it paints forgotten.
    #[must_use]
    pub fn with_blur(mut self, sigma: f32) -> Self {
        self.blur = sigma;
        self.painted = OnceLock::new();
        self
    }

    /// The pixels the blur reaches beyond the outline.
    pub fn reach(&self) -> f32 {
        if self.blur > 0. {
            crate::render::blur::reach(self.blur) as f32
        } else {
            0.
        }
    }

    /// The box of what the shape paints: its bounding box grown by the blur's reach.
    pub fn paint_min(&self) -> Vec2 {
        self.min - Vec2::splat(self.reach())
    }

    pub fn paint_max(&self) -> Vec2 {
        self.max + Vec2::splat(self.reach())
    }

    /// What the shape paints over the pixels of its box within the image: its coverage, blurred
    /// when the shape is, computed the first time it is asked for.
    pub fn painted(&self) -> &Raster {
        if self.blur == 0. {
            return self.raster();
        }
        self.painted
            .get_or_init(|| Arc::new(blurred_raster(&self.path(Vec2::ZERO), self.fill_rule(), self.blur)))
    }

    /// [`Fill::paint_pixel`] over the painted box (see [`Shape::painted`]) for a gradient fill,
    /// evaluated once while the fill and the box stay; `None` for a flat fill, cheap to read.
    pub fn paint(&self) -> Option<Arc<Paint>> {
        self.fill.gradient?;
        let r = self.painted();
        let at = [r.x0, r.y0, r.w, r.h];
        let mut kept = self.paint.lock();
        if let Some(p) = kept.as_ref()
            && p.fill == self.fill
            && p.at == at
        {
            return Some(p.clone());
        }
        let values = (r.y0..r.y0 + r.h)
            .flat_map(|y| (r.x0..r.x0 + r.w).map(move |x| (x, y)))
            .map(|(x, y)| self.fill.paint_pixel(x, y))
            .collect();
        let p = Arc::new(Paint {
            fill: self.fill,
            at,
            values,
        });
        *kept = Some(p.clone());
        Some(p)
    }

    /// The shape's coverage of the pixels of its bounding box within the image, by its outline
    /// without a blur, rasterized the first time it is asked for.
    pub fn raster(&self) -> &Raster {
        self.raster
            .get_or_init(|| Arc::new(coverage_raster(&self.path(Vec2::ZERO), self.fill_rule())))
    }

    /// How much of each pixel of `window` the shape's outline covers, by itself and without its
    /// blur: what a shape clipped to it shows through.
    pub fn coverage(&self, window: Window) -> Vec<f32> {
        let mut out = vec![0.; window.width * window.height];
        if !window.intersects(self.min, self.max) {
            return out;
        }
        let raster = self.raster();
        if let Some(s) = raster.overlap(window) {
            for y in s.y..s.y + s.height {
                let from = (y - raster.y0) * raster.w + s.x - raster.x0;
                let to = (y - window.y) * window.width + s.x - window.x;
                out[to..to + s.width].copy_from_slice(&raster.values[from..from + s.width]);
            }
        }
        out
    }

    /// The shape for the rasterizer, moved by `-origin`.
    pub fn path(&self, origin: Vec2) -> Path {
        Path {
            subpaths: subpaths(&self.loops, origin),
            color: [
                self.fill.color.x,
                self.fill.color.y,
                self.fill.color.z,
                self.fill.opacity,
            ],
            gradient: self.fill.gradient.map(|mut g| {
                g.translate(-origin);
                g
            }),
            fill_rule: self.fill_rule(),
            blur: self.blur,
        }
    }

    /// How the loops fill: even-odd, except for a stroke, whose outline (see
    /// [`crate::stroke::Stroke::outline`]) winds twice where it overlaps itself.
    pub fn fill_rule(&self) -> FillRule {
        if self.stroke.is_some() {
            FillRule::NonZero
        } else {
            FillRule::EvenOdd
        }
    }
}

/// The scene as a trace builds it, shown to the host's progress callback as it grows
/// ([`crate::Progress::scene`]); [`Scene::model`] is its model.
#[derive(Clone, Debug)]
pub struct Scene {
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Bottom to top.
    pub(crate) shapes: Vec<Shape>,
    next_id: usize,
    /// Counts changes to the scene, see [`Shape::changed`].
    changes: u64,
    /// The whole scene rendered, with the count of changes it is current for: kept up to date
    /// while shapes are pushed on top, which is what fitting one shape after another does, and
    /// rendered again when something else changed it.
    canvas: Kept<Option<(u64, Arc<Vec<Vec3>>)>>,
    /// Objects drawn with the shapes but outside the scene's own render: a translucent layer's
    /// marks (see `translucency::vocabulary`), added once the scene is decided. A mark with a
    /// clip shows through the layer's shape, the first of the layer's group.
    pub(crate) marks: Vec<crate::model::Object>,
    /// The layer the marks belong to (see [`Shape::layer`]), whose group they close.
    pub(crate) marks_layer: Option<usize>,
    /// What an output of the decided scene may miss by, set by the trace.
    pub(crate) fidelity: Option<crate::model::Fidelity>,
    /// Shapes drawn as instances of one prototype (see [`crate::instancing`]); the shapes keep
    /// each instance's own geometry, the model draws the prototype once and places it.
    pub(crate) repeats: Vec<crate::instancing::Repeat>,
}

/// What shapes above a fitted shape do to it, per pixel of a window: the composite is
/// `color + transmittance * below`.
#[derive(Clone)]
pub struct Layer {
    pub color: Vec<Vec3>,
    pub transmittance: Vec<Vec3>,
}

impl Scene {
    /// The scene with no shape in a layer, so no layer group: the lab checks that the groups
    /// render as the flat shapes do.
    #[cfg(feature = "diagnostics")]
    #[doc(hidden)]
    #[must_use]
    pub fn flat(&self) -> Scene {
        let mut flat = self.clone();
        flat.shapes.iter_mut().for_each(|s| s.layer = None);
        flat.marks_layer = None;
        flat
    }

    /// A scene of one shape, the whole image as a rectangle in `fill`.
    pub(crate) fn with_background(width: usize, height: usize, fill: impl Into<Fill>) -> Self {
        let (w, h) = (width as f32, height as f32);
        let corners = [
            Vec2::ZERO,
            Vec2::new(w, 0.),
            Vec2::new(w, h),
            Vec2::new(0., h),
        ];
        let rectangle = (0..4)
            .map(|k| {
                let (a, b) = (corners[k], corners[(k + 1) % 4]);
                [a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.), b]
            })
            .collect();
        let background = Shape {
            id: 0,
            ..Shape::new(vec![rectangle], fill)
        };
        Self {
            width,
            height,
            shapes: vec![background],
            next_id: 1,
            changes: 0,
            canvas: Kept::new(None),
            marks: Vec::new(),
            marks_layer: None,
            fidelity: None,
            repeats: Vec::new(),
        }
    }

    /// Puts `shape` on top, returning its id.
    pub(crate) fn push(&mut self, shape: Shape) -> usize {
        let at = self.shapes.len();
        self.insert(at, shape)
    }

    /// Puts `shape` at `position` in the stack, below the shapes from there up, returning its id.
    pub(crate) fn insert(&mut self, position: usize, shape: Shape) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let before = self.changes;
        let change = self.change();
        self.shapes.insert(
            position,
            Shape {
                id,
                changed: change,
                fitted: change,
                ..shape
            },
        );
        // A shape on top keeps the canvas current by being drawn on it
        let mut canvas = self.canvas.lock();
        if let Some((current, pixels)) = canvas.as_mut()
            && *current == before
            && position + 1 == self.shapes.len()
        {
            let full = Window::full(self.width, self.height);
            let pixels: &mut Vec<Vec3> = Arc::make_mut(pixels);
            self.composite_onto(full, position, pixels);
            *current = change;
        }
        id
    }

    /// Gives a new id to each shape whose id a shape below it has, unless a clip names that id:
    /// scenes put together from traces of their own repeat ids.
    pub(crate) fn unique_ids(&mut self) {
        let clips: crate::HashSet<usize> = self.shapes.iter().filter_map(|s| s.clip).collect();
        let mut seen = crate::HashSet::default();
        for shape in &mut self.shapes {
            if !seen.insert(shape.id) && !clips.contains(&shape.id) {
                shape.id = self.next_id;
                self.next_id += 1;
            }
        }
    }

    /// The next count of changes.
    pub(crate) fn change(&mut self) -> u64 {
        self.changes += 1;
        self.changes
    }

    /// The count of changes so far.
    pub(crate) fn changes(&self) -> u64 {
        self.changes
    }

    /// Sets the fill of the shape at `position` in the stack (by place, since ids are not
    /// unique in every scene), counting a change where the fill looks different.
    pub(crate) fn set_fill_at(&mut self, position: usize, fill: Fill) {
        if self.shapes[position].fill.differs(&fill) {
            let change = self.change();
            self.shapes[position].changed = change;
        }
        self.shapes[position].fill = fill;
    }

    /// Whether the shape at `position` changed, or anything reaching into `window` around it did,
    /// since it was last fitted.
    pub(crate) fn is_stale(&self, position: usize, window: Window) -> bool {
        let fitted = self.shapes[position].fitted;
        self.shapes.iter().any(|s| {
            s.changed > fitted
                && (s.id == self.shapes[position].id
                    || window.intersects(s.paint_min(), s.paint_max()))
        })
    }

    /// The place in the stack of the shape with `id`.
    pub(crate) fn position(&self, id: usize) -> usize {
        self.shapes
            .iter()
            .position(|s| s.id == id)
            .expect("a shape of the scene")
    }

    pub(crate) fn get(&self, id: usize) -> &Shape {
        &self.shapes[self.position(id)]
    }

    /// Renders the pixels of `window` showing the shapes in `range` over `background`. The whole
    /// scene over white comes from the canvas.
    pub(crate) fn render(
        &self,
        window: Window,
        range: std::ops::Range<usize>,
        background: [f32; 3],
    ) -> Vec<Vec3> {
        if range == (0..self.shapes.len()) && background == [1.; 3] {
            let mut canvas = self.canvas.lock();
            if canvas
                .as_ref()
                .is_none_or(|(current, _)| *current != self.changes)
            {
                let full = Window::full(self.width, self.height);
                *canvas = Some((self.changes, Arc::new(self.render_shapes(full, range, background))));
            }
            let (_, pixels) = canvas.as_ref().expect("just rendered");
            return window.pixels(self.width).map(|i| pixels[i]).collect();
        }
        self.render_shapes(window, range, background)
    }

    fn render_shapes(
        &self,
        window: Window,
        range: std::ops::Range<usize>,
        background: [f32; 3],
    ) -> Vec<Vec3> {
        let mut pixels = vec![Vec3::from(background); window.width * window.height];
        for k in range {
            self.composite_onto(window, k, &mut pixels);
        }
        pixels
    }

    /// Composites the shape at place `k` onto `pixels`, a render of `window`, as
    /// [`Scene::render`] draws each shape of its range in turn.
    pub(crate) fn composite_onto(&self, window: Window, k: usize, pixels: &mut [Vec3]) {
        let shape = &self.shapes[k];
        if !window.intersects(shape.paint_min(), shape.paint_max()) {
            return;
        }
        // Straight from the raster over the rows and columns it shares with the window, its
        // clip read over those alone
        let raster = shape.painted();
        let Some(shared) = raster.overlap(window) else {
            return;
        };
        let clip = self.clip_mask(shared, k);
        let paint = shape.paint();
        // A flat fill unclipped composites row by row with one colour
        if paint.is_none() && clip.is_none() {
            let (c, a) = (shape.fill.color, shape.fill.opacity);
            for y in shared.y..shared.y + shared.height {
                let from = (y - raster.y0) * raster.w + shared.x - raster.x0;
                let to = (y - window.y) * window.width + shared.x - window.x;
                let row = &mut pixels[to..to + shared.width];
                for (p, &alpha) in row.iter_mut().zip(&raster.values[from..from + shared.width]) {
                    if alpha != 0. {
                        *p = *p + alpha * a * (c - *p);
                    }
                }
            }
            return;
        }
        let paint_at = |x: usize, y: usize| match &paint {
            Some(p) => p.at(x, y),
            None => (shape.fill.color, shape.fill.opacity),
        };
        for y in shared.y..shared.y + shared.height {
            for x in shared.x..shared.x + shared.width {
                let j = (y - window.y) * window.width + x - window.x;
                let alpha = raster.values[(y - raster.y0) * raster.w + x - raster.x0]
                    * clip.as_ref().map_or(1., |c| {
                        c[(y - shared.y) * shared.width + x - shared.x]
                    });
                // an uncovered pixel keeps what it shows; its paint is not evaluated
                if alpha == 0. {
                    continue;
                }
                let p = pixels[j];
                let (c, a) = paint_at(x, y);
                pixels[j] = p + alpha * a * (c - p);
            }
        }
    }

    /// How much of each pixel of `window` the shape at place `k` may show through its clip, or
    /// `None` when it is not clipped.
    pub(crate) fn clip_mask(&self, window: Window, k: usize) -> Option<Vec<f32>> {
        self.clip_of(window, self.shapes[k].clip)
    }

    /// [`Scene::clip_mask`] for a shape clipped to the shape with id `clip`, for shapes that are
    /// not in the scene yet; unclipped while the clip is out of the scene to be refitted.
    pub(crate) fn clip_of(&self, window: Window, clip: Option<usize>) -> Option<Vec<f32>> {
        let parent = self.shapes.iter().position(|s| Some(s.id) == clip)?;
        let mut mask = self.shapes[parent].coverage(window);
        if let Some(outer) = self.clip_mask(window, parent) {
            mask.iter_mut().zip(&outer).for_each(|(m, o)| *m *= o);
        }
        Some(mask)
    }

    /// The shapes from `from` up as a [`Layer`] over `window`, or `None` when none reach into it.
    /// Compositing is affine in the backdrop, so renders over black and white give the layer.
    pub(crate) fn layer(&self, window: Window, from: usize) -> Option<Layer> {
        let range = from..self.shapes.len();
        if !self.shapes[range.clone()]
            .iter()
            .any(|s| window.intersects(s.paint_min(), s.paint_max()))
        {
            return None;
        }
        // Over black, the composite is the layer's own color; what it lets through is the product
        // of what every shape leaves uncovered
        let n = window.width * window.height;
        let (mut color, mut transmittance) = (vec![Vec3::ZERO; n], vec![Vec3::ONE; n]);
        for k in range {
            let shape = &self.shapes[k];
            if !window.intersects(shape.paint_min(), shape.paint_max()) {
                continue;
            }
            let raster = shape.painted();
            let Some(shared) = raster.overlap(window) else {
                continue;
            };
            let clip = self.clip_mask(window, k);
            let kept = shape.paint();
            for y in shared.y..shared.y + shared.height {
                for x in shared.x..shared.x + shared.width {
                    let j = (y - window.y) * window.width + x - window.x;
                    // At the fill's opacity, as `render_shapes` composites it, so a translucent
                    // shape above a fitted one lets the fit see what lies under it
                    let covered = raster.values[(y - raster.y0) * raster.w + x - raster.x0]
                        * clip.as_ref().map_or(1., |c| c[j]);
                    if covered == 0. {
                        continue;
                    }
                    let (paint, a) = match &kept {
                        Some(p) => p.at(x, y),
                        None => (shape.fill.color, shape.fill.opacity),
                    };
                    let alpha = covered * a;
                    let c = color[j];
                    color[j] = c + alpha * (paint - c);
                    transmittance[j] *= 1. - alpha;
                }
            }
        }
        Some(Layer {
            color,
            transmittance,
        })
    }
}

/// The loops as the rasterizer's subpaths, moved by `-origin`.
fn subpaths(loops: &[Vec<Cubic>], origin: Vec2) -> Vec<Subpath> {
    loops
        .iter()
        .map(|cubics| Subpath {
            points: cubics
                .iter()
                .flat_map(|c| [c[0], c[1], c[2]])
                .map(|p| p - origin)
                .collect(),
        })
        .collect()
}

/// The loops as a path for the rasterizer's coverage, painted with nothing.
pub fn loops_path(loops: &[Vec<Cubic>]) -> Path {
    Path {
        subpaths: subpaths(loops, Vec2::ZERO),
        color: [0.; 4],
        gradient: None,
        fill_rule: FillRule::EvenOdd,
        blur: 0.,
    }
}

/// The coverage by `rule` of the pixels of the path's box within the image.
pub fn coverage_raster(path: &Path, rule: FillRule) -> Raster {
    let [raster] = coverage_rasters(path, [rule]);
    raster
}

/// [`coverage_raster`] by each of `rules`, the path rasterized once.
pub fn coverage_rasters<const N: usize>(path: &Path, rules: [FillRule; N]) -> [Raster; N] {
    let polys = crate::render::scene::flatten(path);
    let winding = crate::render::raster::rasterize(&polys, i32::MAX as u32, i32::MAX as u32);
    rules.map(|rule| Raster {
        x0: winding.x0.max(0) as usize,
        y0: winding.y0.max(0) as usize,
        w: winding.w,
        h: winding.h,
        values: winding
            .values
            .iter()
            .map(|&v| {
                crate::render::raster::fill_coverage(v, rule)
                    .0
                    .clamp(0., 1.)
            })
            .collect(),
    })
}

/// [`coverage_raster`] blurred by a Gaussian of `sigma` pixels, over the box it reaches.
pub fn blurred_raster(path: &Path, rule: FillRule, sigma: f32) -> Raster {
    // A frame larger than any image, so only the left and top edges cut the box
    let frame = 1 << 24;
    let polys = crate::render::scene::flatten(path);
    let b = crate::render::blur::Blurred::new(polys, rule, sigma, frame, frame);
    Raster {
        x0: b.x0,
        y0: b.y0,
        w: b.w,
        h: b.h,
        values: b.values,
    }
}

/// A shape's coverage of the pixels of its bounding box, rasterized once and kept with the shape
/// until its outline changes, since every window render and recoloring needs it.
#[derive(Clone, Debug, Default)]
pub struct Raster {
    /// Top left pixel of the box in the image.
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    /// Coverage per pixel of the box, row-major.
    pub values: Vec<f32>,
}

impl Raster {
    /// The pixels the box shares with `window`, if any.
    pub fn overlap(&self, window: Window) -> Option<Window> {
        let (x0, y0) = (self.x0.max(window.x), self.y0.max(window.y));
        let x1 = (self.x0 + self.w).min(window.x + window.width);
        let y1 = (self.y0 + self.h).min(window.y + window.height);
        (x1 > x0 && y1 > y0).then(|| Window {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        })
    }

    /// The coverage of image pixel `i`, zero outside the box.
    pub fn at(&self, i: usize, width: usize) -> f32 {
        let (x, y) = (i % width, i / width);
        if x < self.x0 || y < self.y0 || x >= self.x0 + self.w || y >= self.y0 + self.h {
            return 0.;
        }
        self.values[(y - self.y0) * self.w + x - self.x0]
    }

    /// Image indices and coverage of the pixels in the box.
    pub fn pixels(&self, width: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        (0..self.h).flat_map(move |y| {
            (0..self.w).map(move |x| {
                let i = (self.y0 + y) * width + self.x0 + x;
                (i, self.values[y * self.w + x])
            })
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
