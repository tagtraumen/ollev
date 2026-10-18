//! What the tracer optimizes: the image, how to measure error against it, and what a segment
//! costs in units of that error.

use glam::{DVec3, Vec2, Vec3};

use crate::hierarchy::Stats;

#[derive(Clone)]
pub struct Context {
    pub width: usize,
    pub height: usize,
    /// The image, RGB in `0..=1`, row-major, shared by the copies made for other weights.
    pub target: std::sync::Arc<Vec<Vec3>>,
    /// The picture with anti-aliased pixels unmixed to the side they mostly belong to, in RGB,
    /// for the translucency stage's materials: no chain of blends links two materials (as the
    /// skin and sky between niabot's legs). None where built without the analysis.
    pub unmixed: Option<std::sync::Arc<Vec<Vec3>>>,
    /// What a path costs on top of its segments, in units of the error.
    pub path_cost: f64,
    /// The squared color error a cubic segment has to save to be worth drawing. It grows with
    /// the image area, so the same picture at another size gets the same shapes, and falls with
    /// the detail level.
    pub segment_cost: f64,
    /// How much the fit pulls every handle's length toward a third of its chord, per squared
    /// pixel of difference; a small pull resolves where the image does not care, such as handles
    /// sliding along a straight edge, see [`crate::refine::HANDLE_PULL`].
    pub pull: f32,
    /// How much each pixel's error counts, or `None` for all fully. A pixel at zero is left to
    /// shapes drawn later, like detail a coarse shape should not bend towards.
    pub weights: Option<Weights>,
    /// The size the costs and tolerances are scaled for, the square root of the area: this
    /// image's, or the whole image's for a window traced on its own, so it gets the same shapes.
    pub frame: f32,
    /// How far each pixel may deviate before it counts as error, a colour distance (see
    /// [`crate::uncertainty`]), or `None` for the plain squared error.
    pub tolerance: Option<std::sync::Arc<Vec<f32>>>,
    /// Under a tolerance, the prices of a segment and of a path for whether a planned shape is
    /// drawn at all, apart from [`Context::segment_cost`] which prices describing what is drawn
    /// (see [`crate::uncertainty::apply`]). None prices both alike.
    pub accept: Option<(f64, f64)>,
    /// Per pixel, whether nothing in the image says what it shows (a beneath under the opaque
    /// parts above a layer, see [`crate::translucency::completion`]). Every fit leaves these out
    /// of its error, so an outline across them follows its evidence on either side.
    pub free: Option<std::sync::Arc<Vec<bool>>>,
}

/// Segment cost at detail 1 for a 128 x 128 image.
const SEGMENT_COST: f64 = 0.5;

impl Context {
    /// A context for the RGB image `rgb` at the detail level `detail`.
    pub fn new(rgb: &[u8], width: usize, height: usize, detail: f64) -> Self {
        let target = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) / 255.)
            .collect::<Vec<_>>()
            .into();
        let segment_cost =
            SEGMENT_COST * (width * height) as f64 / (128. * 128.) / (detail * detail);
        Self {
            width,
            height,
            target,
            path_cost: 0.,
            weights: None,
            pull: 0.,
            segment_cost,
            frame: ((width * height) as f32).sqrt(),
            tolerance: None,
            accept: None,
            unmixed: None,
            free: None,
        }
    }

    /// What drawing the pixels of `stats` in their mean saves over showing `under`.
    pub fn saving(&self, stats: &Stats, under: DVec3) -> f64 {
        stats.sse_about(under) - stats.sse()
    }

    /// The error of `rendered`, the pixels of `window` in RGB: the one place that decides what
    /// counts as error.
    pub fn error(&self, window: Window, rendered: &[Vec3]) -> f32 {
        window
            .pixels(self.width)
            .zip(rendered)
            .map(|(i, r)| self.pixel_error(i, *r))
            .sum()
    }

    /// The error of pixel `i` showing `rendered` (RGB): its weight times the squared residual
    /// beyond its tolerance.
    pub fn pixel_error(&self, i: usize, rendered: Vec3) -> f32 {
        let (w, target) = self.weighted_target(i);
        w * crate::uncertainty::excess(
            self.distance_squared(rendered, target),
            self.tolerance_at(i),
        )
    }

    /// The price of a segment for whether a planned shape is drawn at all.
    pub fn accept_cost(&self) -> f64 {
        self.accept.map_or(self.segment_cost, |a| a.0)
    }

    /// The price of a path for whether a planned shape is drawn at all.
    pub fn accept_path_cost(&self) -> f64 {
        self.accept.map_or(self.path_cost, |a| a.1)
    }

    /// How far pixel `i` may deviate before it counts as error, 0 without a tolerance.
    pub fn tolerance_at(&self, i: usize) -> f32 {
        self.tolerance.as_ref().map_or(0., |t| t[i])
    }

    /// The squared difference between two RGB colors.
    pub fn distance_squared(&self, a: Vec3, b: Vec3) -> f32 {
        a.distance_squared(b)
    }

    /// A color from the hierarchy's statistics as a shape's.
    pub fn rgb(&self, color: DVec3) -> Vec3 {
        color.as_vec3()
    }

    /// Whether pixel `i` is left out of the error, so a shape is free to show anything there.
    pub fn is_free(&self, i: usize) -> bool {
        self.weights.as_ref().is_some_and(|w| {
            w.masked_color.is_none()
                && w.window
                    .index(i, self.width)
                    .is_some_and(|k| w.values[k] == 0.)
        })
    }

    /// How much pixel `i` counts and what it should show.
    pub fn weighted_target(&self, i: usize) -> (f32, Vec3) {
        let target = self.target[i];
        let Some(weights) = &self.weights else {
            return (1., target);
        };
        let Some(k) = weights.window.index(i, self.width) else {
            return (1., target);
        };
        match weights.masked_color {
            Some(color) if weights.values[k] == 0. => (1., color),
            _ => (weights.values[k], target),
        }
    }
}

/// Weights for the pixels of a window, and one everywhere else.
#[derive(Clone, Debug)]
pub struct Weights {
    pub window: Window,
    pub values: Vec<f32>,
    /// What the pixels at weight zero should show instead of being left free, such as a parent's
    /// own color where children clipped to it will show: the parent must cover them.
    pub masked_color: Option<Vec3>,
}

impl Weights {
    /// Zero on the pixels of `window` that are `masked` and their four neighbors, one elsewhere.
    pub fn masking(window: Window, image_width: usize, masked: impl Fn(usize) -> bool) -> Self {
        let (w, h) = (window.width, window.height);
        let inside: Vec<bool> = window.pixels(image_width).map(masked).collect();
        let values = (0..w * h)
            .map(|k| {
                let (x, y) = (k % w, k / w);
                let near = inside[k]
                    || (x > 0 && inside[k - 1])
                    || (x + 1 < w && inside[k + 1])
                    || (y > 0 && inside[k - w])
                    || (y + 1 < h && inside[k + w]);
                if near { 0. } else { 1. }
            })
            .collect();
        Self {
            window,
            values,
            masked_color: None,
        }
    }
}

/// A rectangle of whole pixels within the image.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Window {
    /// The pixels touched by the box from `min` to `max`, grown by `margin`, clipped to the image.
    pub fn around(min: Vec2, max: Vec2, margin: f32, width: usize, height: usize) -> Self {
        let x0 = ((min.x - margin).floor().max(0.) as usize).min(width);
        let y0 = ((min.y - margin).floor().max(0.) as usize).min(height);
        let x1 = ((max.x + margin).ceil().max(0.) as usize).clamp(x0, width);
        let y1 = ((max.y + margin).ceil().max(0.) as usize).clamp(y0, height);
        Self {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        }
    }

    pub fn origin(&self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    pub fn intersects(&self, min: Vec2, max: Vec2) -> bool {
        min.x < (self.x + self.width) as f32
            && min.y < (self.y + self.height) as f32
            && max.x > self.x as f32
            && max.y > self.y as f32
    }

    /// The whole image as a window.
    pub fn full(width: usize, height: usize) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    /// The position of image pixel `i` in the window, if it lies within.
    pub fn index(&self, i: usize, image_width: usize) -> Option<usize> {
        let (x, y) = (i % image_width, i / image_width);
        (x >= self.x && y >= self.y && x < self.x + self.width && y < self.y + self.height)
            .then(|| (y - self.y) * self.width + x - self.x)
    }

    /// Image indices of the window's pixels, row by row.
    pub fn pixels(&self, image_width: usize) -> impl Iterator<Item = usize> {
        let w = *self;
        (w.y..w.y + w.height)
            .flat_map(move |y| (w.x..w.x + w.width).map(move |x| y * image_width + x))
    }
}
