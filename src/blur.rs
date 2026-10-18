//! Blurred shapes: a shape whose boundary the picture shows as a soft ramp, not a step, is
//! drawn with a Gaussian blur over its coverage. The blur is an option of the shape like a
//! gradient, priced in segments and kept where the error it removes pays for it.

use crate::render::coverage_cost::CoverageCost;

use crate::refine::{MARGIN, Place, fit};
use glam::Vec3;

use crate::{Context, Scene, Shape, Window};

/// What a blur costs, in segments: one number an artist sets once on the shape, like one point
/// of its outline.
pub const BLUR_SEGMENTS: f64 = 1.;

/// The least blur tried, in drawing units ([`crate::translucency::raster::unit_of`]): a
/// quarter unit, about the anti-aliasing of an edge, which a narrower blur cannot be told from.
pub const LEAST: f64 = 0.25;

/// `shape`, to be drawn at `position` in `scene`, with the blur that lowers its error most, where
/// the plain error it removes pays for it at the segment's price (as every description of a
/// shape is priced), judged where the blur reaches. The blur is tried as it is and clipped to
/// `parent`, the shape it is drawn on, so it is soft inside and crisp at the parent's edge. σ is
/// searched by doubling from [`LEAST`] while the error falls, then fitted with the outline.
/// `below` is the image's render of the shapes below `position`.
pub fn fit_blur(
    ctx: &Context,
    scene: &Scene,
    shape: &Shape,
    position: usize,
    parent: Option<usize>,
    below: &[Vec3],
) -> Option<Shape> {
    if shape.stroke.is_some() || shape.blur > 0. {
        return None;
    }
    let unit = crate::translucency::raster::unit_of(ctx.width) as f32;
    // The place where `blurred` reaches, with the shape as given and as `blurred` is clipped,
    // their costs and the given shape's error; each is made with room for σ to double twice,
    // kept for the next σ of its clip that fits it, its given place shared by both clips
    type Places = std::rc::Rc<Made>;
    let made: std::cell::RefCell<Vec<(f32, Option<usize>, Places)>> = Default::default();
    let given_at: std::cell::RefCell<Vec<(f32, Given)>> = Default::default();
    let places = |blurred: &Shape| -> Option<Places> {
        let fits = |(room, clip, _): &&(f32, Option<usize>, Places)| {
            *clip == blurred.clip && *room >= blurred.blur
        };
        if let Some((_, _, p)) = made.borrow().iter().find(fits) {
            return Some(p.clone());
        }
        let room = 4. * blurred.blur;
        let shared = given_at
            .borrow()
            .iter()
            .find(|(r, _)| *r == room)
            .map(|g| g.1.clone());
        let given = match shared {
            Some(g) => g,
            None => {
                let wide = shape.clone().with_blur(room);
                let crop = |w: Window| w.pixels(ctx.width).map(|i| below[i]).collect();
                let place = Place::over(ctx, scene, &wide, position, MARGIN, crop)?;
                let cost = place.cost(ctx, shape.fill);
                let hard = cost.value(&shape.path(place.window.origin())) as f64;
                let g = std::rc::Rc::new((place, cost, hard));
                given_at.borrow_mut().push((room, g.clone()));
                g
            }
        };
        let own = (blurred.clip != shape.clip).then(|| {
            let own = given.0.clipped(scene, blurred.clip);
            let cost = own.cost(ctx, shape.fill);
            (own, cost)
        });
        let p = std::rc::Rc::new(Made { given, own });
        made.borrow_mut().push((room, blurred.clip, p.clone()));
        Some(p)
    };
    // The change in plain error `blurred` makes against the shape as given
    let delta = |blurred: &Shape| -> Option<f64> {
        let p = places(blurred)?;
        let origin = p.given.0.window.origin();
        Some(p.own_cost().value(&blurred.path(origin)) as f64 - p.given.2)
    };
    let clips = [shape.clip, parent.filter(|_| shape.clip.is_none())];
    let mut best: Option<(f64, Shape)> = None;
    for clip in clips
        .iter()
        .enumerate()
        .filter(|(k, c)| *k == 0 || c.is_some())
    {
        let candidate = Shape {
            clip: *clip.1,
            ..shape.clone()
        };
        let mut sigma = LEAST as f32 * unit;
        let mut here = (delta(&candidate.clone().with_blur(sigma))?, sigma);
        // A blur as wide as the shape washes it out, so the search stops there
        let size = (shape.max - shape.min).max_element();
        while 2. * sigma < size {
            sigma *= 2.;
            let Some(d) = delta(&candidate.clone().with_blur(sigma)) else {
                break;
            };
            if d >= here.0 {
                break;
            }
            here = (d, sigma);
        }
        if best.as_ref().is_none_or(|b| here.0 < b.0) {
            best = Some((here.0, candidate.with_blur(here.1)));
        }
    }
    let (best, mut blurred) = best?;
    let log = crate::diagnostics::log();
    if log {
        eprintln!(
            "blur: node {:?} at {:?}: best {best:.1} at sigma {:.2}, clipped {}, for {:.1}",
            shape.node,
            shape.min,
            blurred.blur,
            blurred.clip.is_some(),
            BLUR_SEGMENTS * ctx.segment_cost
        );
    }
    if best + BLUR_SEGMENTS * ctx.segment_cost >= 0. {
        return None;
    }
    // The outline and σ fitted together, in a window with room for σ to grow by half
    let room = places(&blurred.clone().with_blur(1.5 * blurred.blur))?;
    fit(
        room.own_cost(),
        &mut blurred,
        room.given.0.window.origin(),
        ctx.pull,
    );
    // Measured where the fitted blur reaches
    let described = delta(&blurred)?;
    if log {
        eprintln!(
            "blur: fitted sigma {:.2}, change {described:.1}",
            blurred.blur
        );
    }
    (described + BLUR_SEGMENTS * ctx.segment_cost < 0.).then_some(blurred)
}

/// The place of the shape as given, its cost and its error there.
type Given = std::rc::Rc<(Place, CoverageCost, f64)>;

/// A place [`fit_blur`] measures in: the shape as given (its place, cost and error) and, where
/// the blur is clipped otherwise, the clipped place and its cost.
struct Made {
    given: Given,
    own: Option<(Place, CoverageCost)>,
}

impl Made {
    fn own_cost(&self) -> &CoverageCost {
        self.own.as_ref().map_or(&self.given.1, |o| &o.1)
    }
}

/// Blurs each opaque filled shape of the decided `scene` where its blur pays (see
/// [`fit_blur`]), judged under the shapes above it; the parent it may be clipped to is the
/// nearest shape below whose box holds its box. Returns how many shapes were blurred.
pub fn blur_scene(ctx: &Context, scene: &mut Scene) -> usize {
    let mut blurred = 0;
    // the render of the shapes below the place at hand, each composited on as the pass passes it
    let full = Window::full(ctx.width, ctx.height);
    let mut below = vec![Vec3::ONE; ctx.width * ctx.height];
    if !scene.shapes.is_empty() {
        scene.composite_onto(full, 0, &mut below);
    }
    for position in 1..scene.shapes.len() {
        if crate::progress::poll() {
            break;
        }
        let s = &scene.shapes[position];
        if s.layer.is_some()
            || s.stroke.is_some()
            || s.primitive.is_some()
            || s.fill.gradient.is_some_and(|g| !g.is_opaque())
        {
            scene.composite_onto(full, position, &mut below);
            continue;
        }
        let holds = |p: &Shape| p.min.cmple(s.min).all() && p.max.cmpge(s.max).all();
        let parent = scene.shapes[1..position]
            .iter()
            .rev()
            .find(|p| holds(p))
            .map(|p| p.id)
            .filter(|&id| scene.shapes.iter().filter(|q| q.id == id).count() == 1);
        // Out of the scene while it is placed, so it is not above itself
        let shape = scene.shapes.remove(position);
        scene.change();
        let soft = fit_blur(ctx, scene, &shape, position, parent, &below);
        blurred += usize::from(soft.is_some());
        let change = scene.change();
        scene.shapes.insert(
            position,
            match soft {
                Some(soft) => Shape {
                    changed: change,
                    fitted: change,
                    ..soft
                },
                None => shape,
            },
        );
        scene.composite_onto(full, position, &mut below);
    }
    blurred
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use crate::{Scene, Shape, Window};

    fn square(a: Vec2, b: Vec2) -> Vec<Vec<crate::tracer::Cubic>> {
        let corners = [a, Vec2::new(b.x, a.y), b, Vec2::new(a.x, b.y)];
        vec![
            (0..4)
                .map(|k| {
                    let (p, q) = (corners[k], corners[(k + 1) % 4]);
                    [p, p.lerp(q, 1. / 3.), p.lerp(q, 2. / 3.), q]
                })
                .collect(),
        ]
    }

    /// A blurred shape clipped to the shape below is soft inside it and cut crisp at its
    /// edge, and the SVG writes the blur and the clip on the one path.
    #[test]
    fn a_clipped_blur_is_soft_inside_and_crisp_at_the_clip() {
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
        let pixels = scene.render(Window::full(24, 24), 0..3, [1.; 3]);
        let at = |x: usize, y: usize| pixels[y * 24 + x];
        // Outside the parent nothing of the blur shows
        assert!((at(2, 12) - Vec3::ONE).abs().max_element() < 1e-5);
        // Across the blurred edge inside the parent the red falls off smoothly
        let (a, b, c) = (at(9, 12).x, at(12, 12).x, at(15, 12).x);
        assert!(a > b && b > c && c > 0.01 && a < 1., "{a} {b} {c}");
    }
}
