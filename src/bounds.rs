//! The model clipped to the image's rectangle (see [`crate::TraceOptions::clip_to_image`]): a
//! shape the trace lets run past the picture's edge (a background, a primitive, an outline
//! completed under the frame) is cut at it. A filled object's loops are cut themselves, each
//! against the four sides in turn, the part outside a side replaced by its run along the side,
//! which keeps every winding inside; a stroke or a blurred object reaching out is clipped to
//! the rectangle instead, and an instance reaching out is drawn out into objects first.

// Panic ratchet: the lints this module has yet to meet, each removed as it is cleared
#![expect(
    clippy::arithmetic_side_effects,
    reason = "panic ratchet: remove a lint once the module is clean of it"
)]

use glam::DVec2;

use crate::model::{Clip, Curve, FillRule, Geometry, Item, Model, Object};

/// How far a point may lie past a side and still count as on it, in pixels.
const EPSILON: f64 = 1e-6;
/// How many standard deviations of a blur reach past its object's outline.
const BLUR_REACH: f64 = 3.;

/// Clips `model` to its `width` by `height` rectangle.
pub(crate) fn clip(model: &mut Model) {
    let size = DVec2::new(
        f64::from(u32::try_from(model.width).unwrap_or(u32::MAX)),
        f64::from(u32::try_from(model.height).unwrap_or(u32::MAX)),
    );
    let items = std::mem::take(&mut model.items);
    let mut frame = None;
    model.items = clipped(model, items, size, &mut frame);
}

/// `items` clipped to the rectangle of `size`, the rectangle's clip added to `model` as
/// `frame` where an object needs it.
fn clipped(
    model: &mut Model,
    items: Vec<Item>,
    size: DVec2,
    frame: &mut Option<usize>,
) -> Vec<Item> {
    let mut out = Vec::new();
    for item in items {
        match item {
            Item::Object(o) => {
                if let Some(o) = object(model, *o, size, frame) {
                    out.push(Item::Object(Box::new(o)));
                }
            }
            Item::Group { layer, items } => {
                let items = clipped(model, items, size, frame);
                if !items.is_empty() {
                    out.push(Item::Group { layer, items });
                }
            }
            Item::Instance(_) | Item::Repeat { .. } => {
                let one = Model {
                    items: Vec::from([item.clone()]),
                    ..model.clone()
                };
                let drawn = one.drawn();
                if drawn.iter().all(|(_, o)| within(o, size)) {
                    out.push(item);
                    continue;
                }
                let objects = drawn.into_iter().map(|(_, o)| Item::Object(Box::new(o)));
                out.extend(clipped(model, objects.collect(), size, frame));
            }
        }
    }
    out
}

/// `o` clipped to the rectangle of `size`, or none where nothing of it is left inside.
fn object(
    model: &mut Model,
    mut o: Object,
    size: DVec2,
    frame: &mut Option<usize>,
) -> Option<Object> {
    if within(&o, size) {
        return Some(o);
    }
    match &mut o.geometry {
        Geometry::Fill { loops, .. } if o.blur <= 0. => {
            let cut: Vec<Vec<Curve>> = loops.iter().filter_map(|l| cut_loop(l, size)).collect();
            if cut.is_empty() {
                return None;
            }
            *loops = cut;
        }
        _ => {
            let frame = *frame.get_or_insert_with(|| {
                model.clips.push(rectangle(size));
                model.clips.len() - 1
            });
            match o.clip {
                None => o.clip = Some(frame),
                Some(c) => {
                    // the object's clip, and so every object under it, inside the frame
                    let mut root = c;
                    while let Some(parent) = model.clips.get(root).and_then(|c| c.parent) {
                        root = parent;
                    }
                    if root != frame
                        && let Some(clip) = model.clips.get_mut(root)
                    {
                        clip.parent = Some(frame);
                    }
                }
            }
        }
    }
    Some(o)
}

/// Whether everything `o` draws lies within the rectangle of `size`: its control points, a
/// stroke's half width and a blur's reach around them.
fn within(o: &Object, size: DVec2) -> bool {
    let (points, reach): (Vec<DVec2>, f64) = match &o.geometry {
        Geometry::Fill { loops, .. } => (loops.iter().flatten().flatten().copied().collect(), 0.),
        Geometry::Stroke(s) => (
            s.centerline.iter().flatten().copied().collect(),
            // a miter reaches out to the limit of 4, a square cap by the diagonal
            s.width / 2.
                * if s.join == crate::model::Join::Miter {
                    4.
                } else {
                    std::f64::consts::SQRT_2
                },
        ),
    };
    let reach = reach + BLUR_REACH * o.blur.max(0.) + EPSILON;
    points
        .iter()
        .all(|p| p.x >= -reach && p.y >= -reach && p.x <= size.x + reach && p.y <= size.y + reach)
}

/// The rectangle of `size` as a clip.
fn rectangle(size: DVec2) -> Clip {
    let (a, b, c, d) = (
        DVec2::ZERO,
        DVec2::new(size.x, 0.),
        size,
        DVec2::new(0., size.y),
    );
    Clip {
        loops: Vec::from([Vec::from([line(a, b), line(b, c), line(c, d), line(d, a)])]),
        rule: FillRule::NonZero,
        parent: None,
    }
}

/// A side of the rectangle: where along `axis` it lies, and which way is inside.
#[derive(Clone, Copy)]
struct Side {
    axis: usize,
    at: f64,
    /// Whether inside is below `at` along the axis.
    below: bool,
}

impl Side {
    /// How far `p` lies past the side, negative inside.
    fn past(self, p: DVec2) -> f64 {
        let v = p[self.axis] - self.at;
        if self.below { v } else { -v }
    }

    /// `p` moved onto the side along its axis.
    fn onto(self, mut p: DVec2) -> DVec2 {
        p[self.axis] = self.at;
        p
    }

    /// Whether the cubic lies on the side.
    fn holds(self, c: &Curve) -> bool {
        c.iter().all(|p| self.past(*p).abs() <= EPSILON)
    }
}

/// The loop cut to the rectangle of `size`, none where nothing of it is left inside.
fn cut_loop(l: &[Curve], size: DVec2) -> Option<Vec<Curve>> {
    let sides = [
        Side {
            axis: 0,
            at: 0.,
            below: false,
        },
        Side {
            axis: 0,
            at: size.x,
            below: true,
        },
        Side {
            axis: 1,
            at: 0.,
            below: false,
        },
        Side {
            axis: 1,
            at: size.y,
            below: true,
        },
    ];
    let mut l = l.to_vec();
    for side in sides {
        if l.iter().flatten().all(|p| side.past(*p) <= EPSILON) {
            continue;
        }
        l = cut(&l, side)?;
    }
    (area(&l).abs() > EPSILON).then_some(l)
}

/// The loop with what lies past `side` replaced by its run along the side, none where
/// nothing is left inside.
fn cut(l: &[Curve], side: Side) -> Option<Vec<Curve>> {
    let mut pieces: Vec<(Curve, bool)> = Vec::new();
    for c in l {
        let mut ts = crossings(c, side);
        ts.push(1.);
        let mut t0 = 0.;
        for t1 in ts {
            if t1 - t0 <= EPSILON {
                continue;
            }
            let mut piece = segment(c, t0, t1);
            // the cut points on the side exactly, so the pieces meet the runs along it
            if t0 > 0. {
                piece[0] = side.onto(piece[0]);
            }
            if t1 < 1. {
                piece[3] = side.onto(piece[3]);
            }
            let inside = side.past(point(c, f64::midpoint(t0, t1))) <= EPSILON;
            pieces.push((piece, inside));
            t0 = t1;
        }
    }
    if !pieces.iter().any(|p| p.1) {
        return None;
    }
    let mut out: Vec<Curve> = Vec::new();
    for (piece, inside) in pieces {
        let piece = if inside {
            piece
        } else {
            line(side.onto(piece[0]), side.onto(piece[3]))
        };
        // runs along the side one after another are one run
        if let Some(last) = out.last_mut()
            && side.holds(last)
            && side.holds(&piece)
        {
            *last = line(last[0], piece[3]);
        } else {
            out.push(piece);
        }
    }
    // and the last run with the first, around the loop's start
    if out.len() > 1
        && let (Some(first), Some(last)) = (out.first(), out.last())
        && side.holds(first)
        && side.holds(last)
    {
        let joined = line(last[0], first[3]);
        out.pop();
        if let Some(first) = out.first_mut() {
            *first = joined;
        }
    }
    out.retain(|c| !side.holds(c) || c[0].distance(c[3]) > EPSILON);
    (!out.is_empty()).then_some(out)
}

/// Where in (0, 1) the cubic crosses the side, in order: the roots of its coordinate along
/// the side's axis less the side's place, by bisection between the coordinate's extremes.
fn crossings(c: &Curve, side: Side) -> Vec<f64> {
    let v = |t: f64| side.past(point(c, t));
    // the coordinate's derivative, a quadratic, is zero at its extremes
    let [p0, p1, p2, p3] = c.map(|p| p[side.axis]);
    let (a, b, d) = (p1 - p0, p2 - p1, p3 - p2);
    let (qa, qb, qc) = (a - 2. * b + d, 2. * (b - a), a);
    let mut stops = Vec::from([0.]);
    if qa.abs() > 1e-12 {
        let disc = qb * qb - 4. * qa * qc;
        if disc >= 0. {
            let r = disc.sqrt();
            stops.extend([(-qb - r) / (2. * qa), (-qb + r) / (2. * qa)]);
        }
    } else if qb.abs() > 1e-12 {
        stops.push(-qc / qb);
    }
    stops.push(1.);
    stops.retain(|t| (0. ..=1.).contains(t));
    stops.sort_by(f64::total_cmp);
    let mut roots = Vec::new();
    for w in stops.windows(2) {
        let &[mut lo, mut hi] = w else {
            continue;
        };
        let (vlo, vhi) = (v(lo), v(hi));
        if vlo.signum() == vhi.signum() || vlo == 0. || vhi == 0. {
            continue;
        }
        for _ in 0..60 {
            let mid = f64::midpoint(lo, hi);
            if v(mid).signum() == vlo.signum() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = f64::midpoint(lo, hi);
        if t > EPSILON && t < 1. - EPSILON {
            roots.push(t);
        }
    }
    roots
}

/// The cubic's point at `t`.
fn point(c: &Curve, t: f64) -> DVec2 {
    let s = 1. - t;
    c[0] * (s * s * s) + c[1] * (3. * s * s * t) + c[2] * (3. * s * t * t) + c[3] * (t * t * t)
}

/// The part of the cubic from `t0` to `t1`.
fn segment(c: &Curve, t0: f64, t1: f64) -> Curve {
    // the end at t1, then its start at t0 scaled into it
    let (_, right) = split(c, t0);
    let u = if t0 < 1. { (t1 - t0) / (1. - t0) } else { 1. };
    split(&right, u).0
}

/// The cubic split at `t` (de Casteljau).
fn split(c: &Curve, t: f64) -> (Curve, Curve) {
    let ab = c[0].lerp(c[1], t);
    let bc = c[1].lerp(c[2], t);
    let cd = c[2].lerp(c[3], t);
    let abc = ab.lerp(bc, t);
    let bcd = bc.lerp(cd, t);
    let m = abc.lerp(bcd, t);
    ([c[0], ab, abc, m], [m, bcd, cd, c[3]])
}

/// The straight cubic from `a` to `b`.
fn line(a: DVec2, b: DVec2) -> Curve {
    [a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.), b]
}

/// The loop's signed area, exact for cubics.
fn area(l: &[Curve]) -> f64 {
    l.iter()
        .map(|[p0, p1, p2, p3]| {
            (6. * p0.perp_dot(*p1)
                + 3. * p0.perp_dot(*p2)
                + p0.perp_dot(*p3)
                + 3. * p1.perp_dot(*p2)
                + 3. * p1.perp_dot(*p3)
                + 6. * p2.perp_dot(*p3))
                / 20.
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square loop from `min` to `max`, counterclockwise in image space.
    fn square(min: DVec2, max: DVec2) -> Vec<Curve> {
        let c = [min, DVec2::new(max.x, min.y), max, DVec2::new(min.x, max.y)];
        (0..4).map(|k| line(c[k], c[(k + 1) % 4])).collect()
    }

    fn bounds(l: &[Curve]) -> (DVec2, DVec2) {
        let points = l.iter().flatten();
        (
            points.clone().fold(DVec2::INFINITY, |a, p| a.min(*p)),
            points.fold(DVec2::NEG_INFINITY, |a, p| a.max(*p)),
        )
    }

    #[test]
    fn a_square_over_the_edge_is_cut_at_it() {
        let size = DVec2::new(100., 50.);
        let l = square(DVec2::new(-20., 10.), DVec2::new(40., 80.));
        let cut = cut_loop(&l, size).unwrap();
        let (min, max) = bounds(&cut);
        assert!(
            (min - DVec2::new(0., 10.)).abs().max_element() < 1e-9,
            "{min}"
        );
        assert!(
            (max - DVec2::new(40., 50.)).abs().max_element() < 1e-9,
            "{max}"
        );
        assert!(
            (area(&cut).abs() - 40. * 40.).abs() < 1e-6,
            "{}",
            area(&cut)
        );
        assert_eq!(cut.len(), 4, "the runs along the sides joined: {cut:?}");
    }

    #[test]
    fn a_background_larger_than_the_picture_becomes_its_rectangle() {
        let size = DVec2::new(64., 48.);
        let l = square(DVec2::splat(-10.), DVec2::new(80., 60.));
        let cut = cut_loop(&l, size).unwrap();
        assert!((area(&cut).abs() - 64. * 48.).abs() < 1e-6);
        assert_eq!(cut.len(), 4);
    }

    #[test]
    fn a_loop_outside_is_dropped() {
        let size = DVec2::new(64., 48.);
        assert!(cut_loop(&square(DVec2::new(70., 0.), DVec2::new(90., 20.)), size).is_none());
        assert!(cut_loop(&square(DVec2::new(-30., -30.), DVec2::new(-1., 60.)), size).is_none());
    }

    #[test]
    fn a_circle_over_a_corner_keeps_its_arc_inside() {
        // a circle of radius 20 about the corner, its quarter inside
        let k = 0.552_284_749_8 * 20.;
        let o = DVec2::ZERO;
        let p = |x: f64, y: f64| o + DVec2::new(x, y);
        let circle = vec![
            [p(20., 0.), p(20., k), p(k, 20.), p(0., 20.)],
            [p(0., 20.), p(-k, 20.), p(-20., k), p(-20., 0.)],
            [p(-20., 0.), p(-20., -k), p(-k, -20.), p(0., -20.)],
            [p(0., -20.), p(k, -20.), p(20., -k), p(20., 0.)],
        ];
        let cut = cut_loop(&circle, DVec2::new(100., 100.)).unwrap();
        let quarter = std::f64::consts::PI * 400. / 4.;
        assert!((area(&cut).abs() - quarter).abs() < 0.5, "{}", area(&cut));
        let (min, max) = bounds(&cut);
        assert!(min.min_element() >= -1e-9 && max.max_element() <= 20. + 1e-9);
    }

    #[test]
    fn a_curve_crossing_twice_is_cut_twice() {
        // a lobe bulging past the right side and back
        let c: Curve = [
            DVec2::new(80., 0.),
            DVec2::new(140., 10.),
            DVec2::new(140., 30.),
            DVec2::new(80., 40.),
        ];
        let side = Side {
            axis: 0,
            at: 100.,
            below: true,
        };
        let ts = crossings(&c, side);
        assert_eq!(ts.len(), 2, "{ts:?}");
        for t in ts {
            assert!((point(&c, t).x - 100.).abs() < 1e-9);
        }
    }

    #[test]
    fn the_model_is_cut_and_reaching_strokes_clipped() {
        use crate::model::{Cap, Join, Paint, StrokeGeometry};
        let fill = |l| Object {
            geometry: Geometry::Fill {
                loops: vec![l],
                rule: FillRule::NonZero,
            },
            paint: Paint::Solid(glam::DVec3::ONE),
            opacity: 1.,
            blur: 0.,
            clip: None,
        };
        let stroke = Object {
            geometry: Geometry::Stroke(StrokeGeometry {
                centerline: vec![line(DVec2::new(10., 10.), DVec2::new(10., 70.))],
                closed: false,
                width: 2.,
                cap: Cap::Butt,
                join: Join::Round,
                corners: vec![],
                dashes: None,
            }),
            ..fill(Vec::new())
        };
        let mut model = Model {
            width: 64,
            height: 48,
            items: vec![
                Item::Object(Box::new(fill(square(DVec2::splat(-5.), DVec2::splat(70.))))),
                Item::Object(Box::new(fill(square(DVec2::splat(80.), DVec2::splat(90.))))),
                Item::Object(Box::new(fill(square(DVec2::splat(10.), DVec2::splat(20.))))),
                Item::Object(Box::new(stroke)),
            ],
            clips: Vec::new(),
            definitions: Vec::new(),
            fidelity: None,
            clip: None,
        };
        clip(&mut model);
        let objects = model.objects();
        assert_eq!(objects.len(), 3, "the square outside dropped");
        let size = DVec2::new(64., 48.);
        assert!(
            objects
                .iter()
                .filter(|o| o.clip.is_none())
                .all(|o| within(o, size))
        );
        assert_eq!(objects[2].clip, Some(0), "the stroke reaching out clipped");
        assert_eq!(model.clips.len(), 1);
    }
}
