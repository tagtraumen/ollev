//! Finding shapes alike: each outline is resampled about its centroid, candidates are paired by
//! their perimeter over size and their fill, and confirmed by aligning the outlines.

use crate::tracer::fit::eval;
use glam::Vec2;

use super::{ALIGN, Transform};
use crate::translucency::raster::NOISE;
use crate::{Scene, Shape};

/// Points an outline's loop is resampled at for aligning.
const SAMPLES: usize = 64;
/// Phases between neighbouring samples tried on either side of the best shift.
const PHASES: i32 = 4;
/// How far apart, per channel, two fills may be to be drawn in one (a level of 8 bits and its
/// anti-aliasing, twice).
pub(super) const COLOUR_MERGE: f32 = 2. * NOISE as f32;

/// A shape's outline about its centroid, for aligning it to others.
pub(super) struct Outline {
    pub id: usize,
    pub centroid: Vec2,
    /// The outer loop's perimeter and the square root of its area.
    perimeter: f32,
    size: f32,
    /// Each loop resampled by arc length, counter-clockwise, largest first.
    samples: Vec<Vec<Vec2>>,
    /// Each loop flattened, in the same order.
    dense: Vec<Vec<Vec2>>,
}

impl Outline {
    pub fn of(shape: &Shape) -> Option<Self> {
        let mut loops: Vec<(f32, Vec<Vec2>)> = shape
            .loops
            .iter()
            .filter(|l| !l.is_empty())
            .map(|l| {
                let steps = if l.len() > SAMPLES { 2 } else { 8 };
                let mut points = Vec::with_capacity(l.len() * steps);
                for c in l {
                    for i in 0..steps {
                        points.push(eval(c, i as f32 / steps as f32));
                    }
                }
                let area = signed_area(&points);
                if area < 0. {
                    points.reverse();
                }
                (area.abs(), points)
            })
            .collect();
        loops.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (area, outer) = loops.first()?;
        let size = area.sqrt();
        if size < 2. * ALIGN {
            return None;
        }
        let centroid = polygon_centroid(outer);
        let dense: Vec<Vec<Vec2>> = loops
            .iter()
            .map(|(_, l)| l.iter().map(|p| *p - centroid).collect())
            .collect();
        let samples = dense.iter().map(|l| resample(l, SAMPLES)).collect();
        // what the nearest points are searched on: the loops within a tenth of the alignment
        let dense = dense.iter().map(|l| simplified(l, ALIGN / 10.)).collect();
        Some(Self {
            id: shape.id,
            centroid,
            perimeter: perimeter(outer),
            size,
            samples,
            dense,
        })
    }

    /// Whether `o`'s perimeter over size could be this one's with every point moved by up to
    /// [`ALIGN`]: the perimeter moves by up to 2 pi times that, the area by the perimeter.
    fn could_match(&self, o: &Self) -> bool {
        if self.samples.len() != o.samples.len() {
            return false;
        }
        let ratio = |p: f32, s: f32| p / s;
        let r = ratio(self.perimeter, self.size);
        let slack = r
            * (std::f32::consts::TAU * ALIGN / self.perimeter
                + self.perimeter * ALIGN / (2. * self.size * self.size));
        (ratio(o.perimeter, o.size) - r).abs() <= slack
    }
}

/// The transform placing `a`'s outline (about its centroid) onto `b`'s in the image, if every
/// point of each lies within [`ALIGN`] of the other; a rotation and scale that move no point by
/// half of that are left out.
pub(super) fn align(a: &Outline, b: &Outline) -> Option<Transform> {
    if !a.could_match(b) {
        return None;
    }
    let (p, q) = (&a.samples[0], &b.samples[0]);
    let n = p.len();
    let pp: f32 = p.iter().map(|v| v.length_squared()).sum();
    let qq: f32 = q.iter().map(|v| v.length_squared()).sum();
    // Each cyclic shift's rotation and scale and the residual they leave
    let shifts: Vec<(f32, f32, usize)> = (0..n)
        .map(|k| {
            let (mut dot, mut cross) = (0f32, 0f32);
            for i in 0..n {
                let (u, v) = (p[i], q[(i + k) % n]);
                dot += u.dot(v);
                cross += u.perp_dot(v);
            }
            let fit = (dot * dot + cross * cross) / pp;
            (qq - fit, cross.atan2(dot), k)
        })
        .collect();
    // Of the shifts best among their neighbours and within half an alignment step of the best,
    // the least turn: a square aligns onto a square at every quarter turn
    let least = shifts.iter().map(|s| s.0).fold(f32::INFINITY, f32::min);
    let slack = n as f32 * (ALIGN / 2.) * (ALIGN / 2.);
    let local = |s: &(f32, f32, usize)| {
        s.0 <= shifts[(s.2 + 1) % n].0 && s.0 <= shifts[(s.2 + n - 1) % n].0
    };
    let &(_, _, k) = shifts
        .iter()
        .filter(|s| s.0 <= least + slack && local(s))
        .min_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))?;
    // The samples' phase refined between that shift and its neighbours, in eighths of a step,
    // so the correspondence does not bias the turn
    let (mut best, mut angle, mut scale, mut offset) = (f32::INFINITY, 0., 1., Vec2::ZERO);
    for e in -PHASES..=PHASES {
        let f = e as f32 / (2 * PHASES) as f32;
        let at = |i: usize| {
            let j = (i + k + n) % n;
            let next = if f >= 0. {
                (j + 1) % n
            } else {
                (j + n - 1) % n
            };
            q[j].lerp(q[next], f.abs())
        };
        let targets: Vec<Vec2> = (0..n).map(at).collect();
        let (a, s, o) = procrustes(p, &targets);
        let r = Vec2::from_angle(a);
        let residual: f32 = p
            .iter()
            .zip(&targets)
            .map(|(u, v)| (o + s * r.rotate(*u)).distance_squared(*v))
            .sum();
        if residual < best {
            (best, angle, scale, offset) = (residual, a, s, o);
        }
    }
    // Instances are the prototype's size: a shape alike at another size is another drawing
    let reach = p.iter().map(|v| v.length()).fold(0., f32::max);
    if (scale - 1.).abs() * reach > ALIGN / 2. {
        return None;
    }
    let mut t = Transform {
        scale: 1.,
        angle,
        offset: b.centroid + offset,
    };
    if angle.abs() * reach <= ALIGN / 2. {
        t = Transform::translation(b.centroid + offset);
    }
    let local = |v: Vec2| t.apply(v) - b.centroid;
    for (l, (samples, dense)) in a.samples.iter().zip(&a.dense).enumerate() {
        let moved: Vec<Vec2> = dense.iter().map(|v| local(*v)).collect();
        let far = |from: &[Vec2], onto: &[Vec2]| {
            from.iter().any(|v| nearest(onto, *v).distance(*v) > ALIGN)
        };
        let samples_moved: Vec<Vec2> = samples.iter().map(|v| local(*v)).collect();
        if far(&samples_moved, &b.dense[l]) || far(&b.samples[l], &moved) {
            return None;
        }
    }
    Some(t)
}

/// Whether two shapes could be drawn in one fill and pen.
fn paint_matches(a: &Shape, b: &Shape) -> bool {
    let colour = (a.fill.color - b.fill.color).abs().max_element() <= COLOUR_MERGE;
    let opacity = (a.fill.opacity - b.fill.opacity).abs() <= 0.5 / 255. && a.blur == b.blur;
    let pen = match (&a.stroke, &b.stroke) {
        (None, None) => true,
        (Some(s), Some(t)) => {
            s.cap == t.cap
                && s.join == t.join
                && s.closed == t.closed
                && s.dashes.is_some() == t.dashes.is_some()
        }
        _ => false,
    };
    colour && opacity && pen
}

/// Whether a shape may be an instance: of the opaque scene, unclipped, flat filled.
pub(super) fn eligible(shape: &Shape) -> bool {
    shape.clip.is_none() && shape.layer.is_none() && shape.fill.gradient.is_none()
}

/// Groups of at least two shapes alike, each led by its first shape in the stack; a shape joins
/// the first leader its paint and outline match moved, or else turned. Shapes in `taken` are
/// left out.
pub(super) fn clusters(scene: &Scene, taken: &crate::HashSet<usize>) -> Vec<Vec<Outline>> {
    let mut count: crate::HashMap<usize, usize> = crate::HashMap::default();
    for s in &scene.shapes {
        *count.entry(s.id).or_default() += 1;
    }
    let mut groups: Vec<Vec<Outline>> = Vec::new();
    // each group's leader in the stack, and the groups by the leader's colour cell: a paint
    // within COLOUR_MERGE per channel lies in a neighbouring cell (cells a little wider, past
    // the division's rounding), so only those are tried, in the groups' order
    let mut leaders: Vec<&Shape> = Vec::new();
    let cell = |s: &Shape| (s.fill.color / (1.001 * COLOUR_MERGE)).floor().as_ivec3();
    let mut by_cell: crate::HashMap<glam::IVec3, Vec<usize>> = crate::HashMap::default();
    let mut near: Vec<usize> = Vec::new();
    for shape in scene
        .shapes
        .iter()
        .skip(1)
        .filter(|s| eligible(s) && count[&s.id] == 1 && !taken.contains(&s.id))
    {
        let Some(outline) = Outline::of(shape) else {
            continue;
        };
        let mut turned = None;
        let mut moved = None;
        let c = cell(shape);
        near.clear();
        for d in (-1..=1).flat_map(|x| (-1..=1).flat_map(move |y| (-1..=1).map(move |z| (x, y, z))))
        {
            if let Some(ks) = by_cell.get(&(c + glam::IVec3::from(d))) {
                near.extend(ks);
            }
        }
        near.sort_unstable();
        for &k in &near {
            let leader = &groups[k][0];
            if !paint_matches(leaders[k], shape) || !leader.could_match(&outline) {
                continue;
            }
            match align(leader, &outline) {
                Some(t) if t.is_translation() => {
                    moved = Some(k);
                    break;
                }
                Some(_) if turned.is_none() => turned = Some(k),
                _ => {}
            }
        }
        match moved.or(turned) {
            Some(k) => groups[k].push(outline),
            None => {
                by_cell.entry(c).or_default().push(groups.len());
                leaders.push(shape);
                groups.push(vec![outline]);
            }
        }
    }
    groups.retain(|g| g.len() >= 2);
    groups
}

/// The point of the closed polyline `line` nearest to `v`.
fn nearest(line: &[Vec2], v: Vec2) -> Vec2 {
    let n = line.len();
    let mut best = (f32::INFINITY, line[0]);
    for i in 0..n {
        let (a, b) = (line[i], line[(i + 1) % n]);
        let d = b - a;
        let t = if d.length_squared() > 0. {
            ((v - a).dot(d) / d.length_squared()).clamp(0., 1.)
        } else {
            0.
        };
        let p = a + t * d;
        let dist = p.distance_squared(v);
        if dist < best.0 {
            best = (dist, p);
        }
    }
    best.1
}

/// The rotation, scale and translation taking the points `p` nearest onto `q`, least squares.
fn procrustes(p: &[Vec2], q: &[Vec2]) -> (f32, f32, Vec2) {
    let n = p.len() as f32;
    let (cp, cq) = (p.iter().sum::<Vec2>() / n, q.iter().sum::<Vec2>() / n);
    let (mut dot, mut cross, mut pp) = (0f32, 0f32, 0f32);
    for (u, v) in p.iter().zip(q) {
        let (u, v) = (*u - cp, *v - cq);
        dot += u.dot(v);
        cross += u.perp_dot(v);
        pp += u.length_squared();
    }
    let angle = cross.atan2(dot);
    let scale = (dot * dot + cross * cross).sqrt() / pp.max(1e-9);
    let offset = cq - scale * Vec2::from_angle(angle).rotate(cp);
    (angle, scale, offset)
}

fn signed_area(points: &[Vec2]) -> f32 {
    let n = points.len();
    (0..n)
        .map(|i| points[i].perp_dot(points[(i + 1) % n]))
        .sum::<f32>()
        / 2.
}

fn perimeter(points: &[Vec2]) -> f32 {
    let n = points.len();
    (0..n)
        .map(|i| points[i].distance(points[(i + 1) % n]))
        .sum()
}

fn polygon_centroid(points: &[Vec2]) -> Vec2 {
    let n = points.len();
    let (mut c, mut a) = (Vec2::ZERO, 0f32);
    for i in 0..n {
        let (p, q) = (points[i], points[(i + 1) % n]);
        let w = p.perp_dot(q);
        a += w;
        c += (p + q) * w;
    }
    if a.abs() < 1e-9 {
        return points.iter().sum::<Vec2>() / n as f32;
    }
    c / (3. * a)
}

/// The closed polyline with the points dropped that lie within `tolerance` of the chord
/// between those kept (Douglas and Peucker's), from its first point and its farthest.
fn simplified(points: &[Vec2], tolerance: f32) -> Vec<Vec2> {
    let n = points.len();
    if n < 4 {
        return points.to_vec();
    }
    let far = (1..n)
        .max_by(|&i, &j| {
            points[i]
                .distance_squared(points[0])
                .total_cmp(&points[j].distance_squared(points[0]))
        })
        .unwrap_or(n / 2);
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[far] = true;
    let mut spans = vec![(0, far), (far, n)];
    while let Some((a, b)) = spans.pop() {
        let (p, q) = (points[a], points[b % n]);
        let d = q - p;
        let len = d.length();
        let dist = |v: Vec2| {
            if len > 0. {
                (v - p).perp_dot(d).abs() / len
            } else {
                v.distance(p)
            }
        };
        if let Some(k) = (a + 1..b).max_by(|&i, &j| dist(points[i]).total_cmp(&dist(points[j])))
            && dist(points[k]) > tolerance
        {
            keep[k] = true;
            spans.push((a, k));
            spans.push((k, b));
        }
    }
    points
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(p, _)| *p)
        .collect()
}

/// `m` points spaced evenly by arc length along the closed polyline, from its first point.
fn resample(points: &[Vec2], m: usize) -> Vec<Vec2> {
    let n = points.len();
    let total = perimeter(points);
    let mut out = Vec::with_capacity(m);
    let (mut i, mut walked) = (0, 0f32);
    for k in 0..m {
        let at = total * k as f32 / m as f32;
        loop {
            let (a, b) = (points[i % n], points[(i + 1) % n]);
            let len = a.distance(b);
            if walked + len >= at || i >= 2 * n {
                let t = if len > 0. { (at - walked) / len } else { 0. };
                out.push(a.lerp(b, t.clamp(0., 1.)));
                break;
            }
            walked += len;
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(c: Vec2, h: f32, angle: f32) -> Shape {
        let corners = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
            .map(|(x, y)| c + Vec2::from_angle(angle).rotate(Vec2::new(x, y) * h));
        let l = (0..4)
            .map(|k| {
                let (a, b) = (corners[k], corners[(k + 1) % 4]);
                [a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.), b]
            })
            .collect();
        Shape::new(vec![l], glam::Vec3::ONE)
    }

    /// A square turned and moved aligns onto the other by that rotation and move; a rectangle
    /// does not align onto a square.
    #[test]
    fn a_turned_square_aligns() {
        let a = Outline::of(&square(Vec2::new(20., 20.), 8., 0.)).unwrap();
        let b = Outline::of(&square(Vec2::new(60., 30.), 8., 0.3)).unwrap();
        let t = align(&a, &b).expect("the squares align");
        assert!((t.offset - Vec2::new(60., 30.)).length() < 0.05, "{t:?}");
        let turn = t.angle.rem_euclid(std::f32::consts::FRAC_PI_2);
        assert!((turn - 0.3).abs() < 0.01, "{t:?}");
        assert!((t.scale - 1.).abs() < 0.01, "{t:?}");
        let moved = Outline::of(&square(Vec2::new(90., 30.), 8., 0.)).unwrap();
        assert!(align(&a, &moved).unwrap().is_translation());
        let mut long = square(Vec2::new(60., 60.), 8., 0.);
        long.loops[0].iter_mut().flatten().for_each(|p| p.x *= 1.5);
        long.update_bounds();
        assert!(align(&a, &Outline::of(&long).unwrap()).is_none());
    }
}
