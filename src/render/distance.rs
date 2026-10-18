//! The curve of a path gradient: a chain of cubic Béziers, closed or open, and the signed
//! distance from it by a polyline the chain is flattened to once. A closed chain is negative
//! inside (even-odd). An open one goes on along its end tangents, and is negative on the side
//! closed by its rays to a box and the box walked clockwise from its end back to its start.

use glam::Vec2;

/// The most cubics a path gradient's curve has.
pub const MAX_CUBICS: usize = 32;
/// The most points of the polyline the curve is measured on.
const MAX_POINTS: usize = 192;
/// The most a chord of the polyline strays from the curve, in pixels: a tenth, below what
/// the anti-aliasing of an edge shows.
const SAGITTA: f32 = 0.1;

/// A chain of up to [`MAX_CUBICS`] cubics, each ending where the next starts, and the
/// polyline it is flattened to; an open chain also its rays and the box corners closing it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contour {
    cubics: u8,
    /// Control points `[p0, c1, c2, p1, c1, c2, p2, ...]`, the last end at the first start.
    control: [Vec2; 3 * MAX_CUBICS + 1],
    points: u8,
    polyline: [Vec2; MAX_POINTS],
    closed: bool,
    /// An open chain's closure: from its end, the end ray's exit from the box, the corners
    /// walked clockwise, the start ray's exit, back to its start.
    closure: [Vec2; 7],
    closure_len: u8,
}

fn eval(c: &[Vec2; 4], t: f32) -> Vec2 {
    let u = 1. - t;
    c[0] * (u * u * u) + c[1] * (3. * u * u * t) + c[2] * (3. * u * t * t) + c[3] * (t * t * t)
}

impl Contour {
    /// The closed chain of `cubics`, at most [`MAX_CUBICS`]; `None` when empty or longer.
    pub fn new(cubics: &[[Vec2; 4]]) -> Option<Self> {
        Self::chain(cubics, true)
    }

    /// The open chain of `cubics`, its sides closed by the box from `min` to `max`, which
    /// holds the chain.
    pub fn open(cubics: &[[Vec2; 4]], min: Vec2, max: Vec2) -> Option<Self> {
        let mut c = Self::chain(cubics, false)?;
        let line = c.polyline();
        let (start, end) = (line[0], line[line.len() - 1]);
        let tangent = |a: Vec2, b: Vec2| (b - a).normalize_or(Vec2::X);
        let (first, last) = (&cubics[0], &cubics[cubics.len() - 1]);
        let t0 = if first[1] != first[0] {
            tangent(first[1], first[0])
        } else {
            tangent(first[3], first[0])
        };
        let t1 = if last[2] != last[3] {
            tangent(last[2], last[3])
        } else {
            tangent(last[0], last[3])
        };
        let (lo, hi) = (min.min(start.min(end)) - 1., max.max(start.max(end)) + 1.);
        // Where a ray from inside the box leaves it, and the place of that along the boundary,
        // clockwise from the top left corner
        let exit = |p: Vec2, d: Vec2| {
            let tx = if d.x > 0. {
                (hi.x - p.x) / d.x
            } else if d.x < 0. {
                (lo.x - p.x) / d.x
            } else {
                f32::INFINITY
            };
            let ty = if d.y > 0. {
                (hi.y - p.y) / d.y
            } else if d.y < 0. {
                (lo.y - p.y) / d.y
            } else {
                f32::INFINITY
            };
            (p + d * tx.min(ty)).clamp(lo, hi)
        };
        let (w, h) = (hi.x - lo.x, hi.y - lo.y);
        let place = |q: Vec2| {
            if (q.y - lo.y).abs() < 1e-3 {
                q.x - lo.x
            } else if (q.x - hi.x).abs() < 1e-3 {
                w + q.y - lo.y
            } else if (q.y - hi.y).abs() < 1e-3 {
                w + h + hi.x - q.x
            } else {
                2. * w + h + hi.y - q.y
            }
        };
        let (out_end, out_start) = (exit(end, t1), exit(start, t0));
        let corners = [
            (w, Vec2::new(hi.x, lo.y)),
            (w + h, hi),
            (2. * w + h, Vec2::new(lo.x, hi.y)),
            (2. * (w + h), lo),
        ];
        let (a, mut b) = (place(out_end), place(out_start));
        if b < a {
            b += 2. * (w + h);
        }
        let mut n = 0;
        c.closure[n] = out_end;
        n += 1;
        for lap in [0., 2. * (w + h)] {
            for (at, q) in corners {
                if at + lap > a && at + lap < b && n < 6 {
                    c.closure[n] = q;
                    n += 1;
                }
            }
        }
        c.closure[n] = out_start;
        c.closure_len = (n + 1) as u8;
        Some(c)
    }

    fn chain(cubics: &[[Vec2; 4]], closed: bool) -> Option<Self> {
        if cubics.is_empty() || cubics.len() > MAX_CUBICS {
            return None;
        }
        let mut control = [Vec2::ZERO; 3 * MAX_CUBICS + 1];
        for (k, c) in cubics.iter().enumerate() {
            control[3 * k..3 * k + 3].copy_from_slice(&c[..3]);
        }
        control[3 * cubics.len()] = if closed {
            cubics[0][0]
        } else {
            cubics[cubics.len() - 1][3]
        };
        let length: f32 = cubics
            .iter()
            .map(|c| c[0].distance(c[1]) + c[1].distance(c[2]) + c[2].distance(c[3]))
            .sum();
        // As many points per cubic as its share of the length at the chord, within the room
        let room = (MAX_POINTS - 1) as f32;
        let mut polyline = [Vec2::ZERO; MAX_POINTS];
        let mut n = 0;
        for c in cubics {
            let l = c[0].distance(c[1]) + c[1].distance(c[2]) + c[2].distance(c[3]);
            // Enough steps for the sagitta, within its share of the room
            let bend = (c[0] - 2. * c[1] + c[2])
                .length()
                .max((c[1] - 2. * c[2] + c[3]).length());
            let sagitta = (0.75 * bend / SAGITTA).sqrt().ceil();
            let steps = (sagitta.min(room * l / length.max(1e-6)).floor() as usize).max(1);
            for s in 0..steps {
                if n < MAX_POINTS - 1 {
                    polyline[n] = eval(c, s as f32 / steps as f32);
                    n += 1;
                }
            }
        }
        polyline[n] = control[3 * cubics.len()];
        Some(Contour {
            cubics: cubics.len() as u8,
            control,
            points: (n + 1) as u8,
            polyline,
            closed,
            closure: [Vec2::ZERO; 7],
            closure_len: 0,
        })
    }

    /// The cubics of the chain.
    pub fn cubics(&self) -> impl Iterator<Item = [Vec2; 4]> + '_ {
        (0..self.cubics as usize).map(|k| {
            let c = &self.control[3 * k..3 * k + 4];
            [c[0], c[1], c[2], c[3]]
        })
    }

    pub fn len(&self) -> usize {
        self.cubics as usize
    }

    #[cfg(feature = "diagnostics")]
    pub fn is_empty(&self) -> bool {
        self.cubics == 0
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// An open chain's closure (see [`Contour::open`]), empty for a closed one.
    pub fn closure(&self) -> &[Vec2] {
        &self.closure[..self.closure_len as usize]
    }

    fn polyline(&self) -> &[Vec2] {
        &self.polyline[..self.points as usize]
    }

    /// The signed distance from `p`: negative inside a closed chain, on the closed side of an
    /// open one, by the even-odd rule.
    pub fn distance(&self, p: Vec2) -> f32 {
        let line = self.polyline();
        let (mut best, mut inside) = (f32::INFINITY, false);
        let mut edge = |a: Vec2, b: Vec2, near: bool| {
            let ab = b - a;
            if near {
                let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0., 1.);
                best = best.min((a + ab * t).distance_squared(p));
            }
            if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * ab.x {
                inside = !inside;
            }
        };
        for w in line.windows(2) {
            edge(w[0], w[1], true);
        }
        if !self.closed {
            // The rays measure, the box's corners only close the side
            let closure = self.closure();
            let (end, start) = (line[line.len() - 1], line[0]);
            edge(end, closure[0], true);
            for w in closure.windows(2) {
                edge(w[0], w[1], false);
            }
            edge(closure[closure.len() - 1], start, true);
        }
        let d = best.sqrt();
        if inside { -d } else { d }
    }

    /// Moves the chain by `offset`.
    pub fn translate(&mut self, offset: Vec2) {
        let (c, p) = (3 * self.cubics as usize + 1, self.points as usize);
        self.control[..c].iter_mut().for_each(|q| *q += offset);
        self.polyline[..p].iter_mut().for_each(|q| *q += offset);
        self.closure.iter_mut().for_each(|q| *q += offset);
    }

    /// Maps every point of the chain by `f`, a similarity, which scales distances by its
    /// factor.
    pub fn map(&mut self, f: impl Fn(Vec2) -> Vec2) {
        let (c, p) = (3 * self.cubics as usize + 1, self.points as usize);
        self.control[..c].iter_mut().for_each(|q| *q = f(*q));
        self.polyline[..p].iter_mut().for_each(|q| *q = f(*q));
        self.closure.iter_mut().for_each(|q| *q = f(*q));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A circle's signed distance is the distance from its center less its radius, to within
    /// the flattening.
    #[test]
    fn a_circle_measures_its_radius() {
        let (c, r, k) = (Vec2::splat(20.), 10., 0.552_284_8 * 10.);
        let p = [Vec2::X, Vec2::Y, -Vec2::X, -Vec2::Y].map(|d| c + d * r);
        let t = [Vec2::Y, -Vec2::X, -Vec2::Y, Vec2::X];
        let cubics: Vec<[Vec2; 4]> = (0..4)
            .map(|i| {
                let j = (i + 1) % 4;
                [p[i], p[i] + t[i] * k, p[j] - t[j] * k, p[j]]
            })
            .collect();
        let contour = Contour::new(&cubics).unwrap();
        for q in [
            Vec2::new(20., 20.),
            Vec2::new(33., 20.),
            Vec2::new(24., 27.),
        ] {
            let want = q.distance(c) - r;
            assert!((contour.distance(q) - want).abs() <= SAGITTA, "{q}");
        }
    }

    /// An open chain's sides are opposite in sign, and its rays go on past its ends.
    #[test]
    fn an_open_curve_has_two_sides() {
        let (a, b) = (Vec2::new(5., 10.), Vec2::new(15., 10.));
        let line = [a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.), b];
        let c = Contour::open(&[line], Vec2::ZERO, Vec2::splat(20.)).unwrap();
        let (up, down) = (
            c.distance(Vec2::new(10., 5.)),
            c.distance(Vec2::new(10., 15.)),
        );
        assert!((up.abs() - 5.).abs() < 1e-3 && (down.abs() - 5.).abs() < 1e-3);
        assert!(up * down < 0., "{up} {down}");
        // Beyond the end the ray measures: 3 px above it, on the upper side
        let beyond = c.distance(Vec2::new(18., 7.));
        assert!((beyond - up.signum() * 3.).abs() < 1e-3, "{beyond}");
    }
}
