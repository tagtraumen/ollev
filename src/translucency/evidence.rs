//! The least evidence a translucent layer leaves: a contour of two materials running on under
//! it as their images under one tint `o = k b + m` (one contour fixes k), read on the leaves
//! before anything is proposed and on a candidate's rim before it is traced.

use glam::Vec3;

use super::leaves::Leaves;
use super::{NOISE, OPACITY_MARGIN};

/// A contour crossing the layer's edge: on the leaves' contact map, a pair `x1` (under the
/// layer) beside `y1` (bare), and a pair `x2` beside `y2`, with `x1 | x2` and `y1 | y2` touching.
struct Crossings<'a> {
    lv: &'a Leaves,
    /// per directed pair its two sides' colours at the contact, weighted by the points
    sides: Vec<(Vec3, Vec3)>,
    /// the pairs each leaf is the inside of
    by_inside: Vec<Vec<usize>>,
}

impl<'a> Crossings<'a> {
    fn new(lv: &'a Leaves) -> Self {
        let sides = lv
            .pairs
            .iter()
            .map(|p| {
                let points = &lv.points[p.points.clone()];
                let w: f32 = points.iter().map(|q| q.w).sum::<f32>().max(1e-6);
                (
                    points.iter().map(|q| q.o * q.w).sum::<Vec3>() / w,
                    points.iter().map(|q| q.b * q.w).sum::<Vec3>() / w,
                )
            })
            .collect();
        let mut by_inside = vec![Vec::new(); lv.n];
        for (j, p) in lv.pairs.iter().enumerate() {
            by_inside[p.x as usize].push(j);
        }
        Self {
            lv,
            sides,
            by_inside,
        }
    }

    fn touching(&self, a: u32, b: u32) -> bool {
        self.by_inside[a as usize]
            .iter()
            .any(|&j| self.lv.pairs[j].y == b)
    }

    /// Whether some pair of pairs `(j1, j2)` forming a crossing passes `fits`.
    fn any(&self, mut fits: impl FnMut(usize, usize) -> bool) -> bool {
        let pairs = &self.lv.pairs;
        pairs.iter().enumerate().any(|(j1, p1)| {
            self.by_inside[p1.x as usize].iter().any(|&c| {
                let x2 = pairs[c].y;
                self.by_inside[x2 as usize].iter().any(|&j2| {
                    let y2 = pairs[j2].y;
                    y2 != p1.y && y2 != p1.x && self.touching(p1.y, y2) && fits(j1, j2)
                })
            })
        })
    }

    /// How far a pair's inside may sit off its outside's image: the noise and both planes'.
    fn slack(&self, j: usize, k: f32) -> f32 {
        let p = &self.lv.pairs[j];
        let unc = &self.lv.uncertainty;
        2. * NOISE + unc[p.x as usize] + k * unc[p.y as usize]
    }
}

/// The step beneath a contour must be one the layer's image can be read off.
const STEP: f32 = 3. * NOISE;

/// Whether the picture holds a contour running on under some tint with k a layer's (within
/// [`OPACITY_MARGIN`] of clear and opaque) and its colour in gamut, the layer showing.
pub fn layer_witnessed(lv: &Leaves) -> bool {
    let cross = Crossings::new(lv);
    cross.any(|j1, j2| {
        let ((o1, b1), (o2, b2)) = (cross.sides[j1], cross.sides[j2]);
        let db = b1 - b2;
        if db.length() < STEP {
            return false;
        }
        let k = (o1 - o2).dot(db) / db.length_squared();
        if !(OPACITY_MARGIN..=1. - OPACITY_MARGIN).contains(&k) {
            return false;
        }
        let (s1, s2) = (cross.slack(j1, k), cross.slack(j2, k));
        let m = 0.5 * ((o1 - k * b1) + (o2 - k * b2));
        let off = |o: Vec3, b: Vec3| (o - k * b - m).abs().max_element();
        let shows = (o1 - b1).abs().max_element() > s1 || (o2 - b2).abs().max_element() > s2;
        let c = m / (1. - k);
        let margin = s1.max(s2) / (1. - k);
        shows
            && off(o1, b1) <= s1
            && off(o2, b2) <= s2
            && c.cmpge(Vec3::splat(-margin)).all()
            && c.cmple(Vec3::splat(1. + margin)).all()
    })
}

/// How far a pixel may sit off its outside's image under a relation: the noise of the two
/// pixels and of the relation fitted to them.
const SLACK: f32 = 3. * NOISE;

/// Whether a candidate's own relation `o = k b + m` is seen on a contour running on under its
/// `mask`: on its rim, pixels a step past the anti-aliasing inside and outside that pair
/// under the relation and whose outsides span a step.
pub fn witnessed_at(target: &[Vec3], width: usize, k: f32, m: Vec3, mask: &[bool]) -> bool {
    let height = target.len() / width.max(1);
    let (w, h) = (width as i64, height as i64);
    let at = |x: i64, y: i64| (x >= 0 && y >= 0 && x < w && y < h).then(|| (y * w + x) as usize);
    let (mut lo, mut hi) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
    for y in 0..h {
        for x in 0..w {
            if !mask[(y * w + x) as usize] {
                continue;
            }
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (Some(q), Some(p2), Some(q2)) = (
                    at(x + dx, y + dy),
                    at(x - dx, y - dy),
                    at(x + 2 * dx, y + 2 * dy),
                ) else {
                    continue;
                };
                if mask[q] || !mask[p2] || mask[q2] {
                    continue;
                }
                let (o, b) = (target[p2], target[q2]);
                if (o - k * b - m).abs().max_element() > SLACK {
                    continue;
                }
                (lo, hi) = (lo.min(b), hi.max(b));
                if (hi - lo).length() >= STEP {
                    return true;
                }
            }
        }
    }
    false
}
