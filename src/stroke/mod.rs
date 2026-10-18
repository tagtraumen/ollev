//! Recognizing strokes: a band of nearly constant width becomes a centerline drawn with a pen,
//! from its skeleton (the centerline), the distance to the outline (the width), and caps and
//! joins chosen by rendering each kind. It costs its centerline's segments and one for the width.

use crate::render::coverage_cost::CoverageCost;
use crate::render::{FillRule, Path, Subpath};
use crate::tracer::Cubic;
use crate::tracer::fit::eval;
use glam::{IVec2, Vec2, Vec3};

use crate::hierarchy::Hierarchy;
use crate::refine::{MARGIN, Place};
use crate::{Context, Fill, Raster, Scene, Shape, Window};

mod centerline;
mod fit;
mod network;
mod place;
mod propose;
mod recognize;
mod skeleton;
mod through;

pub(crate) use centerline::*;
pub use fit::*;
use network::*;
use place::*;
pub(crate) use propose::*;
pub use recognize::*;
// for the contour metric in the bench
#[cfg(feature = "diagnostics")]
pub use skeleton::thin;
use skeleton::*;
pub use through::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cap {
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Join {
    Miter,
    Round,
    Bevel,
}

/// A stroked path: the centerline, open or closed, its width, and how its ends and corners
/// are drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct Stroke {
    pub centerline: Vec<Cubic>,
    pub closed: bool,
    pub width: f32,
    pub cap: Cap,
    pub join: Join,
    /// Which joints of the centerline are corners, where the join shows.
    pub corners: Vec<bool>,
    /// The dash pattern, where the stroke is dashed, see [`Dashes`].
    pub dashes: Option<Dashes>,
}

/// A stroke's dash pattern as SVG has it: lengths along the centerline, drawn and left out
/// in turn, repeating, and how far into the pattern the centerline starts.
#[derive(Clone, Debug, PartialEq)]
pub struct Dashes {
    pub array: Vec<f32>,
    pub offset: f32,
}

/// SVG's default miter limit: a sharper corner is beveled.
pub(crate) const MITER_LIMIT: f32 = 4.;
/// Chord length, in pixels, the centerline is flattened at for the outline.
const FLATTEN: f32 = 1.5;
/// Most steps a cubic is flattened in, whatever its length: a cubic longer than that is a
/// runaway, not a stroke.
const MAX_FLATTEN_STEPS: usize = 4096;
/// Points per quarter turn of a round cap or join.
const ROUND_STEPS: usize = 6;
/// Fewest skeleton pixels for a shape to be a stroke.
const MIN_SKELETON: usize = 6;
/// Share of the skeleton's pixels the chain has to pass through; the rest are the thinning's
/// clusters at joints. Below it the skeleton branches, and the shape is a network of strokes.
const CHAIN_SHARE: f32 = 0.8;
/// Share of the skeleton whose width lies within [`WIDTH_SLACK`] of the median for the band
/// to have one width.
const WIDTH_SHARE: f32 = 0.8;
pub(crate) const WIDTH_SLACK: f32 = 0.15;
/// The width read off a thresholded raster is uncertain by half a pixel on either side.
pub(crate) const WIDTH_QUANTUM: f32 = 2.;
/// Widest stroke recognized, in pixels: wider bands are shapes.
pub(crate) const MAX_WIDTH: f32 = 64.;
/// How far the simplified centerline may stray from the skeleton's fit, in pixels.
pub(crate) const CENTERLINE_TOLERANCE: f32 = 1.;
/// Tolerances the fitted centerline is simplified at, in pixels, like an outline's.
const SIMPLIFY_TOLERANCES: [f32; 3] = [2., 4., 8.];
/// Least turn at a joint of the centerline that is a corner: a visible kink.
pub(crate) const KINK: f32 = 20f32.to_radians();
/// Steps each cubic is flattened in for arc lengths.
const ARC_STEPS: usize = 16;

impl Stroke {
    /// The stroke's cost in segments: the centerline's, one for the width, and one per dash
    /// and gap pair of the pattern.
    pub fn segments(&self) -> usize {
        self.centerline.len()
            + 1
            + self
                .dashes
                .as_ref()
                .map_or(0, |d| d.array.len().div_ceil(2))
    }

    /// The centerline's length up to each cubic's end, and in all, by flattening.
    pub(crate) fn arc_lengths(&self) -> Vec<f32> {
        let mut total = 0f32;
        let mut ends = Vec::with_capacity(self.centerline.len());
        for c in &self.centerline {
            let mut from = c[0];
            for i in 1..=ARC_STEPS {
                let to = eval(c, i as f32 / ARC_STEPS as f32);
                total += from.distance(to);
                from = to;
            }
            ends.push(total);
        }
        ends
    }

    /// The cubic and parameter at arc length `s` along the centerline.
    fn locate(&self, s: f32, ends: &[f32]) -> (usize, f32) {
        let total = ends[ends.len() - 1];
        if s >= total {
            return (ends.len() - 1, 1.);
        }
        let k = ends.iter().position(|&e| e >= s).unwrap_or(ends.len() - 1);
        let start = if k == 0 { 0. } else { ends[k - 1] };
        let c = &self.centerline[k];
        // Walk the cubic's flattening to the length wanted
        let want = (s - start).max(0.);
        let mut walked = 0f32;
        let mut from = c[0];
        for i in 1..=ARC_STEPS {
            let t = i as f32 / ARC_STEPS as f32;
            let to = eval(c, t);
            let step = from.distance(to);
            if walked + step >= want {
                let within = if step > 1e-6 {
                    (want - walked) / step
                } else {
                    0.
                };
                return (k, ((i - 1) as f32 + within) / ARC_STEPS as f32);
            }
            walked += step;
            from = to;
        }
        (k, 1.)
    }

    /// The open undashed stroke along the centerline from arc length `s0` to `s1` (past the end
    /// around a closed centerline), with for each of its cubics the centerline cubic and the
    /// parameter range it comes from.
    pub(crate) fn between(
        &self,
        s0: f32,
        s1: f32,
        ends: &[f32],
    ) -> (Stroke, Vec<(usize, f32, f32)>) {
        let total = ends[ends.len() - 1];
        let m = self.centerline.len();
        // Around a closed centerline the stretch may pass the seam; on an open one it is
        // clamped, so a stretch ending exactly at the end does not wrap to the start
        let (s0, s1) = if self.closed {
            (
                s0.rem_euclid(total.max(1e-6)),
                if s1 > total { s1 - total } else { s1 },
            )
        } else {
            (s0.clamp(0., total), s1.clamp(0., total))
        };
        let (k0, t0) = self.locate(s0, ends);
        let (k1, t1) = self.locate(s1, ends);
        let wraps = self.closed && s1 < s0;
        let mut centerline = Vec::new();
        let mut corners = Vec::new();
        let mut mapping = Vec::new();
        if k0 == k1 && !wraps && t1 >= t0 {
            centerline.push(sub_cubic(&self.centerline[k0], t0, t1));
            corners.push(false);
            mapping.push((k0, t0, t1));
        } else {
            centerline.push(sub_cubic(&self.centerline[k0], t0, 1.));
            corners.push(false);
            mapping.push((k0, t0, 1.));
            let mut k = (k0 + 1) % m;
            while k != k1 {
                centerline.push(self.centerline[k]);
                corners.push(self.corners.get(k).copied().unwrap_or(false));
                mapping.push((k, 0., 1.));
                k = (k + 1) % m;
                if centerline.len() > m + 1 {
                    break;
                }
            }
            centerline.push(sub_cubic(&self.centerline[k1], 0., t1));
            corners.push(self.corners.get(k1).copied().unwrap_or(false));
            mapping.push((k1, 0., t1));
        }
        (
            Stroke {
                centerline,
                closed: false,
                corners,
                dashes: None,
                ..self.clone()
            },
            mapping,
        )
    }

    /// The flattened centerline and the outline's pieces together: for a dashed stroke each
    /// dash's own, the samples carrying the centerline's cubic and parameter they lie at,
    /// so the fit reaches the centerline's control points through them.
    fn geometry(&self, steps: Option<&[usize]>) -> (Vec<Sample>, Vec<Piece>) {
        if self.dashes.is_some() {
            return self.geometry_of(&self.dash_mappings());
        }
        let samples = self.flattened_with(steps);
        let pieces = self.strip(&samples);
        (samples, pieces)
    }

    /// The arc length along the centerline at parameter `t` of cubic `k`.
    pub(crate) fn arc_at(&self, k: usize, t: f32, ends: &[f32]) -> f32 {
        let start = if k == 0 { 0. } else { ends[k - 1] };
        let c = &self.centerline[k];
        let mut walked = 0f32;
        let mut from = c[0];
        let whole = (t * ARC_STEPS as f32).floor() as usize;
        for i in 1..=whole.min(ARC_STEPS) {
            let to = eval(c, i as f32 / ARC_STEPS as f32);
            walked += from.distance(to);
            from = to;
        }
        if whole < ARC_STEPS {
            walked += from.distance(eval(c, t));
        }
        start + walked
    }

    /// Where the dashes lie on the centerline, as [`Stroke::between`] maps them.
    pub(crate) fn dash_mappings(&self) -> Vec<Vec<(usize, f32, f32)>> {
        let ends = self.arc_lengths();
        self.dash_spans(&ends)
            .into_iter()
            .map(|(a, b)| self.between(a, b, &ends).1)
            .collect()
    }

    /// The dash at a mapping onto the current centerline.
    fn dash_at(&self, mapping: &[(usize, f32, f32)]) -> Stroke {
        Stroke {
            centerline: mapping
                .iter()
                .map(|&(k, t0, t1)| sub_cubic(&self.centerline[k], t0, t1))
                .collect(),
            closed: false,
            corners: mapping
                .iter()
                .map(|&(k, _, _)| self.corners.get(k).copied().unwrap_or(false))
                .collect(),
            dashes: None,
            ..self.clone()
        }
    }

    /// [`Stroke::geometry`] of a dashed stroke with the dashes at given mappings, which the fit
    /// holds fixed so a dash moves with its cubic instead of sliding as the centerline stretches.
    fn geometry_of(&self, mappings: &[Vec<(usize, f32, f32)>]) -> (Vec<Sample>, Vec<Piece>) {
        let mut samples: Vec<Sample> = Vec::new();
        let mut pieces: Vec<Piece> = Vec::new();
        for mapping in mappings {
            let (dash_samples, dash_pieces) = self.dash_at(mapping).geometry(None);
            let base = samples.len();
            for sample in dash_samples {
                let (k, t0, t1) = mapping[sample.cubic.min(mapping.len() - 1)];
                samples.push(Sample {
                    cubic: k,
                    t: t0 + sample.t * (t1 - t0),
                    ..sample
                });
            }
            for mut piece in dash_pieces {
                for source in &mut piece.sources {
                    source.0 += base;
                }
                pieces.push(piece);
            }
        }
        (samples, pieces)
    }

    /// The stretches of arc length the dashes cover, from the pattern and its offset, over
    /// the centerline's length.
    pub(crate) fn dash_spans(&self, ends: &[f32]) -> Vec<(f32, f32)> {
        let Some(dashes) = &self.dashes else {
            return Vec::new();
        };
        let total = ends[ends.len() - 1];
        let period: f32 = dashes.array.iter().sum();
        if dashes.array.is_empty() || period <= 1e-3 {
            return Vec::new();
        }
        let mut spans = Vec::new();
        // The pattern position at the centerline's start is the offset
        let mut at = -dashes.offset.rem_euclid(period);
        let mut on = true;
        let mut i = 0;
        while at < total {
            let len = dashes.array[i % dashes.array.len()];
            // a dash of no length is a dot where it starts within the centerline
            if on {
                let (a, b) = (at.max(0.), (at + len).min(total));
                if b > a || (len == 0. && at >= 0.) {
                    spans.push((a, b));
                }
            }
            at += len;
            on = !on;
            i += 1;
            if i > 100_000 {
                break;
            }
        }
        spans
    }

    /// The outline as loops filled nonzero: one strip per dash (see [`Stroke::strip`]), which
    /// does not fold where the centerline bends tighter than half the width, since there it
    /// runs through the centerline point and its winding sums the two segments' quads.
    pub fn outline(&self) -> Vec<Vec<Cubic>> {
        self.geometry(None)
            .1
            .iter()
            .map(|piece| polygon(&piece.points))
            .collect()
    }

    /// The shape this stroke paints, with `fill`.
    pub fn shape(&self, fill: Fill) -> Shape {
        let mut shape = Shape::new(self.outline(), fill);
        shape.stroke = Some(self.clone());
        shape
    }

    /// The outline over the flattened `samples` as one polygon (a closed stroke: two loops
    /// wound opposite ways): a quad per segment, a wedge per join and the caps, shared cross
    /// edges cancelled, the inner side cut at the offsets' crossing.
    fn strip(&self, samples: &[Sample]) -> Vec<Piece> {
        let n = samples.len();
        if n < 2 {
            return Vec::new();
        }
        let half = self.width / 2.;
        let count = if self.closed { n } else { n - 1 };
        // the segments of positive length: their ends, length and unit normal
        let segments: Vec<(usize, usize, f32, Vec2)> = (0..count)
            .filter_map(|i| {
                let (ia, ib) = (i, (i + 1) % n);
                let d = samples[ib].point - samples[ia].point;
                let length = d.length();
                (length > 1e-4).then(|| (ia, ib, length, (d / length).perp()))
            })
            .collect();
        let m = segments.len();
        if m == 0 {
            // A dash of no length: a round cap draws it as a disc, the others as nothing
            if self.closed || self.cap != Cap::Round {
                return Vec::new();
            }
            let r = Vec2::new(half, 0.);
            let quarters = [r, r.perp(), -r, -r.perp(), r];
            let sources = quarters
                .windows(2)
                .flat_map(|q| arc(q[0], q[1]).into_iter().skip(1))
                .map(|o| (0, o))
                .collect();
            return vec![Piece::new(sources, samples)];
        }
        // One side: each segment's offset ends; at a joint, on the outer side the join, on the
        // inner the offsets' crossing where both segments reach it, else the centerline point,
        // which keeps the sum of the two quads (a bend tighter than half the width)
        let side = |sign: f32| -> Vec<(usize, Vec2)> {
            let mut out = Vec::new();
            if !self.closed {
                out.push((segments[0].0, segments[0].3 * half * sign));
            }
            for s in 0..m {
                let (_, ib, length, normal) = segments[s];
                let o1 = normal * half * sign;
                let next = if s + 1 < m {
                    segments[s + 1]
                } else if self.closed {
                    segments[0]
                } else {
                    out.push((ib, o1));
                    continue;
                };
                let (ja, _, next_length, next_normal) = next;
                let o2 = next_normal * half * sign;
                let cos = normal.dot(next_normal);
                let cross = normal.perp_dot(next_normal);
                if (o1 - o2).length() < 1e-4 {
                    out.push((ib, o1));
                } else if cross * sign < 0. {
                    // the outer side: the join (a smooth joint mitred, as its turn is small)
                    out.push((ib, o1));
                    let join = if samples[ib].corner {
                        self.join
                    } else {
                        Join::Miter
                    };
                    match join {
                        Join::Round => {
                            let a = arc(o1, o2);
                            out.extend(a[1..a.len() - 1].iter().map(|&o| (ib, o)));
                        }
                        Join::Miter => {
                            if let Some(m) = miter(o1 / half, o2 / half, half, MITER_LIMIT) {
                                out.push((ib, m));
                            }
                        }
                        Join::Bevel => {}
                    }
                    out.push((ja, o2));
                } else {
                    let trim = half * cross.abs() / (1. + cos).max(1e-6);
                    if 1. + cos > 1e-3 && trim <= 0.5 * length.min(next_length) {
                        out.push((ib, (o1 + o2) / (1. + cos)));
                    } else {
                        out.push((ib, o1));
                        out.push((ib, Vec2::ZERO));
                        out.push((ja, o2));
                    }
                }
            }
            out
        };
        let (left, right) = (side(1.), side(-1.));
        if self.closed {
            let back: Vec<(usize, Vec2)> = right.into_iter().rev().collect();
            return [left, back]
                .into_iter()
                .map(|sources| Piece::with_winding(sources, samples))
                .filter(|p| p.points.len() >= 3)
                .collect();
        }
        // The caps along the end segments, from one side to the other
        let cap = |i: usize, normal: Vec2, direction: Vec2| -> Vec<(usize, Vec2)> {
            let left = normal * half;
            let offsets = match self.cap {
                Cap::Butt => Vec::new(),
                Cap::Square => vec![left + direction * half, -left + direction * half],
                // Through the point straight ahead, since half a turn from one side to the
                // other could go either way round
                Cap::Round => {
                    let ahead = direction * half;
                    let mut a = arc(left, ahead);
                    a.extend(arc(ahead, -left).into_iter().skip(1));
                    a[1..a.len() - 1].to_vec()
                }
            };
            offsets.into_iter().map(|o| (i, o)).collect()
        };
        let (first, last) = (segments[0], segments[m - 1]);
        let mut sources = left;
        sources.extend(cap(last.1, last.3, -last.3.perp()));
        sources.extend(right.into_iter().rev());
        sources.extend(cap(first.0, -first.3, first.3.perp()));
        let piece = Piece::new(sources, samples);
        if piece.points.len() >= 3 {
            vec![piece]
        } else {
            Vec::new()
        }
    }

    /// The outline as the rasterizer's path, moved by `-origin`: line cubics with their
    /// handles on the thirds.
    pub(crate) fn path(&self, origin: Vec2) -> Path {
        path_of(origin, &self.geometry(None).1)
    }

    /// How many steps each cubic is flattened in: by its length, at [`FLATTEN`] per step.
    fn steps(&self) -> Vec<usize> {
        self.centerline
            .iter()
            .map(|c| ((hull_length(c) / FLATTEN).ceil() as usize).clamp(1, MAX_FLATTEN_STEPS))
            .collect()
    }

    /// The centerline flattened at the given steps per cubic (by length when `None`, fixed in a
    /// fit so no sample comes or goes), a cubic's first point marked where it is a corner.
    fn flattened_with(&self, fixed: Option<&[usize]>) -> Vec<Sample> {
        let m = self.centerline.len();
        let mut samples = Vec::new();
        let by_length = self.steps();
        for (k, c) in self.centerline.iter().enumerate() {
            let steps = fixed.map_or(by_length[k], |f| f[k]);
            for i in 0..steps {
                let t = i as f32 / steps as f32;
                // the start of an open centerline is no joint
                let corner = i == 0
                    && (k > 0 || self.closed)
                    && self.corners.get(k).copied().unwrap_or(false);
                samples.push(Sample {
                    point: eval(c, t),
                    corner,
                    cubic: k,
                    t,
                });
            }
        }
        if !self.closed {
            samples.push(Sample {
                point: self.centerline[m - 1][3],
                corner: false,
                cubic: m - 1,
                t: 1.,
            });
        }
        samples
    }

    /// The centerline's control points as a flat list: every cubic's anchor and two handles,
    /// and the last anchor of an open centerline, then the width.
    fn params(&self, origin: Vec2) -> Vec<f32> {
        let mut params: Vec<f32> = self
            .centerline
            .iter()
            .flat_map(|c| [c[0], c[1], c[2]])
            .flat_map(|p| (p - origin).to_array())
            .collect();
        if !self.closed {
            let last = self.centerline[self.centerline.len() - 1][3] - origin;
            params.extend(last.to_array());
        }
        params.push(self.width);
        params
    }

    /// The stroke with its centerline and width from `params` (see [`Stroke::params`]).
    fn with_params(&self, params: &[f32], origin: Vec2) -> Stroke {
        let m = self.centerline.len();
        let points = (params.len() - 1) / 2;
        let point = |k: usize| Vec2::new(params[2 * k], params[2 * k + 1]) + origin;
        let centerline = (0..m)
            .map(|k| {
                let next = if self.closed {
                    (3 * k + 3) % points
                } else {
                    3 * k + 3
                };
                [
                    point(3 * k),
                    point(3 * k + 1),
                    point(3 * k + 2),
                    point(next),
                ]
            })
            .collect();
        Stroke {
            centerline,
            width: params[params.len() - 1].max(1.),
            ..self.clone()
        }
    }
}

/// The outline's `pieces` as the rasterizer's path, moved by `-origin`.
fn path_of(origin: Vec2, pieces: &[Piece]) -> Path {
    Path {
        subpaths: pieces
            .iter()
            .map(|piece| Subpath {
                points: polygon(&piece.points)
                    .iter()
                    .flat_map(|c| [c[0], c[1], c[2]])
                    .map(|p| p - origin)
                    .collect(),
            })
            .collect(),
        color: [0., 0., 0., 1.],
        gradient: None,
        fill_rule: FillRule::NonZero,
        blur: 0.,
    }
}

#[cfg(test)]
mod strip_tests {
    use super::*;

    /// A round-capped, round-joined stroke along a zigzag with a hairpin covers what lies within
    /// half its width of the centerline, as a dense sampling counts it, pixel by pixel.
    #[test]
    fn the_strip_covers_the_stroke() {
        let points = [
            Vec2::new(10., 10.),
            Vec2::new(60., 14.),
            Vec2::new(30., 30.),
            Vec2::new(70., 52.),
            Vec2::new(20., 54.),
        ];
        let centerline: Vec<Cubic> = points.windows(2).map(|w| line(w[0], w[1])).collect();
        let width = 9.;
        let stroke = Stroke {
            corners: vec![true; centerline.len()],
            centerline,
            closed: false,
            width,
            cap: Cap::Round,
            join: Join::Round,
            dashes: None,
        };
        let raster = stroke.shape(Fill::solid(Vec3::ZERO)).raster().clone();
        let distance = |p: Vec2| {
            points
                .windows(2)
                .map(|w| {
                    let (a, b) = (w[0], w[1]);
                    let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0., 1.);
                    p.distance(a + (b - a) * t)
                })
                .fold(f32::INFINITY, f32::min)
        };
        let (mut worst, mut drawn, mut sampled) = (0f32, 0f32, 0f32);
        for y in 0..70 {
            for x in 0..90 {
                const N: usize = 8;
                let inside = (0..N * N)
                    .filter(|k| {
                        let p = Vec2::new(
                            x as f32 + (k % N) as f32 / N as f32 + 0.5 / N as f32,
                            y as f32 + (k / N) as f32 / N as f32 + 0.5 / N as f32,
                        );
                        distance(p) <= width / 2.
                    })
                    .count() as f32
                    / (N * N) as f32;
                let got = if x >= raster.x0
                    && y >= raster.y0
                    && x < raster.x0 + raster.w
                    && y < raster.y0 + raster.h
                {
                    raster.values[(y - raster.y0) * raster.w + x - raster.x0]
                } else {
                    0.
                };
                worst = worst.max((got - inside).abs());
                drawn += got;
                sampled += inside;
            }
        }
        assert!(worst < 0.2, "a pixel off by {worst}");
        assert!(
            (drawn - sampled).abs() < 0.01 * sampled,
            "{drawn} against {sampled}"
        );
    }

    /// A closed ring paints an annulus, whichever way round its centerline runs.
    #[test]
    fn a_ring_is_an_annulus() {
        let (c, r, width) = (Vec2::new(40., 40.), 20., 6.);
        for turn in [1f32, -1.] {
            let quarter = |k: usize| {
                let (a, b) = (
                    Vec2::from_angle(turn * k as f32 * std::f32::consts::FRAC_PI_2),
                    Vec2::from_angle(turn * (k + 1) as f32 * std::f32::consts::FRAC_PI_2),
                );
                let h = 0.552_284_8 * r;
                [
                    c + a * r,
                    c + a * r + a.perp() * h * turn,
                    c + b * r - b.perp() * h * turn,
                    c + b * r,
                ]
            };
            let stroke = Stroke {
                centerline: (0..4).map(quarter).collect(),
                closed: true,
                width,
                cap: Cap::Butt,
                join: Join::Miter,
                corners: vec![false; 4],
                dashes: None,
            };
            let covered: f32 = stroke
                .shape(Fill::solid(Vec3::ZERO))
                .raster()
                .values
                .iter()
                .sum();
            let annulus =
                std::f32::consts::PI * ((r + width / 2.).powi(2) - (r - width / 2.).powi(2));
            assert!(
                (covered - annulus).abs() < 0.01 * annulus,
                "{covered} against {annulus}"
            );
        }
    }
}
