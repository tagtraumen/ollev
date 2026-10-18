//! Centerlines of bands: their stretches, ends and extensions.

use super::*;

/// What fixes a piece's end: a corner's knot the anchor goes to, a band end the piece
/// extends to along its own continuation, or nothing (a ring).
pub(crate) enum End {
    Knot(Vec2),
    Band(Vec2),
    Free,
}

/// A stretch of the centerline between two corners or ends: the chain's crossings along it,
/// in image coordinates, whether it is a line, and what fixes its ends.
pub(crate) struct Stretch {
    pub(super) samples: Vec<Vec2>,
    pub(super) line: bool,
    pub(super) start: End,
    pub(super) end: End,
}

/// The centerline as what it is built from, so every simplification candidate is a fresh
/// least-squares fit of the chain at its tolerance instead of a simplification of the
/// previous fit.
pub(crate) struct Centerline {
    pub(super) pieces: Vec<Stretch>,
    pub(super) closed: bool,
    /// For a closed chain: the corners, as indices into the one piece's samples.
    pub(super) corners: Vec<usize>,
}

impl Centerline {
    /// The centerline's cubics at `tolerance`, with a corner flag per cubic.
    pub(crate) fn build(&self, tolerance: f32) -> Option<(Vec<Cubic>, Vec<bool>)> {
        let params = crate::tracer::FitParams {
            corner_angle: KINK,
            tolerance,
            ..crate::outline::FIT
        };
        // A cubic within the tolerance of its chord is a line: a fitted cubic bends toward
        // its end at a corner, and an offset at half the width folds into a loop around a
        // bend tighter than that
        let straighten = |c: Cubic| -> Cubic {
            let chord = c[3] - c[0];
            let off = |p: Vec2| {
                if chord.length_squared() < 1e-6 {
                    0.
                } else {
                    (p - c[0]).perp_dot(chord).abs() / chord.length()
                }
            };
            if off(c[1]) <= CENTERLINE_TOLERANCE && off(c[2]) <= CENTERLINE_TOLERANCE {
                line(c[0], c[3])
            } else {
                c
            }
        };
        if self.closed {
            let s = &self.pieces[0].samples;
            if s.len() < 3 {
                return None;
            }
            let ring = [&s[..], &s[..1]].concat();
            let cubics: Vec<Cubic> = refit(
                &crate::tracer::fit::fit_at_corners(&ring, &self.corners, true, &params),
                &ring,
            )
            .into_iter()
            .map(straighten)
            .collect();
            if cubics.is_empty() {
                return None;
            }
            let flags = cubics
                .iter()
                .map(|c| self.corners.iter().any(|&i| s[i].distance(c[0]) < 1.5))
                .collect();
            return Some((cubics, flags));
        }
        let mut out: Vec<Cubic> = Vec::new();
        let mut flags: Vec<bool> = Vec::new();
        for (j, p) in self.pieces.iter().enumerate() {
            let corner = j > 0;
            let (first, last) = (p.samples[0], p.samples[p.samples.len() - 1]);
            let knot = |e: &End, fallback: Vec2| if let End::Knot(q) = e { *q } else { fallback };
            if p.line || p.samples.len() < 2 {
                out.push(line(knot(&p.start, first), knot(&p.end, last)));
                flags.push(corner);
                continue;
            }
            let mut cubics: Vec<Cubic> = refit(
                &crate::tracer::fit::fit_at_corners(&p.samples, &[], false, &params),
                &p.samples,
            )
            .into_iter()
            .map(straighten)
            .collect();
            if cubics.is_empty() {
                cubics.push(line(first, last));
            }
            let m = cubics.len();
            match p.start {
                End::Knot(q) => cubics[0][0] = q,
                End::Band(q) => {
                    let out = (cubics[0][0] - cubics[0][1]).normalize_or_zero();
                    let by = (q - cubics[0][0]).dot(out);
                    cubics[0] = extend_cubic(&cubics[0], by, 0.);
                }
                End::Free => {}
            }
            match p.end {
                End::Knot(q) => cubics[m - 1][3] = q,
                End::Band(q) => {
                    let out = (cubics[m - 1][3] - cubics[m - 1][2]).normalize_or_zero();
                    let by = (q - cubics[m - 1][3]).dot(out);
                    cubics[m - 1] = extend_cubic(&cubics[m - 1], 0., by);
                }
                End::Free => {}
            }
            for (i, c) in cubics.into_iter().enumerate() {
                out.push(c);
                flags.push(i == 0 && corner);
            }
        }
        Some((out, flags))
    }
}

/// The part of `c` over the parameter range `t0..t1`, by de Casteljau, which holds past the
/// ends as well: the curve's own continuation.
pub(super) fn sub_cubic(c: &Cubic, t0: f32, t1: f32) -> Cubic {
    let u = if (1. - t0).abs() < 1e-6 {
        1.
    } else {
        (t1 - t0) / (1. - t0)
    };
    // The left part up to u of the right part from t0
    split(&split(c, t0).1, u).0
}

/// The parts of `p` before and after `t`, by de Casteljau.
fn split(p: &Cubic, t: f32) -> (Cubic, Cubic) {
    let p01 = p[0].lerp(p[1], t);
    let p12 = p[1].lerp(p[2], t);
    let p23 = p[2].lerp(p[3], t);
    let p012 = p01.lerp(p12, t);
    let p123 = p12.lerp(p23, t);
    let q = p012.lerp(p123, t);
    ([p[0], p01, p012, q], [q, p123, p23, p[3]])
}

/// `c` extended by `start` pixels before its start and `end` pixels past its end (shortened
/// when negative), along its own continuation.
pub(crate) fn extend_cubic(c: &Cubic, start: f32, end: f32) -> Cubic {
    // No farther than the cubic's own span either way: past that the extrapolation runs off
    // (a near-zero handle would give a parameter in the millions)
    let speed0 = 3. * (c[1] - c[0]).length();
    let speed1 = 3. * (c[3] - c[2]).length();
    let t0 = if speed0 > 1e-6 {
        (-start / speed0).clamp(-1., 0.5)
    } else {
        0.
    };
    let t1 = if speed1 > 1e-6 {
        (1. + end / speed1).clamp(0.5, 2.)
    } else {
        1.
    };
    sub_cubic(c, t0, t1)
}

impl Stroke {
    /// Moves an open centerline's ends outward by `start` and `end` (inward when negative),
    /// along the end cubics' continuation.
    pub(crate) fn extend_ends(&mut self, start: f32, end: f32) {
        let n = self.centerline.len();
        self.centerline[0] = extend_cubic(&self.centerline[0], start, 0.);
        self.centerline[n - 1] = extend_cubic(&self.centerline[n - 1], 0., end);
    }
}
