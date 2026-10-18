use glam::Vec2;

use crate::tracer::Cubic;
use crate::tracer::fit::{eval, fit_cubics};

pub fn beautify_chain(chain: &[Cubic], tolerance: f32, corner_angle: f32) -> Vec<Cubic> {
    let corners: Vec<bool> = (0..chain.len())
        .map(|i| {
            // a joint without a tangent on one side (a cubic of no length) is no corner
            i > 0
                && crate::tracer::angle::turn(end_tangent(&chain[i - 1]), start_tangent(&chain[i]))
                    .is_some_and(|t| t.abs() > corner_angle)
        })
        .collect();
    beautify_chain_at(chain, &corners, tolerance)
}

/// [`beautify_chain`] with the corners given: `corners[i]` says whether the joint at the start
/// of `chain[i]` is one.
pub fn beautify_chain_at(chain: &[Cubic], corners: &[bool], tolerance: f32) -> Vec<Cubic> {
    let mut out = Vec::with_capacity(chain.len());
    for piece in split_at(chain, corners) {
        let mut cubics = simplify(piece, tolerance);
        for c in &mut cubics {
            straighten(c, tolerance);
        }
        smooth_joints(&mut cubics);
        out.extend(cubics);
    }
    out
}

fn start_tangent(c: &Cubic) -> Vec2 {
    [c[1], c[2], c[3]]
        .into_iter()
        .map(|p| p - c[0])
        .find(|d| d.length_squared() > 1e-8)
        .unwrap_or(Vec2::ZERO)
        .normalize_or_zero()
}

fn end_tangent(c: &Cubic) -> Vec2 {
    [c[2], c[1], c[0]]
        .into_iter()
        .map(|p| c[3] - p)
        .find(|d| d.length_squared() > 1e-8)
        .unwrap_or(Vec2::ZERO)
        .normalize_or_zero()
}

/// Runs of cubics between the joints flagged as corners.
fn split_at<'a>(chain: &'a [Cubic], corners: &[bool]) -> Vec<&'a [Cubic]> {
    let mut pieces = Vec::new();
    let mut start = 0;
    for i in 1..chain.len() {
        if corners[i] {
            pieces.push(&chain[start..i]);
            start = i;
        }
    }
    pieces.push(&chain[start..]);
    pieces
}

/// Refits a smooth run of cubics with as few as stay within the tolerance of it, keeping its end
/// points and end tangents.
fn simplify(piece: &[Cubic], tolerance: f32) -> Vec<Cubic> {
    const SAMPLES: usize = 8;
    // An empty chain (a loop whose cubics were all pruned)
    if piece.is_empty() {
        return Vec::new();
    }
    let mut points = vec![piece[0][0]];
    for c in piece {
        points.extend((1..=SAMPLES).map(|i| eval(c, i as f32 / SAMPLES as f32)));
    }
    let t0 = start_tangent(&piece[0]);
    let t1 = -end_tangent(&piece[piece.len() - 1]);
    let refit = fit_cubics(&points, t0, t1, tolerance);
    if refit.len() < piece.len() {
        refit
    } else {
        piece.to_vec()
    }
}

/// Makes a cubic exactly straight when its handles stay within the tolerance of its chord.
fn straighten(c: &mut Cubic, tolerance: f32) {
    let chord = c[3] - c[0];
    let length = chord.length();
    if length < 1e-3 {
        return;
    }
    let normal = chord.perp() / length;
    let off = |p: Vec2| (p - c[0]).dot(normal).abs();
    if off(c[1]) <= tolerance && off(c[2]) <= tolerance {
        c[1] = c[0].lerp(c[3], 1. / 3.);
        c[2] = c[0].lerp(c[3], 2. / 3.);
    }
}

/// Aligns the handles on both sides of every joint in a smooth run, so it has no kinks. Handles
/// keep their lengths; straight cubics keep their direction and bend their neighbor instead.
fn smooth_joints(cubics: &mut [Cubic]) {
    let is_straight = |c: &Cubic| {
        let chord = c[3] - c[0];
        chord.length_squared() > 1e-6
            && [c[1], c[2]]
                .iter()
                .all(|&p| (p - c[0]).perp_dot(chord).abs() < 1e-3 * chord.length())
    };
    for i in 1..cubics.len() {
        let (a, b) = (cubics[i - 1], cubics[i]);
        let joint = a[3];
        let (d_in, d_out) = (joint - a[2], b[1] - joint);
        if d_in.length_squared() < 1e-8 || d_out.length_squared() < 1e-8 {
            continue;
        }
        let direction = match (is_straight(&a), is_straight(&b)) {
            (true, true) => continue,
            (true, false) => d_in.normalize(),
            (false, true) => d_out.normalize(),
            (false, false) => (d_in.normalize() + d_out.normalize()).normalize_or_zero(),
        };
        if direction == Vec2::ZERO {
            continue;
        }
        cubics[i - 1][2] = joint - direction * d_in.length();
        cubics[i][1] = joint + direction * d_out.length();
    }
}
