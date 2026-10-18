//! Cubics fitted to point chains.

use glam::DVec2;

use super::Bezier;
use crate::translucency::raster::{linspace, np_sum};

pub(super) fn hyp(v: DVec2) -> f64 {
    v.x.hypot(v.y)
}

pub(super) fn unit(v: DVec2) -> DVec2 {
    let n = hyp(v);
    if n > 1e-12 {
        v / n
    } else {
        DVec2::new(1.0, 0.0)
    }
}

pub(super) fn dot(a: DVec2, b: DVec2) -> f64 {
    a.x * b.x + a.y * b.y
}

/// A point of the cubic at t.
pub fn bez(c: &Bezier, t: f64) -> DVec2 {
    let u = 1.0 - t;
    u.powf(3.0) * c[0]
        + ((3.0 * u) * u * t) * c[1]
        + ((3.0 * u) * t * t) * c[2]
        + t.powf(3.0) * c[3]
}

/// The cubic's first derivative at t.
pub fn bez_d1(c: &Bezier, t: f64) -> DVec2 {
    let u = 1.0 - t;
    ((3.0 * u) * u) * (c[1] - c[0])
        + ((6.0 * u) * t) * (c[2] - c[1])
        + ((3.0 * t) * t) * (c[3] - c[2])
}

/// The cubic's second derivative at t.
pub fn bez_d2(c: &Bezier, t: f64) -> DVec2 {
    (6.0 * (1.0 - t)) * ((c[2] - 2.0 * c[1]) + c[0]) + (6.0 * t) * ((c[3] - 2.0 * c[2]) + c[1])
}

/// The arc lengths at the points: 0 then the running sum of the steps.
pub(super) fn arc_of(p: &[DVec2]) -> Vec<f64> {
    let mut arc = Vec::with_capacity(p.len());
    arc.push(0.0);
    let mut acc = 0.0;
    for w in p.windows(2) {
        acc += hyp(w[1] - w[0]);
        arc.push(acc);
    }
    arc
}

pub(super) fn chord_params(p: &[DVec2]) -> Vec<f64> {
    let d = arc_of(p);
    let last = *d.last().unwrap();
    if last > 0.0 {
        d.iter().map(|v| v / last).collect()
    } else {
        linspace(0.0, 1.0, p.len())
    }
}

/// Schneider's least-squares handles along fixed end tangents.
pub(super) fn bezier_ls(p: &[DVec2], u: &[f64], t1: DVec2, t2: DVec2) -> Bezier {
    let (p0, p3) = (p[0], p[p.len() - 1]);
    let n = p.len();
    let (mut c00, mut c01, mut c11, mut x0, mut x1) = (
        Vec::with_capacity(2 * n),
        Vec::with_capacity(2 * n),
        Vec::with_capacity(2 * n),
        Vec::with_capacity(2 * n),
        Vec::with_capacity(2 * n),
    );
    for i in 0..n {
        let (ui, b) = (u[i], 1.0 - u[i]);
        let a1 = ((3.0 * b) * b * ui) * t1;
        let a2 = ((3.0 * b) * ui * ui) * t2;
        let base =
            (b.powf(3.0) + (3.0 * b) * b * ui) * p0 + ((3.0 * b) * ui * ui + ui.powf(3.0)) * p3;
        let r = p[i] - base;
        c00.extend([a1.x * a1.x, a1.y * a1.y]);
        c01.extend([a1.x * a2.x, a1.y * a2.y]);
        c11.extend([a2.x * a2.x, a2.y * a2.y]);
        x0.extend([a1.x * r.x, a1.y * r.y]);
        x1.extend([a2.x * r.x, a2.y * r.y]);
    }
    let (c00, c01, c11, x0, x1) = (
        np_sum(&c00),
        np_sum(&c01),
        np_sum(&c11),
        np_sum(&x0),
        np_sum(&x1),
    );
    let det = c00 * c11 - c01 * c01;
    let seg = hyp(p3 - p0);
    let (mut a1, mut a2) = if det.abs() > 1e-12 {
        ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det)
    } else {
        (seg / 3.0, seg / 3.0)
    };
    if a1 < 1e-6 * seg || a2 < 1e-6 * seg {
        a1 = seg / 3.0;
        a2 = seg / 3.0;
    }
    [p0, p0 + a1 * t1, p3 + a2 * t2, p3]
}

pub(super) fn newton(c: &Bezier, p: &[DVec2], u: &[f64]) -> Vec<f64> {
    p.iter()
        .zip(u)
        .map(|(pt, &t)| {
            let d = bez(c, t) - *pt;
            let (d1, d2) = (bez_d1(c, t), bez_d2(c, t));
            let num = d.x * d1.x + d.y * d1.y;
            let den = (d1.x * d1.x + d.x * d2.x) + (d1.y * d1.y + d.y * d2.y);
            let un = if den.abs() > 1e-12 { t - num / den } else { t };
            un.clamp(0.0, 1.0)
        })
        .collect()
}

pub(super) fn errors(c: &Bezier, p: &[DVec2], u: &[f64]) -> Vec<f64> {
    p.iter()
        .zip(u)
        .map(|(pt, &t)| hyp(bez(c, t) - *pt))
        .collect()
}

pub(super) fn max_of(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// The tangents at a run's ends, from its points within `reach` of each end (a chain's last
/// pixel alone gives one of eight directions).
pub(super) fn end_tangents(p: &[DVec2], reach: f64) -> (DVec2, DVec2) {
    let tangent = |end: DVec2, next: DVec2| {
        let near: Vec<DVec2> = p
            .iter()
            .copied()
            .filter(|q| hyp(*q - end) <= reach)
            .collect();
        if near.len() > 1 {
            unit(rows_mean2(&near) - end)
        } else {
            unit(next - end)
        }
    };
    let n = p.len();
    (tangent(p[0], p[1]), tangent(p[n - 1], p[n - 2]))
}

pub(super) fn rows_mean2(rows: &[DVec2]) -> DVec2 {
    let mut s = DVec2::ZERO;
    for r in rows {
        s += *r;
    }
    s / rows.len() as f64
}

/// Schneider's fit: one cubic by least squares along the end tangents, parameters improved
/// by Newton steps; where it misses a point by more than `tol`, split at the worst point and
/// fit each half.
pub fn fit_cubics(p: &[DVec2], tol: f64) -> Vec<Bezier> {
    if p.len() < 2 {
        return Vec::new();
    }
    let (t1, t2) = end_tangents(p, 3.0 * tol);
    fit(p, t1, t2, tol, 0)
}

pub(super) fn fit(p: &[DVec2], t1: DVec2, t2: DVec2, tol: f64, depth: usize) -> Vec<Bezier> {
    let n = p.len();
    if n == 2 || hyp(p[n - 1] - p[0]) < 1e-9 {
        let d = hyp(p[n - 1] - p[0]) / 3.0;
        return vec![[p[0], p[0] + d * t1, p[n - 1] + d * t2, p[n - 1]]];
    }
    let mut u = chord_params(p);
    let mut c = bezier_ls(p, &u, t1, t2);
    for _ in 0..8 {
        if max_of(&errors(&c, p, &u)) <= tol {
            return vec![c];
        }
        u = newton(&c, p, &u);
        c = bezier_ls(p, &u, t1, t2);
    }
    let err = errors(&c, p, &u);
    if max_of(&err) <= tol || depth > 12 || n < 4 {
        return vec![c];
    }
    let mut worst = 0;
    for (i, e) in err.iter().enumerate() {
        if *e > err[worst] {
            worst = i;
        }
    }
    let i = worst.clamp(1, n - 2);
    let tc = unit(p[i - 1] - p[i + 1]);
    let mut out = fit(&p[..=i], t1, tc, tol, depth + 1);
    out.extend(fit(&p[i..], -tc, t2, tol, depth + 1));
    out
}

/// An open polyline smoothed by [1, 2, 1] / 4 inside (a pixel chain's staircase).
pub fn smooth_open(p: &[DVec2]) -> Vec<DVec2> {
    let mut q = p.to_vec();
    for i in 1..p.len().saturating_sub(1) {
        q[i] = 0.25 * p[i - 1] + 0.5 * p[i] + 0.25 * p[i + 1];
    }
    q
}

/// The signed curvature of the cubic at t (positive turning left).
pub fn curvature(c: &Bezier, t: f64) -> f64 {
    let (d1, d2) = (bez_d1(c, t), bez_d2(c, t));
    let v = hyp(d1);
    (d1.x * d2.y - d1.y * d2.x) / v.powf(3.0).max(1e-9)
}
