//! Chains of edge points with their ends, split at corners.

use super::*;

// ----------------------------------------------------------------------------- chains

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Edge,
    Ridge,
}

/// One end of a chain: its point, its outward heading, the curvature along the outward
/// heading, and the chord of its end cubic (how far its curve is known).
#[derive(Clone, Copy, Debug)]
pub struct End {
    pub p: DVec2,
    pub heading: DVec2,
    pub k: f64,
    pub span: f64,
}

/// One line: subpixel points (x, y) in order, arc length, kind, width, cubics, and per point
/// the unit normal (the left, rotating the direction a quarter turn) and the colors left
/// and right of it (and at its centre, for a ridge).
#[derive(Clone, Debug)]
pub struct Chain {
    pub p: Vec<DVec2>,
    pub kind: Kind,
    pub width: f64,
    pub arc: Vec<f64>,
    pub length: f64,
    pub cubics: Vec<Bezier>,
    pub normal: Vec<DVec2>,
    pub side_d: f64,
    pub left: Vec<DVec3>,
    pub right: Vec<DVec3>,
    pub centre: Option<Vec<DVec3>>,
    pub ends: [End; 2],
    /// the image's length unit, which the link tests' lengths scale by
    pub unit: f64,
}

/// Edge and ridge chains of an image, fitted, with their side colors.
pub fn build_chains(img: &Image) -> Vec<Chain> {
    // the shortest chain: 5 px at 330, in the image's unit
    let min_chain = length_px(img.width, MIN_CHAIN as f64);
    let (w, h) = (img.width, img.height);
    let em = edge_pixels(img);
    let l = img.luminance();
    let mut out: Vec<Chain> = Vec::new();
    let mut traced: Vec<(Vec<(usize, usize)>, bool)> =
        trace_chains(&skeletonize(&em.edges, w, h), w, h, min_chain)
            .into_iter()
            .map(|o| (o, false))
            .collect();
    traced.extend(
        trace_chains(&skeletonize(&em.ridge, w, h), w, h, min_chain)
            .into_iter()
            .map(|o| (o, true)),
    );
    for (order, is_ridge) in traced {
        // subpixel positions: the parabola through the measure at -1, 0, +1 across the line
        // (the gradient for an edge, the ridge strength for a ridge)
        let f = if is_ridge { &em.strength } else { &em.mag };
        let mut p: Vec<DVec2> = order
            .iter()
            .map(|&(y, x)| {
                let i = y * w + x;
                let (ny, nx) = if is_ridge {
                    let (dy, dx) = em.normal[i];
                    let (ny, nx) = (dy as f64, dx as f64);
                    let nn = ny.hypot(nx).max(1e-9);
                    (ny / nn, nx / nn)
                } else {
                    (em.theta[i].sin(), em.theta[i].cos())
                };
                let (xf, yf) = (x as f64, y as f64);
                let f0 = f[i];
                let fa = map1(f, w, h, xf + nx, yf + ny);
                let fb = map1(f, w, h, xf - nx, yf - ny);
                let den = fa - 2.0 * f0 + fb;
                let off = if den < -1e-12 {
                    0.5 * (fb - fa) / den
                } else {
                    0.0
                };
                let off = off.clamp(-0.5, 0.5);
                DVec2::new(xf + off * nx, yf + off * ny)
            })
            .collect();
        // a light smoothing along the chain: the subpixel offsets of neighbors disagree by
        // the interpolation's own error
        if p.len() >= 3 {
            let q = p.clone();
            for i in 1..p.len() - 1 {
                p[i] = 0.25 * q[i - 1] + 0.5 * q[i] + 0.25 * q[i + 1];
            }
        }
        // at least two points whatever the size: a single point has no direction, so no ends
        // (on a 32 px icon the shortest chain rounds to one pixel)
        for piece in split_at_corners(&p, length_px(w, CORNER_REACH as f64)) {
            if piece.len() >= min_chain.max(2) {
                out.push(make_chain(img, &l, piece, is_ridge));
            }
        }
    }
    out
}

/// A chain cut at corners: where the directions over r px either side differ by over twice
/// the kink and over 2r px it turns less than 1.5 times that (a smooth arc turns twice).
/// The corner's own points go to neither side.
pub fn split_at_corners(p: &[DVec2], r: usize) -> Vec<Vec<DVec2>> {
    let n = p.len();
    if n < 2 * r + 3 {
        return vec![p.to_vec()];
    }
    let arc = arc_of(p);
    let turn_at = |i: usize, reach: f64| {
        let a = arc.partition_point(|&v| v < arc[i] - reach);
        let b = arc.partition_point(|&v| v < arc[i] + reach).min(n - 1);
        if a >= i || b <= i {
            return 0.0;
        }
        let (u, v) = (unit(p[i] - p[a]), unit(p[b] - p[i]));
        dot(u, v).clamp(-1.0, 1.0).acos()
    };
    let t1: Vec<f64> = (0..n).map(|i| turn_at(i, r as f64)).collect();
    let mut cuts: Vec<usize> = Vec::new();
    for i in r..n - r {
        if t1[i] <= 2.0 * kink() || t1[i] < max_of(&t1[i.saturating_sub(r)..i + r + 1]) {
            continue;
        }
        if cuts.last().is_some_and(|&c| i - c <= r) {
            continue;
        }
        if turn_at(i, 2.0 * r as f64) < 1.5 * t1[i] {
            cuts.push(i);
        }
    }
    if cuts.is_empty() {
        return vec![p.to_vec()];
    }
    let mut pieces = Vec::new();
    let mut start = 0;
    for &i in &cuts {
        pieces.push(p[start..start.max(i - 1)].to_vec());
        start = i + 2;
    }
    pieces.push(p[start.min(n)..].to_vec());
    pieces
}

/// A chain of points `p`, fitted, with its normals, width and side colors.
pub fn make_chain(img: &Image, l: &[f64], p: Vec<DVec2>, is_ridge: bool) -> Chain {
    let (w, h) = (img.width, img.height);
    let u = unit_of(w);
    let n = p.len();
    let arc = arc_of(&p);
    let cubics = fit_cubics(&p, FIT_TOL * u);
    let nr = length_px(w, 2.0);
    // direction per point from neighbors two apart, normal its left
    let normal: Vec<DVec2> = (0..n)
        .map(|i| {
            let d = p[(i + nr).min(n - 1)] - p[i.saturating_sub(nr)];
            let d = d / hyp(d).max(1e-9);
            DVec2::new(-d.y, d.x)
        })
        .collect();
    let mut width = EDGE_WIDTH * u;
    if is_ridge {
        // the ridge's width: its darkness below the sides integrated across, over its peak
        // (an anti-aliased hairline reads as a pixel or so)
        let offs = arange(-4.0 * u, 4.0 * u + 0.01, 0.5);
        let mut wd: Vec<f64> = (0..n)
            .map(|i| {
                let prof: Vec<f64> = offs
                    .iter()
                    .map(|o| map1(l, w, h, p[i].x + o * normal[i].x, p[i].y + o * normal[i].y))
                    .collect();
                let base = prof[0].max(prof[prof.len() - 1]);
                let dark: Vec<f64> = prof.iter().map(|v| (base - v).max(0.0)).collect();
                let pk = max_of(&dark);
                if pk > NOISE {
                    np_sum(&dark) * 0.5 / pk.max(1e-9)
                } else {
                    u
                }
            })
            .collect();
        width = median(&mut wd);
    }
    let side_d = if is_ridge {
        (SIDE_OFFSET * u).max(width / 2.0 + 1.5 * u)
    } else {
        SIDE_OFFSET * u
    };
    let left = (0..n)
        .map(|i| img.sample(p[i].x + side_d * normal[i].x, p[i].y + side_d * normal[i].y))
        .collect();
    let right = (0..n)
        .map(|i| img.sample(p[i].x - side_d * normal[i].x, p[i].y - side_d * normal[i].y))
        .collect();
    let centre = is_ridge.then(|| p.iter().map(|q| img.sample(q.x, q.y)).collect());
    let ends = [end_geometry(&cubics, 0), end_geometry(&cubics, 1)];
    Chain {
        length: *arc.last().unwrap(),
        p,
        kind: if is_ridge { Kind::Ridge } else { Kind::Edge },
        width,
        arc,
        cubics,
        normal,
        side_d,
        left,
        right,
        centre,
        ends,
        unit: u,
    }
}

/// An end's point, its outward heading, its curvature along the outward heading (positive
/// turning left), and the chord of its end cubic.
pub(super) fn end_geometry(cubics: &[Bezier], e: usize) -> End {
    let (c, p, heading, k) = if e == 1 {
        let c = cubics[cubics.len() - 1];
        (c, c[3], unit(bez_d1(&c, 1.0)), curvature(&c, 1.0))
    } else {
        // walking outward reverses the direction, and the turning changes sign with it
        let c = cubics[0];
        (c, c[0], -unit(bez_d1(&c, 0.0)), -curvature(&c, 0.0))
    };
    End {
        p,
        heading,
        k,
        span: hyp(c[3] - c[0]),
    }
}

/// The point and heading arc length g from p along the circle of curvature k leaving it in
/// direction hd (a straight line for k near nothing).
pub fn along_circle(p: DVec2, hd: DVec2, k: f64, g: f64) -> (DVec2, DVec2) {
    if k.abs() * g < 1e-4 {
        return (p + g * hd, hd);
    }
    let ang = k * g;
    let nrm = DVec2::new(-hd.y, hd.x);
    let q = p + (ang.sin() * hd + (1.0 - ang.cos()) * nrm) / k;
    (q, ang.cos() * hd + ang.sin() * nrm)
}
