//! The edge map: suppressed edges, their skeleton graph and its chains.

use super::{EDGE_SIGMA, TOL, ridge_radii};
use crate::translucency::raster::{
    Conn, Image, NOISE, di_zenzo, dilate, label, length_px, line_pixels_at, map1, unit_of,
};

/// Edge pixels as gradient ridges: non-maximum suppression along the gradient by linear
/// interpolation, hysteresis at 6 and 3 NOISE of step (a step h smoothed at sigma peaks at
/// `h / (sigma sqrt(2 pi))` in the gradient), 8-connected.
pub fn suppressed_edges(
    mag: &[f64],
    theta: &[f64],
    width: usize,
    height: usize,
    sigma: f64,
) -> Vec<bool> {
    let n = mag.len();
    let mut peak = vec![false; n];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let (dy, dx) = (theta[i].sin(), theta[i].cos());
            let a = map1(mag, width, height, x as f64 + dx, y as f64 + dy);
            let b = map1(mag, width, height, x as f64 - dx, y as f64 - dy);
            peak[i] = mag[i] >= a && mag[i] >= b;
        }
    }
    let unit = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    let weak: Vec<bool> = (0..n)
        .map(|i| peak[i] && mag[i] >= 3.0 * NOISE * unit)
        .collect();
    let (lab, nl) = label(&weak, width, height, Conn::Eight);
    let mut keep = vec![false; nl + 1];
    for i in 0..n {
        if peak[i] && mag[i] >= 6.0 * NOISE * unit {
            keep[lab[i] as usize] = true;
        }
    }
    keep[0] = false;
    lab.iter().map(|&l| keep[l as usize]).collect()
}

/// The edge map, its two kinds apart: step edges as gradient ridges, and thin dark ridges.
/// Step edges within a pixel of a ridge are its gradient flanks and left out (else the
/// skeleton of a drawn outline is a ladder of junctions).
pub struct EdgeMap {
    pub edges: Vec<bool>,
    pub ridge: Vec<bool>,
    pub mag: Vec<f64>,
    pub theta: Vec<f64>,
    pub strength: Vec<f64>,
    pub normal: Vec<(i32, i32)>,
}

pub fn edge_pixels(img: &Image) -> EdgeMap {
    let (w, h) = (img.width, img.height);
    let u = unit_of(w);
    let (mag, theta) = di_zenzo(img, EDGE_SIGMA * u);
    let mut edges = suppressed_edges(&mag, &theta, w, h, EDGE_SIGMA * u);
    let lp = line_pixels_at(img, &ridge_radii(u));
    let ridge: Vec<bool> = lp.strength.iter().map(|&s| s > TOL).collect();
    let flank = dilate(&ridge, w, h, Conn::Eight, length_px(img.width, 1.0));
    for (e, f) in edges.iter_mut().zip(&flank) {
        *e &= !*f;
    }
    EdgeMap {
        edges,
        ridge,
        mag,
        theta,
        strength: lp.strength,
        normal: lp.normal,
    }
}

/// The skeleton's pixels as a graph: 4-neighbors always, a diagonal neighbor only where no
/// 4-neighbor of both joins them (so a staircase corner is not a branch point). The nodes in
/// raster order, each pixel's node and the nodes' neighbor lists.
fn skeleton_graph(sk: &[bool], width: usize, height: usize) -> (Vec<usize>, Vec<usize>, Vec<Vec<usize>>) {
    let at = |y: isize, x: isize| {
        y >= 0
            && x >= 0
            && y < height as isize
            && x < width as isize
            && sk[y as usize * width + x as usize]
    };
    let nodes: Vec<usize> = (0..sk.len()).filter(|&i| sk[i]).collect();
    let mut id = vec![usize::MAX; sk.len()];
    for (k, &p) in nodes.iter().enumerate() {
        id[p] = k;
    }
    let mut nb: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (dy, dx, diagonal) in [
        (0isize, 1isize, false),
        (1, 0, false),
        (1, 1, true),
        (1, -1, true),
    ] {
        for &p in &nodes {
            let (y, x) = ((p / width) as isize, (p % width) as isize);
            if !at(y + dy, x + dx) {
                continue;
            }
            if diagonal && (at(y + dy, x) || at(y, x + dx)) {
                continue;
            }
            let q = (y + dy) as usize * width + (x + dx) as usize;
            nb[id[p]].push(q);
            nb[id[q]].push(p);
        }
    }
    (nodes, id, nb)
}

/// The skeleton cut into simple paths as (y, x), end to end (a loop cut at its first pixel):
/// junctions of three or more branches belong to no chain, a line beneath ending at the
/// layer's edge there and linked across by continuation.
pub fn trace_chains(
    sk: &[bool],
    width: usize,
    height: usize,
    min_len: usize,
) -> Vec<Vec<(usize, usize)>> {
    let (nodes, id, nb) = skeleton_graph(sk, width, height);
    let junction: Vec<bool> = nb.iter().map(|v| v.len() >= 3).collect();
    let mut seen = junction.clone();
    let mut chains = Vec::new();
    let walk = |start: usize, seen: &mut Vec<bool>| {
        let mut order = vec![start];
        seen[id[start]] = true;
        let mut cur = start;
        loop {
            match nb[id[cur]].iter().find(|&&v| !seen[id[v]]) {
                None => return order,
                Some(&v) => {
                    cur = v;
                    seen[id[cur]] = true;
                    order.push(cur);
                }
            }
        }
    };
    let as_yx = |o: Vec<usize>| {
        o.into_iter()
            .map(|q| (q / width, q % width))
            .collect::<Vec<_>>()
    };
    // paths from their ends first (a pixel with one neighbor off the junctions), then the
    // loops that are left
    for &u in &nodes {
        if seen[id[u]] {
            continue;
        }
        if nb[id[u]].iter().filter(|&&v| !junction[id[v]]).count() <= 1 {
            let order = walk(u, &mut seen);
            if order.len() >= min_len {
                chains.push(as_yx(order));
            }
        }
    }
    for &u in &nodes {
        if !seen[id[u]] {
            let order = walk(u, &mut seen);
            if order.len() >= min_len {
                chains.push(as_yx(order));
            }
        }
    }
    chains
}
