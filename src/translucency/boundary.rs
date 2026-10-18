//! A layer region's boundary read against the beneath: the colours seen in the open beyond
//! the region ([`open_beyond`]) and the region placed by the beneath's continuity
//! ([`continuity`]).

use crate::HashSet;

use super::raster::{ChebTree, Conn, DVec3, Image, SHOW_THROUGH, cheb, dilate, unit_of};

/// How far beyond a region its open colours are read: 20 px at 330.
pub fn reach(img: &Image) -> usize {
    ((20.0 * unit_of(img.width)) as usize).max(1)
}

/// The colors seen in the open beyond `reach` of a region, distinct: what a beneath may be
/// where it is not continuous with the pixel across the boundary (a layer's edge on a
/// silhouette, one thing under it and another beside it).
pub fn open_beyond(img: &Image, region: &[bool], reach: usize) -> Vec<DVec3> {
    let near = dilate(region, img.width, img.height, Conn::Four, reach);
    let mut seen = HashSet::<[u64; 3]>::default();
    let mut out = Vec::new();
    for (p, n) in img.px.iter().zip(&near) {
        if !n && seen.insert([p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]) {
            out.push(*p);
        }
    }
    out
}

/// The region moved ring by ring to where peeling under (k, c) leaves the beneath continuous with
/// the outside: a boundary pixel whose peel invents a step (or leaves the gamut) and removes none
/// goes; an outside neighbour joins where its peel continues outward; each pixel moves once.
pub fn continuity(
    img: &Image,
    region: &[bool],
    k: f64,
    c: DVec3,
    veil: &[bool],
    open: &[DVec3],
) -> Vec<bool> {
    const DIRS: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let (w, h) = (img.width as isize, img.height as isize);
    let m = (1.0 - k) * c;
    let tol = SHOW_THROUGH / k;
    let peel = |o: DVec3| (o - m) / k;
    let gamut_out = |b: DVec3| (-b).max(b - DVec3::ONE).max(DVec3::ZERO).max_element();
    let image_out = |o: DVec3| {
        (m - o)
            .max(o - m - DVec3::splat(k))
            .max(DVec3::ZERO)
            .max_element()
    };
    let readable: Vec<bool> = img
        .px
        .iter()
        .zip(veil)
        .map(|(o, v)| !v && image_out(*o) <= SHOW_THROUGH)
        .collect();
    let tree = (!open.is_empty()).then(|| ChebTree::new(open));
    let seen_open = |b: DVec3| {
        tree.as_ref().is_some_and(|t| {
            t.nearest(b.clamp(DVec3::ZERO, DVec3::ONE), SHOW_THROUGH + 1e-9)
                .is_some()
        })
    };
    let at = |x: isize, y: isize| -> Option<usize> {
        (x >= 0 && y >= 0 && x < w && y < h).then(|| (y * w + x) as usize)
    };
    let mut cur = region.to_vec();
    let mut moved = vec![false; img.len()];
    loop {
        let mut out_now = Vec::new();
        let mut in_now = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) as usize;
                if moved[p] || !readable[p] {
                    continue;
                }
                let o = img.px[p];
                let b = peel(o);
                if cur[p] {
                    let (mut against, mut support) = (false, false);
                    for (dx, dy) in DIRS {
                        let Some(q) = at(x + dx, y + dy) else {
                            continue;
                        };
                        if cur[q] {
                            continue;
                        }
                        let Some(q2) = at(x + 2 * dx, y + 2 * dy) else {
                            continue;
                        };
                        let far = img.px[q2];
                        let raw = cheb(o, far);
                        let pe = cheb(b, far);
                        let step = pe > raw + tol;
                        let no = gamut_out(b) > SHOW_THROUGH || (step && !seen_open(b));
                        let yes = raw > SHOW_THROUGH && pe <= tol;
                        against |= no;
                        support |= yes;
                    }
                    if against && !support {
                        out_now.push(p);
                    }
                } else if gamut_out(b) <= SHOW_THROUGH {
                    // an outside pixel: its inside neighbour i = p - d, its far side p + 2d
                    for (dx, dy) in DIRS {
                        let (Some(i), Some(q2)) = (at(x - dx, y - dy), at(x + 2 * dx, y + 2 * dy))
                        else {
                            continue;
                        };
                        if !cur[i] || cheb(img.px[i], o) > SHOW_THROUGH {
                            continue;
                        }
                        let far = img.px[q2];
                        if cheb(o, far) > SHOW_THROUGH && cheb(b, far) <= tol {
                            in_now.push(p);
                            break;
                        }
                    }
                }
            }
        }
        if out_now.is_empty() && in_now.is_empty() {
            break;
        }
        for p in out_now {
            cur[p] = false;
            moved[p] = true;
        }
        for p in in_now {
            cur[p] = true;
            moved[p] = true;
        }
    }
    cur
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layer over a flat beneath on x < 15: a region past its edge onto the bare beneath
    /// goes back to the edge, one a pixel or two short grows to it.
    #[test]
    fn continuity_places_the_boundary_at_the_layer_edge() {
        let (w, h) = (30, 12);
        let (k, c) = (0.6, DVec3::new(0.8, 1.0, 0.8));
        let b = DVec3::new(0.9, 0.7, 0.6);
        let px: Vec<DVec3> = (0..w * h)
            .map(|i| if i % w < 15 { k * b + (1.0 - k) * c } else { b })
            .collect();
        let img = Image {
            width: w,
            height: h,
            px,
        };
        let veil = vec![false; w * h];
        let open = [b];
        for end in [18, 13] {
            let region: Vec<bool> = (0..w * h).map(|i| i % w < end).collect();
            let ct = continuity(&img, &region, k, c, &veil, &open);
            for y in 0..h {
                assert!(ct[y * w + 14], "end {end} row {y}");
                assert!(!ct[y * w + 15], "end {end} row {y}");
            }
        }
    }
}
