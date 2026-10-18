//! A layer's region from its relation: a pixel is explained when its peel `(o - (1 - k) c) / k`
//! is in gamut and within the noise (over k) of a color seen in the open, nearer one than the
//! pixel itself; open colors are those whose peel leaves the gamut, as they cannot lie under it.

use crate::HashMap;

use super::raster::{ChebTree, DVec3, Image, NOISE};

/// The interior rule's tolerance: 3 levels.
pub const EXPLAIN_TOL: f64 = 3.0 * NOISE;
/// The open colors are quantized to this many levels before they are searched.
const PEEL_Q: f64 = 4.0 / 255.0;

/// An 8-bit color as one integer, sorting as the colors do.
fn code_of(c: DVec3) -> u32 {
    let q = |v: f64| (v * 255.0).round_ties_even() as u32;
    (q(c.x) * 256 + q(c.y)) * 256 + q(c.z)
}

/// The distinct colors of an image, sorted, and per pixel the index of its color.
pub fn distinct_colors(img: &Image) -> (Vec<DVec3>, Vec<u32>) {
    let codes: Vec<u32> = img.px.iter().map(|c| code_of(*c)).collect();
    let mut u: Vec<u32> = codes.clone();
    u.sort_unstable();
    u.dedup();
    let index: HashMap<u32, u32> = u.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
    let colors: Vec<DVec3> = u
        .iter()
        .map(|&c| {
            DVec3::new(
                (c >> 16) as f64 / 255.0,
                ((c >> 8) & 255) as f64 / 255.0,
                (c & 255) as f64 / 255.0,
            )
        })
        .collect();
    (colors, codes.iter().map(|c| index[c]).collect())
}

/// Colors quantized to [`PEEL_Q`], distinct and sorted by rows, as
/// `np.unique(np.round(cols / q), axis=0) * q`.
fn quantized_palette(cols: impl Iterator<Item = DVec3>) -> Vec<DVec3> {
    let mut keys: Vec<[i64; 3]> = cols
        .map(|c| {
            [
                (c.x / PEEL_Q).round_ties_even() as i64,
                (c.y / PEEL_Q).round_ties_even() as i64,
                (c.z / PEEL_Q).round_ties_even() as i64,
            ]
        })
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys.iter()
        .map(|q| {
            DVec3::new(
                q[0] as f64 * PEEL_Q,
                q[1] as f64 * PEEL_Q,
                q[2] as f64 * PEEL_Q,
            )
        })
        .collect()
}

/// What the interior rule finds under one relation: per pixel whether it is explained.
pub struct Explained {
    pub mask: Vec<bool>,
}

/// The interior rule under (k, c), computed per distinct color with the searches bounded by
/// what their tests read. `context`, when given, is the mask of the open pixels in place of
/// those whose peel leaves the gamut.
pub fn explained_pixels(
    img: &Image,
    k: f64,
    c: DVec3,
    context: Option<&[bool]>,
    tol: f64,
) -> Explained {
    let m = (1.0 - k) * c;
    let margin = 2.0 * NOISE / k;
    let (u, inv) = distinct_colors(img);
    let pu: Vec<DVec3> = u.iter().map(|o| (*o - m) / k).collect();
    let gam_u: Vec<bool> = pu
        .iter()
        .map(|p| (0..3).all(|ch| p[ch] >= -margin && p[ch] <= 1.0 + margin))
        .collect();
    let open = match context {
        None => quantized_palette((0..u.len()).filter(|&i| !gam_u[i]).map(|i| u[i])),
        Some(ctx) => {
            quantized_palette(img.px.iter().zip(ctx).filter(|(_, c)| **c).map(|(p, _)| *p))
        }
    };
    if open.is_empty() {
        return Explained {
            mask: vec![false; img.len()],
        };
    }
    let tree = ChebTree::new(&open);
    let mut ok = vec![false; u.len()];
    let lim = tol / k;
    let cand: Vec<usize> = (0..u.len()).filter(|&i| gam_u[i]).collect();
    let mut near: Vec<(usize, f64)> = Vec::new();
    for &i in &cand {
        let q = pu[i].clamp(DVec3::ZERO, DVec3::ONE);
        let d = tree
            .nearest(q, lim * (1.0 + 1e-9) + 1e-12)
            .map_or(f64::INFINITY, |b| b.0);
        if d <= tol / k {
            near.push((i, d));
        }
    }
    if let Some(dmax) = near.iter().map(|n| n.1).reduce(f64::max) {
        let bound = dmax * (1.0 + 1e-9) + 1e-12;
        for &(i, dp) in &near {
            let d_own = tree.nearest(u[i], bound).map_or(f64::INFINITY, |b| b.0);
            if dp < d_own {
                ok[i] = true;
            }
        }
    }
    Explained {
        mask: inv.iter().map(|&i| ok[i as usize]).collect(),
    }
}

/// The interior rule's reach: a pixel is deep when its neighbours within this are in the
/// mask, one unit (1 px at 330, the 3 x 3 square), as a rim or a strand is a unit wide.
pub fn interior_radius(width: usize) -> usize {
    super::raster::length_px(width, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tint_over_two_colors_is_explained() {
        // left half open (white and red stripes), right half the same under k 0.5 over grey
        let (w, h) = (20, 10);
        let (k, c) = (0.5, DVec3::splat(0.4));
        let q = |v: DVec3| (v * 255.0).round() / 255.0;
        let px: Vec<DVec3> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let b = if y < 5 {
                    DVec3::ONE
                } else {
                    DVec3::new(1.0, 0.0, 0.0)
                };
                if x < 10 { b } else { q(k * b + (1.0 - k) * c) }
            })
            .collect();
        let img = Image {
            width: w,
            height: h,
            px,
        };
        let e = explained_pixels(&img, k, c, None, EXPLAIN_TOL);
        assert!(e.mask[15]);
        assert!(e.mask[7 * w + 15]);
        assert!(!e.mask[0]);
    }
}
