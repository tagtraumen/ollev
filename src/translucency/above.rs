//! What lies over the beneath by geometry alone: the thin objects the linework reads as lines
//! with other things on their two sides (hair strands over the skin and the sky, a glass's
//! outlines over the wall's grid). The caller takes the layer's own rims out.

use super::raster::{Conn, Image, NOISE, dilate, length_px};
use super::vocabulary::{DIRS, colour_lines, thin_lines};

/// Per pixel, whether it belongs to a thin object a line centre puts above the objects on its
/// two sides. Thin: no pixel farther than half the widest line (4 units) from its boundary,
/// so nowhere wider than a line. A centre with one object on both sides is a gap in it.
pub fn thin_objects(img: &Image, leaves: &[u32]) -> Vec<bool> {
    let (w, h) = (img.width, img.height);
    let n = leaves.iter().max().map_or(0, |&l| l as usize + 1);
    let (strength, dir) = colour_lines(img);
    let centres = thin_lines(&strength, &dir, w, h, 3.0 * NOISE, 6.0 * NOISE);
    let reach = length_px(w, 4.0) as isize;
    let boundary: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let l = leaves[i];
            (x + 1 < w && leaves[i + 1] != l)
                || (y + 1 < h && leaves[i + w] != l)
                || (x > 0 && leaves[i - 1] != l)
                || (y > 0 && leaves[i - w] != l)
        })
        .collect();
    let near = dilate(&boundary, w, h, Conn::Eight, length_px(w, 2.0));
    let mut thin = vec![true; n];
    for (i, &l) in leaves.iter().enumerate() {
        if !near[i] {
            thin[l as usize] = false;
        }
    }
    let mut above = vec![false; n];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let i = y as usize * w + x as usize;
            let l = leaves[i];
            if !centres[i] || !thin[l as usize] || above[l as usize] {
                continue;
            }
            let (dy, dx) = DIRS[dir[i] as usize];
            let side = |sg: isize| -> Option<u32> {
                for r in 1..=reach {
                    let (yy, xx) = (y + sg * dy * r, x + sg * dx * r);
                    if yy < 0 || xx < 0 || yy >= h as isize || xx >= w as isize {
                        return None;
                    }
                    let q = leaves[yy as usize * w + xx as usize];
                    if q != l {
                        return Some(q);
                    }
                }
                None
            };
            if let (Some(a), Some(b)) = (side(1), side(-1))
                && a != b
            {
                above[l as usize] = true;
            }
        }
    }
    leaves.iter().map(|&l| above[l as usize]).collect()
}
