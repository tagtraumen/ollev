//! Mixed pixels: the anti-aliased blends between two shapes, which belong to neither. Mixed
//! pixels and slivers (shapes of no interior whose colour lies between two neighbours') take
//! the nearest real shape's label, so adjacency and extent are read between real shapes.

use std::collections::VecDeque;

use super::materials::MATCH_TOL;
use super::raster::{Conn, DVec3, Image, dilate, length_px};

/// Whether `c` lies strictly between `a` and `b`: within the match tolerance of the segment
/// from `a` to `b`, and farther than the tolerance from both ends.
fn between(c: DVec3, a: DVec3, b: DVec3) -> bool {
    let ab = b - a;
    let len2 = ab.dot(ab);
    if len2 <= 1e-12 {
        return false;
    }
    let t = (c - a).dot(ab) / len2;
    if !(0.0..=1.0).contains(&t) {
        return false;
    }
    let foot = a + ab * t;
    (c - foot).abs().max_element() <= MATCH_TOL
        && (c - a).abs().max_element() > MATCH_TOL
        && (c - b).abs().max_element() > MATCH_TOL
}

/// The labels with the mixed pixels and the slivers reassigned to the nearest real shape,
/// and the mixed mask. `labels` is a label per pixel, dense from 0.
pub fn clean_labels(labels: &[u32], img: &Image, w: usize, h: usize) -> (Vec<u32>, Vec<bool>) {
    let n = labels.iter().max().map_or(0, |&l| l as usize + 1);
    let mut sum = vec![DVec3::ZERO; n];
    let mut px = vec![0usize; n];
    for (i, &l) in labels.iter().enumerate() {
        sum[l as usize] += img.px[i];
        px[l as usize] += 1;
    }
    let mean: Vec<DVec3> = (0..n)
        .map(|l| {
            if px[l] > 0 {
                sum[l] / px[l] as f64
            } else {
                DVec3::ZERO
            }
        })
        .collect();
    let at = |x: isize, y: isize| -> Option<usize> {
        (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h)
            .then(|| y as usize * w + x as usize)
    };
    // the shapes' interiors beyond the anti-aliasing scale: a boundary pixel grown by the scale
    let boundary: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            [(1, 0), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .any(|&(dx, dy)| at(x + dx, y + dy).is_some_and(|j| labels[j] != labels[i]))
        })
        .collect();
    let band = dilate(&boundary, w, h, Conn::Eight, length_px(w, 1.0).max(1));
    let mut interior = vec![false; n];
    for (i, &l) in labels.iter().enumerate() {
        if !band[i] {
            interior[l as usize] = true;
        }
    }
    // per shape its neighbouring shapes
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in 0..w * h {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        let l = labels[i];
        for (dx, dy) in [(1, 0), (0, 1)] {
            if let Some(j) = at(x + dx, y + dy)
                && labels[j] != l
            {
                let m = labels[j];
                if !neighbours[l as usize].contains(&m) {
                    neighbours[l as usize].push(m);
                }
                if !neighbours[m as usize].contains(&l) {
                    neighbours[m as usize].push(l);
                }
            }
        }
    }
    // a sliver: no interior, its colour between two neighbours' (the blend band drawn as a
    // shape); a thin dark outline or a highlight line is darker or lighter than both sides
    // and is no sliver
    let sliver: Vec<bool> = (0..n)
        .map(|l| {
            !interior[l]
                && px[l] > 0
                && neighbours[l].iter().enumerate().any(|(ia, &a)| {
                    neighbours[l][ia + 1..]
                        .iter()
                        .any(|&b| between(mean[l], mean[a as usize], mean[b as usize]))
                })
        })
        .collect();
    // a mixed pixel: its colour between its own shape's and a neighbouring shape's
    let mut mixed: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            let l = labels[i] as usize;
            sliver[l]
                || (band[i]
                    && (-1..=1).any(|dy| {
                        (-1..=1).any(|dx| {
                            (dx != 0 || dy != 0)
                                && at(x + dx, y + dy).is_some_and(|j| {
                                    let m = labels[j] as usize;
                                    m != l && !sliver[m] && between(img.px[i], mean[l], mean[m])
                                })
                        })
                    }))
        })
        .collect();
    // every pixel of a shape mixed: the shape is a sliver after all
    let mut unmixed = vec![0usize; n];
    for (i, &l) in labels.iter().enumerate() {
        if !mixed[i] {
            unmixed[l as usize] += 1;
        }
    }
    for (i, &l) in labels.iter().enumerate() {
        if unmixed[l as usize] == 0 {
            mixed[i] = true;
        }
    }
    // the nearest real pixel's label, by a wave from every real pixel
    let mut out = labels.to_vec();
    let mut queue = VecDeque::new();
    let mut done: Vec<bool> = mixed.iter().map(|m| !m).collect();
    queue.extend((0..w * h).filter(|&i| !mixed[i]));
    while let Some(i) = queue.pop_front() {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            if let Some(j) = at(x + dx, y + dy)
                && !done[j]
            {
                out[j] = out[i];
                done[j] = true;
                queue.push_back(j);
            }
        }
    }
    (out, mixed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blend_band_and_a_sliver_dissolve_but_a_dark_line_stays() {
        // two flat shapes with a one-pixel blend column between them (labelled as a third
        // shape), and a dark line as a fourth shape: the blend column joins its sides, the
        // dark line keeps its label
        let (w, h) = (40usize, 8usize);
        let (a, b, d) = (
            DVec3::new(1.0, 0.5, 0.1),
            DVec3::new(0.9, 0.8, 0.7),
            DVec3::ZERO,
        );
        let mut labels = vec![0u32; w * h];
        let mut px = vec![a; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x == 19 {
                    labels[i] = 1;
                    px[i] = (a + b) * 0.5;
                } else if (20..30).contains(&x) {
                    labels[i] = 2;
                    px[i] = b;
                } else if x == 30 {
                    labels[i] = 3;
                    px[i] = d;
                } else if x > 30 {
                    labels[i] = 2;
                    px[i] = b;
                }
            }
        }
        let img = Image {
            width: w,
            height: h,
            px,
        };
        let (out, mixed) = clean_labels(&labels, &img, w, h);
        assert!(mixed[4 * w + 19]);
        assert!(out[4 * w + 19] == 0 || out[4 * w + 19] == 2);
        assert_eq!(out[4 * w + 30], 3, "the dark line is no sliver");
        assert_eq!(out[4 * w + 5], 0);
    }
}
