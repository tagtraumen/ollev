//! Connectivity, labels, dilation and skeletons as scipy and scikit-image compute them.

/// The neighbors a component joins through: the 4 of scipy's default (`CROSS`) or all 8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conn {
    Four,
    Eight,
}

impl Conn {
    pub(super) fn offsets(self) -> &'static [(isize, isize)] {
        match self {
            Conn::Four => &[(-1, 0), (0, -1), (0, 1), (1, 0)],
            Conn::Eight => &[
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ],
        }
    }
}

/// The 4-neighbours of pixel `i` within the frame: left, right, up, down.
pub fn neighbours4(i: usize, width: usize, height: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (i % width, i / width);
    [
        (x > 0).then(|| i - 1),
        (x + 1 < width).then(|| i + 1),
        (y > 0).then(|| i - width),
        (y + 1 < height).then(|| i + width),
    ]
    .into_iter()
    .flatten()
}

/// The right and lower neighbours of pixel `i` within the frame: each 4-adjacent pair once.
pub fn forward(i: usize, width: usize, height: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (i % width, i / width);
    [(x + 1 < width).then(|| i + 1), (y + 1 < height).then(|| i + width)]
        .into_iter()
        .flatten()
}

/// scipy's `ndimage.label`: 0 off the mask, components 1..=n numbered in raster order of
/// their first pixel.
pub fn label(mask: &[bool], width: usize, height: usize, conn: Conn) -> (Vec<u32>, usize) {
    let mut lab = vec![0u32; mask.len()];
    let mut n = 0u32;
    let mut stack = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || lab[start] != 0 {
            continue;
        }
        n += 1;
        lab[start] = n;
        stack.push(start);
        while let Some(p) = stack.pop() {
            let (y, x) = ((p / width) as isize, (p % width) as isize);
            for &(dy, dx) in conn.offsets() {
                let (yy, xx) = (y + dy, x + dx);
                if yy < 0 || xx < 0 || yy >= height as isize || xx >= width as isize {
                    continue;
                }
                let q = yy as usize * width + xx as usize;
                if mask[q] && lab[q] == 0 {
                    lab[q] = n;
                    stack.push(q);
                }
            }
        }
    }
    (lab, n as usize)
}

/// scipy's `binary_dilation(mask, structure, iterations)` with the border outside, the
/// structure the cross or the 3 x 3 square: the pixels within `iterations` of the mask in the
/// city-block or chessboard distance, which two raster passes give exactly (Rosenfeld, Pfaltz).
pub fn dilate(
    mask: &[bool],
    width: usize,
    height: usize,
    conn: Conn,
    iterations: usize,
) -> Vec<bool> {
    let far = u32::MAX / 2;
    let mut d: Vec<u32> = mask.iter().map(|&m| if m { 0 } else { far }).collect();
    let eight = conn == Conn::Eight;
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let mut v = d[i];
            if x > 0 {
                v = v.min(d[i - 1] + 1);
            }
            if y > 0 {
                v = v.min(d[i - width] + 1);
                if eight && x > 0 {
                    v = v.min(d[i - width - 1] + 1);
                }
                if eight && x + 1 < width {
                    v = v.min(d[i - width + 1] + 1);
                }
            }
            d[i] = v;
        }
    }
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let i = y * width + x;
            let mut v = d[i];
            if x + 1 < width {
                v = v.min(d[i + 1] + 1);
            }
            if y + 1 < height {
                v = v.min(d[i + width] + 1);
                if eight && x + 1 < width {
                    v = v.min(d[i + width + 1] + 1);
                }
                if eight && x > 0 {
                    v = v.min(d[i + width - 1] + 1);
                }
            }
            d[i] = v;
        }
    }
    let reach = u32::try_from(iterations).unwrap_or(far);
    d.into_iter().map(|v| v <= reach).collect()
}

/// scikit-image's lookup table for Zhang and Suen: per neighborhood code, 1 removable in
/// the first pass, 2 in the second, 3 in both.
const SKELETON_LUT: [u8; 256] = [
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3,
    3, //
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2,
    2, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, //
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2,
    0, //
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, //
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, //
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, //
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0,
    0, //
];

/// scikit-image's `skeletonize` (Zhang and Suen, `_fast_skeletonize`): both passes per
/// round, each pass reading the skeleton as the last pass left it, until nothing goes.
pub fn skeletonize(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    let (w2, h2) = (width + 2, height + 2);
    let mut sk = vec![0u8; w2 * h2];
    for y in 0..height {
        for x in 0..width {
            sk[(y + 1) * w2 + x + 1] = mask[y * width + x] as u8;
        }
    }
    let mut clean = sk.clone();
    let mut removed = true;
    while removed {
        removed = false;
        for pass in 0..2 {
            let first = pass == 0;
            for r in 1..h2 - 1 {
                for c in 1..w2 - 1 {
                    if sk[r * w2 + c] == 0 {
                        continue;
                    }
                    let at = |dr: isize, dc: isize| {
                        sk[(r as isize + dr) as usize * w2 + (c as isize + dc) as usize] as usize
                    };
                    let code = at(-1, -1)
                        + 2 * at(-1, 0)
                        + 4 * at(-1, 1)
                        + 8 * at(0, 1)
                        + 16 * at(1, 1)
                        + 32 * at(1, 0)
                        + 64 * at(1, -1)
                        + 128 * at(0, -1);
                    let nb = SKELETON_LUT[code];
                    if nb == 0 {
                        continue;
                    }
                    if nb == 3 || (nb == 1 && first) || (nb == 2 && !first) {
                        clean[r * w2 + c] = 0;
                        removed = true;
                    }
                }
            }
            sk.copy_from_slice(&clean);
        }
    }
    let mut out = vec![false; width * height];
    for y in 0..height {
        for x in 0..width {
            out[y * width + x] = sk[(y + 1) * w2 + x + 1] != 0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The distance passes give what the structure applied `iterations` times gives.
    #[test]
    fn dilation_matches_iterated_structure() {
        let (w, h) = (23, 17);
        let mut seed = 11u32;
        for conn in [Conn::Four, Conn::Eight] {
            for density in [1u32, 8, 60] {
                let mask: Vec<bool> = (0..w * h)
                    .map(|_| {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (seed >> 24) % 100 < density
                    })
                    .collect();
                let mut cur = mask.clone();
                for r in 0..12 {
                    assert_eq!(dilate(&mask, w, h, conn, r), cur, "{conn:?} at {r}");
                    cur = (0..w * h)
                        .map(|p| {
                            let (y, x) = ((p / w) as isize, (p % w) as isize);
                            cur[p]
                                || conn.offsets().iter().any(|&(dy, dx)| {
                                    let (yy, xx) = (y + dy, x + dx);
                                    (0..h as isize).contains(&yy)
                                        && (0..w as isize).contains(&xx)
                                        && cur[yy as usize * w + xx as usize]
                                })
                        })
                        .collect();
                }
            }
        }
    }
}
