//! Blind detection of JPEG compression from decoded pixels. JPEG rounds each 8x8 block's cosine
//! frequencies to multiples of a quantization step, so differences within half a step (ringing,
//! blocking) are noise rather than detail.

use glam::{Mat3, Vec3};

/// The standard quantization tables of the JPEG specification (libjpeg), in natural order, which
/// most encoders scale by a quality factor.
const LUMA: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
const CHROMA: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// Qualities above this leave too little noise to bother.
const MAX_QUALITY: u32 = 95;
/// Fewest coefficients that say anything about a quality.
const MIN_EVIDENCE: usize = 100;
/// Rounding noise of decoded coefficients, in `0..255` units.
const NOISE: f32 = 1.5;
/// How many standard errors above chance the regularity has to be to count as compression, see
/// [`score`]. Thousands of block grids and qualities are tried, so chance alone reaches about 4.
const MIN_SIGNIFICANCE: f32 = 8.;
/// Fewest qualities that have to fit, see [`quality_of`].
const MIN_FITTING: usize = 5;
/// Share of the best score a lower quality still needs to be taken.
const SCORE_SHARE: f32 = 0.8;

/// The compression an image went through, as far as its pixels tell.
#[derive(Clone, Debug)]
pub struct Jpeg {
    /// Where the 8x8 block grid starts.
    pub offset: [usize; 2],
    luma: [f32; 64],
    chroma: [f32; 64],
}

fn table(base: &[u16; 64], quality: u32) -> [f32; 64] {
    let scale = if quality < 50 {
        5000 / quality
    } else {
        200 - 2 * quality
    };
    base.map(|b| ((b as u32 * scale + 50) / 100).clamp(1, 255) as f32)
}

/// The orthonormal 8x8 DCT basis, which is what JPEG uses: `basis[k][n]`.
fn basis() -> [[f32; 8]; 8] {
    let mut b = [[0.; 8]; 8];
    for (k, row) in b.iter_mut().enumerate() {
        let a = if k == 0 {
            (1f32 / 8.).sqrt()
        } else {
            (2f32 / 8.).sqrt()
        };
        for (n, v) in row.iter_mut().enumerate() {
            *v = a * ((2 * n + 1) as f32 * k as f32 * std::f32::consts::PI / 16.).cos();
        }
    }
    b
}

fn dct(block: &[f32; 64], b: &[[f32; 8]; 8]) -> [f32; 64] {
    let mut rows = [0.; 64];
    for y in 0..8 {
        for u in 0..8 {
            rows[y * 8 + u] = (0..8).map(|x| b[u][x] * block[y * 8 + x]).sum();
        }
    }
    let mut out = [0.; 64];
    for v in 0..8 {
        for u in 0..8 {
            out[v * 8 + u] = (0..8).map(|y| b[v][y] * rows[y * 8 + u]).sum();
        }
    }
    out
}

/// RGB in `0..1` to JPEG's YCbCr in `0..255` units, without the offsets, which cancel in
/// differences.
fn to_ycc() -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.299, -0.168_736, 0.5),
        Vec3::new(0.587, -0.331_264, -0.418_688),
        Vec3::new(0.114, 0.5, -0.081_312),
    ) * 255.
}

/// A plane of samples and the blocks over it that lie fully inside.
struct Plane {
    values: Vec<f32>,
    width: usize,
    height: usize,
}

impl Plane {
    fn blocks(&self, offset: [usize; 2]) -> impl Iterator<Item = [usize; 2]> + use<> {
        let (w, h) = (self.width, self.height);
        (offset[1]..h.saturating_sub(7))
            .step_by(8)
            .flat_map(move |y| {
                (offset[0]..w.saturating_sub(7))
                    .step_by(8)
                    .map(move |x| [x, y])
            })
    }

    fn block(&self, [x, y]: [usize; 2]) -> [f32; 64] {
        std::array::from_fn(|i| self.values[(y + i / 8) * self.width + x + i % 8])
    }
}

/// The planes of the RGB pixels `rgb` (8 bits per channel) in the colour space `matrix` maps
/// RGB in `0..1` to.
fn planes(rgb: &[u8], width: usize, height: usize, matrix: Mat3) -> [Plane; 3] {
    let colors: Vec<Vec3> = rgb
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) / 255.)
        .collect();
    std::array::from_fn(|c| Plane {
        values: colors.iter().map(|&p| (matrix * p)[c]).collect(),
        width,
        height,
    })
}

/// How regularly the coefficients sit on multiples of their step: the mean of `cos(2 pi c / q)`,
/// 1 on a multiple and 0 on average otherwise. Coefficients rounded to zero, and steps barely
/// above the rounding noise, fit any step and are left out.
fn score(coefficients: &[(u8, f32)], steps: &[f32; 64]) -> Option<f32> {
    let (mut sum, mut count) = (0f32, 0usize);
    for &(i, c) in coefficients {
        let q = steps[i as usize];
        if q >= 4. * NOISE && c.abs() > 0.5 * q {
            sum += (std::f32::consts::TAU * c / q).cos();
            count += 1;
        }
    }
    // Cosines of coefficients that ignore the step average to zero with a spread of 1 / sqrt(2)
    let significance = sum / count as f32 * (2. * count as f32).sqrt();
    (count >= MIN_EVIDENCE && significance >= MIN_SIGNIFICANCE).then(|| sum / count as f32)
}

/// The strongest compression, lowest quality, that explains the coefficients about as well as
/// any. Halving every step explains them just as well, so the best score alone would pick too
/// high a quality.
fn quality_of(
    coefficients: &[(u8, f32)],
    base: &[u16; 64],
    qualities: impl Iterator<Item = u32>,
) -> Option<(u32, f32)> {
    let scores: Vec<(u32, f32)> = qualities
        .filter_map(|q| Some((q, score(coefficients, &table(base, q))?)))
        .collect();
    let best = scores.iter().map(|s| s.1).fold(f32::MIN, f32::max);
    // Coefficients rounded to a step also lie on finer steps that divide it, which chance
    // regularity, like in pixel art of few colors, does not
    if scores.len() < MIN_FITTING {
        return None;
    }
    scores.into_iter().find(|s| s.1 >= SCORE_SHARE * best)
}

/// Detects JPEG compression in RGB pixels (8 bits per channel, row-major), or `None` when the
/// image shows no sign of it.
pub fn detect(rgb: &[u8], width: usize, height: usize) -> Option<Jpeg> {
    if width < 16 || height < 16 {
        return None;
    }
    let [y, ..] = planes(rgb, width, height, to_ycc());
    let b = basis();
    // Each block's frequencies past the mean, in block order: a coefficient within half the
    // least step [`score`] reads is rounded to zero at every quality, so it is left out here
    let coefficients = |plane: &Plane, offset: [usize; 2], limit: usize| -> Vec<(u8, f32)> {
        let blocks: Vec<[usize; 2]> = plane.blocks(offset).collect();
        let stride = blocks.len().div_ceil(limit).max(1);
        blocks
            .iter()
            .step_by(stride)
            .flat_map(|&at| {
                let d = dct(&plane.block(at), &b);
                (1..64u8).map(move |i| (i, d[i as usize]))
            })
            .filter(|(_, c)| c.abs() > 0.5 * 4. * NOISE)
            .collect()
    };

    // Only the true block grid puts the coefficients on multiples of the steps
    let (offset, _) = (0..64)
        .filter_map(|o| {
            let offset = [o % 8, o / 8];
            let (_, score) = quality_of(&coefficients(&y, offset, 128), &LUMA, 1..=MAX_QUALITY)?;
            Some((offset, score))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    let (quality, _) = quality_of(
        &coefficients(&y, offset, usize::MAX),
        &LUMA,
        1..=MAX_QUALITY,
    )?;
    Some(Jpeg {
        offset,
        luma: table(&LUMA, quality),
        chroma: table(&CHROMA, quality),
    })
}

thread_local! {
    static CURRENT: std::cell::RefCell<Option<Jpeg>> = const { std::cell::RefCell::new(None) };
}

impl Jpeg {
    /// The spread of the compression's error per pixel of `rgb`, as an RGB distance in `0..1`:
    /// each kept frequency adds `q / sqrt(12)` spread over its block's 64 pixels (frequencies
    /// rounded to zero are left out), carried to RGB by the inverse color transform.
    pub fn spread(&self, rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
        let from = to_ycc().inverse();
        let b = basis();
        let mut variance = vec![0f32; width * height];
        for (p, plane) in planes(rgb, width, height, to_ycc()).iter().enumerate() {
            let steps = if p == 0 { &self.luma } else { &self.chroma };
            let weight = from.col(p).length_squared();
            for at in plane.blocks(self.offset) {
                let coefficients = dct(&plane.block(at), &b);
                let kept: f32 = coefficients
                    .iter()
                    .zip(steps)
                    .filter(|(c, q)| c.abs() > 0.5 * **q)
                    .map(|(_, q)| q * q / 12.)
                    .sum();
                let v = weight * kept / 64.;
                for i in 0..64 {
                    variance[(at[1] + i / 8) * width + at[0] + i % 8] += v;
                }
            }
        }
        variance.iter().map(|v| v.sqrt()).collect()
    }
}
