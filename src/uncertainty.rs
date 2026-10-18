//! The uncertainty channel: per pixel, how far a drawing may deviate from the raster and still
//! match the picture. The objective counts only the residual beyond it ([`excess`]); it adds the
//! noise, half an anti-aliased or blurred edge's step, and texture's grain, in quadrature.

use glam::Vec3;

use crate::edges::SIGMA;
use crate::translucency::raster::{Image, NOISE, di_zenzo, length_px};

/// The larger gradient scale, in multiples of [`SIGMA`]: a step blurred by `s` peaks at
/// `h / sqrt(2 pi (s^2 + sigma^2))`, so the ratio of the peaks at the two scales gives `s`
/// (so blurred edges such as niabot's sky are uncertain over their whole transition).
pub const BLUR_SCALE: f64 = 2.0;
/// The widest transition the two scales resolve, in multiples of [`SIGMA`].
pub const WIDEST: f64 = 2.0 * BLUR_SCALE;
/// The texture window's radius in the drawing's unit, never below 4 px: its median ignores a
/// stroke or edge covering less than half the window, while hatching and halftone fill it.
pub const TEXTURE_REACH: f64 = 4.0;
/// The median absolute response of the 3 x 3 Laplacian (`1 -2 1 / -2 4 -2 / 1 -2 1`, norm 6) to
/// Gaussian noise of unit spread: `0.6745 * 6`, the noise's spread per unit of the median.
const LAPLACIAN_MEDIAN: f64 = 0.6745 * 6.0;
/// How far grain of a measured spread may move a pixel: three spreads, the range of all but
/// 0.3 % of Gaussian noise, so clumps of grain a leaf keeps together are not drawn.
const GRAIN_RANGE: f64 = 3.0;

/// The uncertainty of each pixel and its parts, colour distances in RGB.
#[derive(Clone, Debug)]
pub struct Channel {
    /// the anti-aliased and blurred edges' part
    #[cfg_attr(not(test), allow(dead_code))] // read by the tests
    pub edge: Vec<f32>,
    /// the texture's part
    #[cfg_attr(not(test), allow(dead_code))] // read by the tests
    pub texture: Vec<f32>,
    /// all of it, the noise included: what a candidate has to explain beyond
    pub total: Vec<f32>,
}

impl Channel {
    /// The channel of an image given as the context's target (RGB in `0..=1`).
    pub fn of_target(target: &[Vec3], width: usize) -> Self {
        let img = Image::from_target(target, width);
        Self::of_image(&img)
    }

    pub fn of_image(img: &Image) -> Self {
        let edge = edge_part(img);
        let texture = grain_of(img);
        let total = edge
            .iter()
            .zip(&texture)
            .map(|(&e, &t)| ((NOISE * NOISE) as f32 + e * e + t * t).sqrt())
            .collect();
        Self {
            edge,
            texture,
            total,
        }
    }
}

/// The texture's part of the channel: how far grain and a JPEG's quantization may move each
/// pixel, as a colour distance.
pub fn grain_of(img: &Image) -> Vec<f32> {
    let reach = length_px(img.width, TEXTURE_REACH).max(TEXTURE_REACH as usize);
    let mut texture = texture_part(img, reach);
    // A JPEG's quantization is noise of the picture where the image shows it, at the
    // range grain covers (see [`compression`])
    if let Some(c) = compression(img) {
        for (t, c) in texture.iter_mut().zip(c) {
            *t = t.hypot(c);
        }
    }
    texture
}

/// Half the step each pixel's gradient comes from, over the transition's width: what an edge
/// leaves uncertain.
fn edge_part(img: &Image) -> Vec<f32> {
    let (fine, _) = di_zenzo(img, SIGMA);
    let (coarse, _) = di_zenzo(img, BLUR_SCALE * SIGMA);
    let root = (2.0 * std::f64::consts::PI).sqrt();
    let b2 = BLUR_SCALE * BLUR_SCALE;
    fine.iter()
        .zip(&coarse)
        .map(|(&m1, &m2)| {
            // the transition's width: sigma for a hard edge, wider where the coarse scale keeps
            // more of the peak, up to WIDEST; off the ridge (ratio at most one) just sigma
            let width = if m2 > 0.0 && m1 > m2 && m1 < BLUR_SCALE * m2 {
                let r2 = (m1 / m2).powi(2);
                ((b2 - 1.0) / (r2 - 1.0)).sqrt().min(WIDEST)
            } else {
                1.0
            } * SIGMA;
            (0.5 * m1 * width * root) as f32
        })
        .collect()
}

/// The spread a JPEG's quantization leaves in each pixel, times [`GRAIN_RANGE`], when the
/// pixels show the compression ([`crate::tracer::jpeg::detect`]): what the encoder rounded away is
/// unknown, so `emoji_fox_jpeg`'s 8 x 8 block steps are not drawn as shapes.
fn compression(img: &Image) -> Option<Vec<f32>> {
    let rgb: Vec<u8> = img
        .px
        .iter()
        .flat_map(|p| {
            p.to_array()
                .map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        })
        .collect();
    let jpeg = crate::tracer::jpeg::detect(&rgb, img.width, img.height)?;
    let spread = jpeg.spread(&rgb, img.width, img.height);
    Some(spread.iter().map(|s| GRAIN_RANGE as f32 * s).collect())
}

/// How far the grain of each pixel's neighbourhood may move it: per channel the median absolute
/// 3 x 3 Laplacian within `reach` over [`LAPLACIAN_MEDIAN`], times [`GRAIN_RANGE`], the channels
/// added as a colour distance.
fn texture_part(img: &Image, reach: usize) -> Vec<f32> {
    let (w, h) = (img.width, img.height);
    let mut total = vec![0f64; w * h];
    if w < 3 || h < 3 {
        return vec![0.; w * h];
    }
    for c in 0..3 {
        let ch = img.channel(c);
        // the response in levels, an integer for an 8-bit image (at most 16 levels of 255)
        let mut levels = vec![0u16; w * h];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let at = |dx: isize, dy: isize| {
                    ch[(y as isize + dy) as usize * w + (x as isize + dx) as usize]
                };
                let v = at(-1, -1) + at(1, -1) + at(-1, 1) + at(1, 1)
                    - 2.0 * (at(0, -1) + at(-1, 0) + at(1, 0) + at(0, 1))
                    + 4.0 * at(0, 0);
                levels[y * w + x] = (v.abs() * 255.0).round().min(BINS as f64 - 1.0) as u16;
            }
        }
        let median = sliding_median(&levels, w, h, reach);
        for (t, m) in total.iter_mut().zip(&median) {
            let s = GRAIN_RANGE * *m as f64 / 255.0 / LAPLACIAN_MEDIAN;
            *t += s * s;
        }
    }
    total.iter().map(|t| t.sqrt() as f32).collect()
}

/// Levels of the absolute Laplacian of an 8-bit image: `16 * 255` and one.
pub const BINS: usize = 16 * 255 + 1;

/// The median of `v` over the square of radius `r` around each pixel (cut by the frame), by a
/// histogram slid along each row (Huang's): the window's count below the running median is
/// kept as columns leave and enter.
pub fn sliding_median(v: &[u16], w: usize, h: usize, r: usize) -> Vec<u16> {
    let mut out = vec![0u16; w * h];
    let mut hist = vec![0u32; BINS];
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(r), (y + r).min(h - 1));
        hist.iter_mut().for_each(|c| *c = 0);
        let mut count = 0u32;
        // the running median and the count strictly below it, kept as columns enter and leave
        let (mut med, mut below) = (0usize, 0u32);
        for x in 0..=r.min(w - 1) {
            for yy in y0..=y1 {
                hist[v[yy * w + x] as usize] += 1;
                count += 1;
            }
        }
        for x in 0..w {
            if x > 0 {
                if x + r < w {
                    for yy in y0..=y1 {
                        let b = v[yy * w + x + r] as usize;
                        hist[b] += 1;
                        count += 1;
                        below += u32::from(b < med);
                    }
                }
                if x > r {
                    for yy in y0..=y1 {
                        let b = v[yy * w + x - r - 1] as usize;
                        hist[b] -= 1;
                        count -= 1;
                        below -= u32::from(b < med);
                    }
                }
            }
            // the median is the smallest level with more than half the window at or below it
            let half = count / 2;
            while below > half {
                med -= 1;
                below -= hist[med];
            }
            while below + hist[med] <= half {
                below += hist[med];
                med += 1;
            }
            out[y * w + x] = med as u16;
        }
    }
    out
}

/// The per-segment price, in squared error, for whether a planned shape is drawn: one pixel
/// at the noise, only a tie break between candidates removing the same residual.
pub const PRICE: f64 = NOISE * NOISE;

/// Puts `ctx` under the tolerance: each pixel may deviate by `tolerance` times its uncertainty,
/// and a planned shape is drawn by the residual beyond that at [`PRICE`] per segment. How a
/// shape is described keeps the plain error and the detail's price, all scaled by `tolerance^2`.
pub fn apply(ctx: &mut crate::Context, tolerance: f64) {
    let channel = Channel::of_target(&ctx.target, ctx.width);
    let t = tolerance as f32;
    ctx.tolerance = Some(std::sync::Arc::new(
        channel.total.iter().map(|u| u * t).collect(),
    ));
    let paths = if ctx.segment_cost > 0. {
        ctx.path_cost / ctx.segment_cost
    } else {
        0.
    };
    let scale = tolerance * tolerance;
    ctx.segment_cost *= scale;
    ctx.path_cost *= scale;
    ctx.accept = Some((PRICE * scale, paths * PRICE * scale));
}

/// Puts `c`, the context beneath a layer peeled off `parent` (or a `window` of it), under
/// `parent`'s tolerance and price, divided inside the layer's `mask` by the share `k` that
/// shows through. The `rim` pixels accept anything, as the peel invents their colours.
pub fn inherit(
    c: &mut crate::Context,
    parent: &crate::Context,
    window: crate::Window,
    peel: Option<(&[bool], f32)>,
    rim: Option<&[bool]>,
) {
    let Some(t) = &parent.tolerance else {
        return;
    };
    let scale = |i: usize| match peel {
        Some((mask, k)) if mask[i] && k > 0. => 1. / k,
        _ => 1.,
    };
    let inherited = window
        .pixels(parent.width)
        .enumerate()
        .map(|(j, i)| {
            if rim.is_some_and(|r| r[j]) {
                ANYTHING
            } else {
                t[i] * scale(i)
            }
        })
        .collect::<Vec<f32>>();
    c.tolerance = Some(std::sync::Arc::new(inherited));
    c.segment_cost = parent.segment_cost;
    c.path_cost = parent.path_cost;
    c.accept = parent.accept;
}

/// A region's tolerances, sorted, for the residual beyond them of a deviation even over the
/// region: `sum over i of max(0, d - t_i)^2`, in a binary search and prefix sums.
#[derive(Clone, Debug, Default)]
pub struct Spread {
    sorted: Vec<f32>,
    /// prefix sums of the tolerances and of their squares, one longer than `sorted`
    sum: Vec<f64>,
    sum2: Vec<f64>,
}

impl Spread {
    pub fn new(mut t: Vec<f32>) -> Self {
        t.sort_by(f32::total_cmp);
        let (mut sum, mut sum2) = (vec![0.], vec![0.]);
        for &v in &t {
            sum.push(sum.last().unwrap() + v as f64);
            sum2.push(sum2.last().unwrap() + (v as f64) * (v as f64));
        }
        Self {
            sorted: t,
            sum,
            sum2,
        }
    }

    /// The squared residual beyond the tolerances of a deviation `d` at every pixel.
    pub fn excess(&self, d: f64) -> f64 {
        let k = self.sorted.partition_point(|&t| (t as f64) < d);
        (k as f64 * d * d - 2. * d * self.sum[k] + self.sum2[k]).max(0.)
    }

    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }
}

/// The tolerance of a pixel that may show anything: the distance from black to white.
pub(crate) const ANYTHING: f32 = 1.732_050_8;

/// The squared residual of a pixel beyond its tolerance `t`: the part of a colour difference
/// (given squared, `d2`) the picture does not account for.
pub fn excess(d2: f32, t: f32) -> f32 {
    if t <= 0. {
        return d2;
    }
    let over = d2.sqrt() - t;
    if over > 0. { over * over } else { 0. }
}

/// [`excess`] in doubles.
pub fn excess64(d2: f64, t: f64) -> f64 {
    if t <= 0. {
        return d2;
    }
    let over = d2.sqrt() - t;
    if over > 0. { over * over } else { 0. }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize, f: impl Fn(usize, usize) -> f64) -> Image {
        let px = (0..w * h)
            .map(|i| {
                let v = (f(i % w, i / w) * 255.0).round() / 255.0;
                glam::DVec3::splat(v)
            })
            .collect();
        Image {
            width: w,
            height: h,
            px,
        }
    }

    /// A flat field keeps the noise alone; the pixel an anti-aliased step runs through is
    /// uncertain by about half the step; a one-pixel line's centre keeps about the noise.
    #[test]
    fn edges_lines_and_flat() {
        let (w, h) = (40, 40);
        let gray = image(w, h, |_, _| 0.5);
        let c = Channel::of_image(&gray);
        let noise = NOISE as f32;
        assert!((c.total[20 * w + 20] - noise).abs() < 1e-4);
        // a white-to-black step through the middle of column 20 (columns 0..20 white)
        let step = image(w, h, |x, _| match x {
            0..20 => 1.0,
            20 => 0.5,
            _ => 0.0,
        });
        let c = Channel::of_image(&step);
        let at = c.edge[20 * w + 20];
        // half the step as a colour distance (sqrt(3) for white to black), within the
        // anti-aliasing's smoothing of a box-sampled step
        let half = 0.5 * 3f32.sqrt();
        assert!(at > 0.8 * half && at < 1.1 * half, "{at} against {half}");
        assert!(c.edge[20 * w + 26] < 0.05 * half, "{}", c.edge[20 * w + 26]);
        // a black one-pixel line on white
        let line = image(w, h, |x, _| if x == 20 { 0.0 } else { 1.0 });
        let c = Channel::of_image(&line);
        assert!(c.edge[20 * w + 20] < 0.02, "{}", c.edge[20 * w + 20]);
        assert!(c.texture[20 * w + 20] < 1e-6, "{}", c.texture[20 * w + 20]);
    }

    /// A checkerboard is texture throughout; a single line in a flat field is not.
    #[test]
    fn texture_fills_dense_detail_only() {
        let (w, h) = (40, 40);
        let checker = image(w, h, |x, y| if (x + y) % 2 == 0 { 0.2 } else { 0.8 });
        let c = Channel::of_image(&checker);
        assert!(c.texture[20 * w + 20] > 0.1, "{}", c.texture[20 * w + 20]);
    }

    /// A disc on a ground, anti-aliased, with a thin dark line across the ground or not.
    fn disc(size: usize, line: bool) -> Vec<u8> {
        let mut rgb = vec![0u8; 3 * size * size];
        for y in 0..size {
            for x in 0..size {
                // coverage of the disc by 4 x 4 samples
                let mut cover = 0.;
                for sy in 0..4 {
                    for sx in 0..4 {
                        let (u, v) = (
                            x as f64 + (sx as f64 + 0.5) / 4.,
                            y as f64 + (sy as f64 + 0.5) / 4.,
                        );
                        let r = (u - 0.45 * size as f64).hypot(v - 0.5 * size as f64);
                        cover += f64::from(u8::from(r < 0.25 * size as f64)) / 16.;
                    }
                }
                let ground = [230., 225., 210.];
                let ink = [40., 90., 170.];
                let mut c: Vec<f64> = (0..3)
                    .map(|k| ground[k] + cover * (ink[k] - ground[k]))
                    .collect();
                if line && x == size - size / 8 {
                    c = vec![60., 50., 40.];
                }
                for k in 0..3 {
                    rgb[3 * (y * size + x) + k] = c[k].round() as u8;
                }
            }
        }
        rgb
    }

    /// A flat picture of one disc keeps that one shape on its ground, and a thin line whose
    /// residual exceeds the noise is drawn.
    #[test]
    fn one_shape_kept_and_a_thin_line_drawn() {
        use crate::plan;
        let size = 96;
        let trace = |rgb: &[u8]| {
            let detail = crate::tracer::auto_detail(rgb, size, size);
            let options = plan::Options::default();
            plan::trace(rgb, size, size, detail, options, |_| ())
        };
        let plain = trace(&disc(size, false));
        assert_eq!(plain.shapes.len(), 2, "the ground and the disc");
        let lined = trace(&disc(size, true));
        let drawn = lined.shapes.len() + usize::from(!lined.marks.is_empty());
        assert!(
            drawn > 2,
            "the line is drawn: {} shapes",
            lined.shapes.len()
        );
    }

    #[test]
    fn spread_sums_each_pixel() {
        let t = vec![0.1, 0.5, 0.2, 0.9];
        let s = Spread::new(t.clone());
        for d in [0.0, 0.15, 0.3, 0.7, 1.2] {
            let direct: f64 = t.iter().map(|&t| excess64(d * d, t as f64)).sum();
            assert!(
                (s.excess(d) - direct).abs() < 1e-9,
                "{d}: {} {direct}",
                s.excess(d)
            );
        }
    }

    #[test]
    fn excess_is_the_residual_beyond() {
        assert_eq!(excess(0.04, 0.3), 0.);
        assert!((excess(0.25, 0.3) - 0.04).abs() < 1e-6);
        assert_eq!(excess(0.25, 0.), 0.25);
    }

    #[test]
    fn median_matches_sorting() {
        let (w, h) = (17, 11);
        let v: Vec<u16> = (0..w * h).map(|i| ((i * 7919) % 23) as u16).collect();
        let m = sliding_median(&v, w, h, 3);
        for y in 0..h {
            for x in 0..w {
                let mut s = Vec::new();
                for yy in y.saturating_sub(3)..=(y + 3).min(h - 1) {
                    for xx in x.saturating_sub(3)..=(x + 3).min(w - 1) {
                        s.push(v[yy * w + xx]);
                    }
                }
                s.sort();
                assert_eq!(m[y * w + x], s[s.len() / 2], "at {x} {y}");
            }
        }
    }
}
