//! The layer's lines: thin colour lines, hem lines and their chains.

use glam::DVec2;

use super::{
    Conn, Image, NOISE, crack_loops, label, length_px, map1, normals, resample, skeletonize,
    ridge_radii, smooth_loop, smooth_open, trace_chains, unit_of,
};

/// The colour line test's four directions as (dy, dx): the pair of pixels it compares lies
/// along it, across the line it finds.
pub const DIRS: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

/// Thin lines of any colour: per pixel the most it lies off the colour segment between the
/// pixels r away along one of four directions, and that direction. An anti-aliased step lies
/// on the segment, a line of a third colour does not, darker or lighter (as pale hems).
pub fn colour_lines(img: &Image) -> (Vec<f64>, Vec<u8>) {
    colour_lines_at(img, unit_of(img.width))
}

type ColourLines = (
    (u64, usize, usize, (u64, u64)),
    std::rc::Rc<(Vec<f64>, Vec<u8>)>,
);

thread_local! {
    /// The colour lines found last: the edge term, the linework and its ridges read the same
    /// picture's at the same unit.
    static COLOUR_LINES: std::cell::RefCell<Vec<ColourLines>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// [`colour_lines`] at the unit `u`, for a window of an image whose unit is the whole's.
pub fn colour_lines_at(img: &Image, u: f64) -> (Vec<f64>, Vec<u8>) {
    let key = (
        u.to_bits(),
        img.width,
        img.height,
        super::super::raster::digest(&img.px),
    );
    (*super::super::raster::remembered(&COLOUR_LINES, key, || colour_lines_of(img, u))).clone()
}

fn colour_lines_of(img: &Image, u: f64) -> (Vec<f64>, Vec<u8>) {
    let (w, h) = (img.width as isize, img.height as isize);
    let radii = ridge_radii(u);
    let n = img.len();
    let mut best = vec![0.0f64; n];
    let mut dir = vec![0u8; n];
    let clamp = |y: isize, x: isize| (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize;
    for &r in &radii {
        for (j, &(dy, dx)) in DIRS.iter().enumerate() {
            for y in 0..h {
                for x in 0..w {
                    let p = (y * w + x) as usize;
                    let a = img.px[clamp(y + dy * r, x + dx * r)];
                    let b = img.px[clamp(y - dy * r, x - dx * r)];
                    let o = img.px[p];
                    let ab = b - a;
                    let t = ((o - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
                    let d = (o - (a + t * ab)).abs().max_element();
                    if d > best[p] {
                        best[p] = d;
                        dir[p] = j as u8;
                    }
                }
            }
        }
    }
    (best, dir)
}

/// The centre pixels of lines: at least as strong as both neighbours across the line (the
/// direction its test found), kept by hysteresis (weak pixels, 8-connected to a strong
/// one), the edge detector's rule.
pub fn thin_lines(
    strength: &[f64],
    dir: &[u8],
    width: usize,
    height: usize,
    lo: f64,
    hi: f64,
) -> Vec<bool> {
    let (w, h) = (width as isize, height as isize);
    let at = |y: isize, x: isize| strength[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize];
    let mut weak = vec![false; strength.len()];
    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) as usize;
            let (dy, dx) = DIRS[dir[p] as usize];
            let s = strength[p];
            weak[p] = s > lo && s >= at(y + dy, x + dx) && s >= at(y - dy, x - dx);
        }
    }
    hysteresis(&weak, strength, hi, width, height)
}

pub(super) fn hysteresis(
    weak: &[bool],
    strength: &[f64],
    hi: f64,
    width: usize,
    height: usize,
) -> Vec<bool> {
    let (lab, nl) = label(weak, width, height, Conn::Eight);
    let mut keep = vec![false; nl + 1];
    for i in 0..weak.len() {
        if weak[i] && strength[i] > hi {
            keep[lab[i] as usize] = true;
        }
    }
    keep[0] = false;
    lab.iter().map(|&l| keep[l as usize]).collect()
}

/// Maximal runs of true flags as index ranges, gaps of up to `gap` false ones bridged.
pub(super) fn runs(flags: &[bool], gap: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let (mut start, mut last) = (None::<usize>, 0usize);
    for (i, &f) in flags.iter().enumerate() {
        if !f {
            continue;
        }
        match start {
            None => start = Some(i),
            Some(s) if i - last - 1 > gap => {
                out.push((s, last + 1));
                start = Some(i);
            }
            _ => {}
        }
        last = i;
    }
    if let Some(s) = start {
        out.push((s, last + 1));
    }
    out
}

pub(super) fn median_filter(v: &[f64], size: usize) -> Vec<f64> {
    let r = size / 2;
    (0..v.len())
        .map(|i| {
            let mut win: Vec<f64> = (i.saturating_sub(r)..(i + r + 1).min(v.len()))
                .map(|j| v[j])
                .collect();
            win.sort_by(f64::total_cmp);
            win[win.len() / 2]
        })
        .collect()
}

pub(super) fn gauss1d(v: &[f64], s: f64) -> Vec<f64> {
    let r = (4.0 * s).ceil() as isize;
    let n = v.len() as isize;
    (0..n)
        .map(|i| {
            let (mut a, mut t) = (0.0, 0.0);
            for j in -r..=r {
                let wj = (-0.5 * (j * j) as f64 / (s * s)).exp();
                a += wj * v[(i + j).clamp(0, n - 1) as usize];
                t += wj;
            }
            a / t
        })
        .collect()
}

/// Hems, lines the layer's edge carries: along the region's outline, the offset across it
/// (within 3 px) where the line strength peaks above 6 NOISE; runs at least 5 px long, gaps
/// of 2 px bridged (anti-aliasing breaks diagonals), offsets median-filtered and smoothed.
pub fn hem_lines(
    region: &[bool],
    strength: &[f64],
    width: usize,
    height: usize,
) -> Vec<Vec<DVec2>> {
    let u = unit_of(width);
    let reach = 3.0 * u;
    let steps = (2.0 * reach / 0.5).round() as usize;
    let offs: Vec<f64> = (0..=steps)
        .map(|i| -reach + i as f64 * 2.0 * reach / steps as f64)
        .collect();
    let mut out = Vec::new();
    for lp in crack_loops(region, width, height) {
        let lp = smooth_loop(&lp, 1.0);
        let p = resample(&lp, 1.0);
        if (p.len() as f64) < 5.0 * u {
            continue;
        }
        let nrm = normals(&p);
        let mut best = vec![0.0; p.len()];
        let mut off = vec![0.0; p.len()];
        for i in 0..p.len() {
            for &o in &offs {
                let q = p[i] + o * nrm[i];
                // the plane's index coordinates are the SVG ones less half a pixel
                let s = map1(strength, width, height, q.x - 0.5, q.y - 0.5);
                if s > best[i] {
                    best[i] = s;
                    off[i] = o;
                }
            }
        }
        let strong: Vec<bool> = best.iter().map(|&s| s > 6.0 * NOISE).collect();
        for (a, b) in runs(&strong, length_px(width, 2.0)) {
            if ((b - a) as f64) < 5.0 * u {
                continue;
            }
            let o = median_filter(&off[a..b], length_px(width, 3.0) | 1);
            let o = gauss1d(&o, u);
            out.push((a..b).map(|i| p[i] + o[i - a] * nrm[i]).collect());
        }
    }
    out
}

/// Chains of a thin mask's skeleton (cut at its junctions), as polylines through the
/// pixel centres, smoothed, at least 5 px long (at 330).
pub fn line_chains(mask: &[bool], width: usize, height: usize, min_px: f64) -> Vec<Vec<DVec2>> {
    let sk = skeletonize(mask, width, height);
    trace_chains(&sk, width, height, length_px(width, min_px).max(3))
        .into_iter()
        .map(|c| {
            let p: Vec<DVec2> = c
                .into_iter()
                .map(|(y, x)| DVec2::new(x as f64 + 0.5, y as f64 + 0.5))
                .collect();
            smooth_open(&p)
        })
        .collect()
}
