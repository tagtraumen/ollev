//! A line's paint and width fitted over the composite, and its gain.

use super::*;

// ----------------------------------------------------------------------------- fitting

/// The squared error of the composite against the image over a window's pixels, before
/// and after drawing `paint` (per pixel: alpha, colour); with a tolerance per pixel, the
/// residual beyond it (see [`crate::uncertainty::excess`]).
pub(super) fn gain_of(
    img: &Image,
    comp: &[DVec3],
    win: Win,
    alpha: &[f64],
    colour: &dyn Fn(usize) -> DVec3,
    tol: Option<&[f32]>,
) -> f64 {
    let err = |p: usize, d: DVec3| {
        crate::uncertainty::excess64(d.length_squared(), tol.map_or(0., |t| t[p] as f64))
    };
    let mut g = 0.0;
    for (i, &a) in alpha.iter().enumerate() {
        if a <= 0.0 {
            continue;
        }
        let p = win.at(i, img.width);
        let d = comp[p];
        let n = d + a * (colour(i) - d);
        g += err(p, d - img.px[p]) - err(p, n - img.px[p]);
    }
    g
}

/// The colour and opacity of a stroke over the composite from its coverage: per covered
/// pixel o = d + a cov (C - d), linear in (a C, a); least squares over the channels, a in
/// (0, 1], the colour refitted at the clipped opacity.
pub(super) fn fit_paint(
    img: &Image,
    comp: &[DVec3],
    win: Win,
    cov: &[f64],
) -> Option<(DVec3, f64)> {
    // normal equations of the unknowns (p.x, p.y, p.z, a): rows w e_c and -w d_c
    let mut ata = [[0.0f64; 4]; 4];
    let mut atb = [0.0f64; 4];
    let mut n = 0;
    for (i, &w) in cov.iter().enumerate() {
        if w <= 0.05 {
            continue;
        }
        n += 1;
        let p = win.at(i, img.width);
        let (o, d) = (img.px[p], comp[p]);
        for c in 0..3 {
            let mut row = [0.0; 4];
            row[c] = w;
            row[3] = -w * d[c];
            let y = o[c] - d[c];
            for r in 0..4 {
                atb[r] += row[r] * y;
                for s in 0..4 {
                    ata[r][s] += row[r] * row[s];
                }
            }
        }
    }
    if n < 3 {
        return None;
    }
    let sol = solve4(ata, atb)?;
    let a = sol[3].clamp(0.0, 1.0);
    if a < 1e-3 {
        return None;
    }
    // the colour at the clipped opacity: per channel (o - (1 - a w) d) against a w
    let (mut num, mut den) = (DVec3::ZERO, 0.0);
    for (i, &w) in cov.iter().enumerate() {
        if w <= 0.05 {
            continue;
        }
        let p = win.at(i, img.width);
        let aw = a * w;
        num += (img.px[p] - (1.0 - aw) * comp[p]) * aw;
        den += aw * aw;
    }
    let raw = num / den.max(1e-12);
    let c = raw.clamp(DVec3::ZERO, DVec3::ONE);
    // a fit whose paint (a times the colour) lies outside the gamut by more than the noise
    // corrects the composite's error (a faint seam of the beneath) rather than drawing a line
    if a * (raw - c).abs().max_element() > NOISE {
        return None;
    }
    Some((c, a))
}

pub(super) fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for col in 0..4 {
        let piv = (col..4).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in 0..4 {
            if r != col {
                let f = a[r][col] / a[col][col];
                let pivot_row = a[col];
                for (x, p) in a[r].iter_mut().zip(pivot_row).skip(col) {
                    *x -= f * p;
                }
                b[r] -= f * b[col];
            }
        }
    }
    Some([
        b[0] / a[0][0],
        b[1] / a[1][1],
        b[2] / a[2][2],
        b[3] / a[3][3],
    ])
}

/// A fitted stroke before it is drawn.
pub(super) struct Fit {
    pub(super) cubics: Vec<Bezier>,
    pub(super) flat: Vec<DVec2>,
    pub(super) width: f64,
    pub(super) colour: DVec3,
    pub(super) opacity: f64,
    pub(super) gain: f64,
}

/// A stroke along a polyline: of widths 1 to 3 px and shifts across of up to a pixel (the
/// centre is read to the grid), colour and opacity fitted against the composite at each, the
/// one that removes the most error.
pub(super) fn fit_line(
    img: &Image,
    comp: &[DVec3],
    pts: &[DVec2],
    tol: Option<&[f32]>,
) -> Option<Fit> {
    let u = unit_of(img.width);
    if pts.len() < 2 {
        return None;
    }
    let nrm = normals(pts);
    let mut best: Option<Fit> = None;
    for sh in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let moved: Vec<DVec2> = pts
            .iter()
            .zip(&nrm)
            .map(|(p, n)| *p + sh * u * *n)
            .collect();
        let cubics = fit_cubics(&moved, FIT_TOL * u);
        let flat = flatten(&cubics);
        for wd in [1.0, 1.5, 2.0, 2.5, 3.0] {
            let width = wd * u;
            let win = Win::around(&flat, width / 2.0 + 2.0, img.width, img.height);
            if win.len() == 0 {
                continue;
            }
            let cov = coverage(&flat, width, win);
            let Some((c, a)) = fit_paint(img, comp, win, &cov) else {
                continue;
            };
            let alpha: Vec<f64> = cov.iter().map(|w| a * w).collect();
            let gain = gain_of(img, comp, win, &alpha, &|_| c, tol);
            if best.as_ref().is_none_or(|b| gain > b.gain) {
                best = Some(Fit {
                    cubics: cubics.clone(),
                    flat: flat.clone(),
                    width,
                    colour: c,
                    opacity: a,
                    gain,
                });
            }
        }
    }
    best
}

/// A line's polyline as strokes: pieces of 8 px fitted on their own (those removing no error
/// dropped), then neighbours merged while one stroke saves more than the two apart, each
/// paying `cost` per cubic, so a line splits where its colour changes.
pub(super) fn stroke_runs(
    img: &Image,
    comp: &[DVec3],
    pts: &[DVec2],
    cost: f64,
    tol: Option<&[f32]>,
) -> Vec<Fit> {
    let step = length_px(img.width, 8.0).max(2);
    let n = pts.len();
    if n < 2 {
        return Vec::new();
    }
    let mut pieces: Vec<(usize, usize)> = (0..n - 1)
        .step_by(step)
        .map(|a| (a, (a + step + 1).min(n)))
        .collect();
    if pieces.len() > 1 && pieces[pieces.len() - 1].1 - pieces[pieces.len() - 1].0 < step / 2 + 1 {
        let last = pieces.pop().unwrap();
        pieces.last_mut().unwrap().1 = last.1;
    }
    let net = |f: &Fit| f.gain - cost * f.cubics.len() as f64;
    let mut runs: Vec<(usize, usize, Fit)> = pieces
        .into_iter()
        .take_while(|_| !crate::progress::poll())
        .filter_map(|(a, b)| fit_line(img, comp, &pts[a..b], tol).map(|f| (a, b, f)))
        .filter(|r| r.2.gain > 0.0)
        .collect();
    loop {
        let mut best: Option<(f64, usize, Fit)> = None;
        for i in 0..runs.len().saturating_sub(1) {
            if runs[i].1 - 1 != runs[i + 1].0 {
                continue;
            }
            let Some(f) = fit_line(img, comp, &pts[runs[i].0..runs[i + 1].1], tol) else {
                continue;
            };
            let here = net(&runs[i].2) + net(&runs[i + 1].2);
            let d = net(&f) - here;
            if f.gain > 0.0 && d > 0.0 && best.as_ref().is_none_or(|b| d > b.0) {
                best = Some((d, i, f));
            }
        }
        let Some((_, i, f)) = best else {
            break;
        };
        let (a, b) = (runs[i].0, runs[i + 1].1);
        runs.splice(i..i + 2, [(a, b, f)]);
    }
    runs.into_iter().map(|r| r.2).collect()
}

/// What a fitted stroke saves over the composite as it is, with what it does for the
/// output's edges under the edge term.
pub(super) fn stroke_gain(img: &Image, comp: &[DVec3], f: &Fit, opt: &Options) -> f64 {
    let win = Win::around(&f.flat, f.width / 2.0 + 2.0, img.width, img.height);
    let cov = coverage(&f.flat, f.width, win);
    let alpha: Vec<f64> = cov.iter().map(|w| f.opacity * w).collect();
    gain_of(img, comp, win, &alpha, &|_| f.colour, opt.tol())
        + opt.edge_credit(img, comp, win, &alpha, f.colour)
}

/// Draws a fitted stroke into the composite; returns the pixels it covers.
pub(super) fn draw_stroke(img: &Image, comp: &mut [DVec3], f: &Fit) -> Vec<usize> {
    let win = Win::around(&f.flat, f.width / 2.0 + 2.0, img.width, img.height);
    let cov = coverage(&f.flat, f.width, win);
    let alpha: Vec<f64> = cov.iter().map(|w| f.opacity * w).collect();
    let mut covered = Vec::new();
    for (i, &a) in alpha.iter().enumerate() {
        if a > 0.0 {
            let p = win.at(i, img.width);
            comp[p] += a * (f.colour - comp[p]);
            covered.push(p);
        }
    }
    covered
}
