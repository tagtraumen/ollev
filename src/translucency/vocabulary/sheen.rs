//! The white over the layer as blurred lenses clipped to it.

use glam::{DVec2, DVec3};

use super::paint::gain_of;
use super::{
    Bezier, FIT_TOL, Image, Lens, Mark, NOISE, Options, Win, fit_cubics, flatten,
    gaussian_filter, hysteresis, line_chains, map1, normals, unit_of,
};

/// The opacity of a white over the composite that brings it to the image per pixel:
/// o = (1 - t) d + t, t by least squares over the channels, in [0, 1], 0 off the region.
pub fn white_target(img: &Image, comp: &[DVec3], region: &[bool]) -> Vec<f64> {
    (0..img.len())
        .map(|i| {
            if !region[i] {
                return 0.0;
            }
            let v = DVec3::ONE - comp[i];
            let t = (img.px[i] - comp[i]).dot(v) / v.length_squared().max(1e-9);
            t.clamp(0.0, 1.0)
        })
        .collect()
}

/// Bright ridges of a plane at scale s: the strength the most negative Hessian eigenvalue
/// of its smoothing times s squared, the centres those at least as strong as their
/// neighbours across the ridge.
pub fn bright_ridges(field: &[f64], width: usize, height: usize, s: f64) -> (Vec<f64>, Vec<bool>) {
    let f = gaussian_filter(field, width, height, s, 0, 0);
    let (w, h) = (width as isize, height as isize);
    let at = |y: isize, x: isize| f[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize];
    let n = field.len();
    let mut st = vec![0.0; n];
    let mut v = vec![DVec2::ZERO; n];
    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) as usize;
            let fxx = at(y, x + 1) - 2.0 * at(y, x) + at(y, x - 1);
            let fyy = at(y + 1, x) - 2.0 * at(y, x) + at(y - 1, x);
            let fxy =
                (at(y + 1, x + 1) - at(y + 1, x - 1) - at(y - 1, x + 1) + at(y - 1, x - 1)) / 4.0;
            let tr = fxx + fyy;
            let det = fxx * fyy - fxy * fxy;
            let lmin = tr / 2.0 - (tr * tr / 4.0 - det).max(0.0).sqrt();
            let mut e = DVec2::new(fxy, lmin - fxx);
            if e.length() < 1e-12 {
                e = DVec2::new(lmin - fyy, fxy);
            }
            v[p] = e / e.length().max(1e-12);
            st[p] = (-lmin).max(0.0) * s * s;
        }
    }
    let centre = (0..n)
        .map(|p| {
            let (x, y) = ((p % width) as f64, (p / width) as f64);
            let a = map1(&st, width, height, x + v[p].x, y + v[p].y);
            let b = map1(&st, width, height, x - v[p].x, y - v[p].y);
            st[p] > 0.0 && st[p] >= a && st[p] >= b
        })
        .collect();
    (st, centre)
}

/// The outline of a lens along a centreline: its width at the middle, tapering to points at
/// both ends as a half sine of the arc length, fitted to the lines' tolerance.
pub fn lens_outline(centre: &[DVec2], width: f64, tol: f64) -> Vec<Bezier> {
    let mut arc = vec![0.0];
    for s in centre.windows(2) {
        arc.push(arc[arc.len() - 1] + (s[1] - s[0]).length());
    }
    let total = arc[arc.len() - 1].max(1e-9);
    let n = normals(centre);
    let half = |i: usize| 0.5 * width * (std::f64::consts::PI * arc[i] / total).sin();
    let left: Vec<DVec2> = (0..centre.len())
        .map(|i| centre[i] + half(i) * n[i])
        .collect();
    let right: Vec<DVec2> = (0..centre.len())
        .rev()
        .map(|i| centre[i] - half(i) * n[i])
        .collect();
    let mut outline = fit_cubics(&left, tol);
    outline.extend(fit_cubics(&right, tol));
    outline
}

/// A lens's alpha over a window: its outline's coverage blurred by `sigma`, as the SVG's
/// `feGaussianBlur` draws it, then cut by `clip` (the layer's coverage per image pixel).
pub(super) fn lens_alpha(
    outline: &[Bezier],
    sigma: f64,
    clip: &dyn Fn(usize) -> f64,
    img_w: usize,
    img_h: usize,
) -> (Win, Vec<f64>) {
    let flat = flatten(outline);
    let win = Win::around(&flat, 3.0 * sigma + 2.0, img_w, img_h);
    let o = DVec2::new(win.x0 as f64, win.y0 as f64);
    let points = outline
        .iter()
        .flat_map(|c| [c[0], c[1], c[2]])
        .map(|p| (p - o).as_vec2())
        .collect();
    let path = crate::render::Path {
        subpaths: vec![crate::render::Subpath { points }],
        color: [1.0; 4],
        gradient: None,
        fill_rule: crate::render::FillRule::EvenOdd,
        blur: sigma as f32,
    };
    let polys = crate::render::scene::flatten(&path);
    let b = crate::render::blur::Blurred::new(polys, path.fill_rule, path.blur, win.w(), win.h());
    let g = (0..win.len())
        .map(|i| f64::from(b.at(i % win.w(), i / win.w())) * clip(win.at(i, img_w)))
        .collect();
    (win, g)
}

/// The opacity of a white of alpha g over the composite, by least squares, and its gain.
pub(super) fn white_fit(
    img: &Image,
    comp: &[DVec3],
    win: Win,
    g: &[f64],
    tol: Option<&[f32]>,
) -> (f64, f64) {
    let (mut num, mut den) = (0.0, 0.0);
    for (i, &gi) in g.iter().enumerate() {
        if gi <= 0.01 {
            continue;
        }
        let p = win.at(i, img.width);
        let v = gi * (DVec3::ONE - comp[p]);
        num += v.dot(img.px[p] - comp[p]);
        den += v.length_squared();
    }
    let a = (num / den.max(1e-12)).clamp(0.0, 1.0);
    let alpha = white_alpha(g, a);
    (a, gain_of(img, comp, win, &alpha, &|_| DVec3::ONE, tol))
}

/// The alpha of a white at opacity `a` over a lens's alpha `g`, nothing where `g` is faint.
fn white_alpha(g: &[f64], a: f64) -> Vec<f64> {
    g.iter()
        .map(|gi| if *gi > 0.01 { a * gi } else { 0.0 })
        .collect()
}

impl Lens {
    /// The lens as the keeping rule sees it, at `gain`.
    fn mark(&self, gain: f64) -> Mark<'_> {
        Mark {
            cubics: &self.outline,
            gain,
        }
    }
}

/// The white as blurred white lenses along the bright ridges of the white each pixel needs
/// over the composite, clipped to the layer (`clip`, its coverage per pixel), each with a
/// width, blur and opacity fitted; those `opt` keeps are drawn, largest savings first.
pub async fn lenses(
    img: &Image,
    comp: &mut [DVec3],
    region: &[bool],
    clip: &(dyn Fn(usize) -> f64 + Sync),
    above: Option<&crate::translucency::Above>,
    opt: &Options,
) -> Vec<Lens> {
    let (w, h) = (img.width, img.height);
    let u = unit_of(w);
    // where the pin's full fit read the white's opacity per pixel, a lens is priced on it
    // (squared, over three channels): over a pale layer a highlight is faint in the composite,
    // while in the opacity a missing highlight is its whole value
    let alpha_gain = |win: Win, g: &[f64], a: f64| -> Option<f64> {
        let above = above?;
        let mut gain = 0.0;
        for (i, &gi) in g.iter().enumerate() {
            if gi <= 0.01 {
                continue;
            }
            let p = win.at(i, img.width);
            let t = f64::from(above.white.get(p).copied().unwrap_or(0.0));
            gain += t * t - (t - a * gi) * (t - a * gi);
        }
        Some(3.0 * gain)
    };
    // what a lens does for the output's edges under the edge term
    let credit = |comp: &[DVec3], l: &Lens| {
        if opt.edge.is_none() {
            return 0.0;
        }
        let (win, g) = lens_alpha(&l.outline, l.sigma, clip, w, h);
        let (a, _) = white_fit(img, comp, win, &g, opt.tol());
        opt.edge_credit(img, comp, win, &white_alpha(&g, a), DVec3::ONE)
    };
    let tgt = white_target(img, comp, region);
    let (st, centre) = bright_ridges(&tgt, w, h, 2.0 * u);
    let weak: Vec<bool> = (0..st.len())
        .map(|i| centre[i] && region[i] && st[i] > 3.0 * NOISE)
        .collect();
    let lines = hysteresis(&weak, &st, 6.0 * NOISE, w, h);
    let mut cands = Vec::new();
    for pts in line_chains(&lines, w, h, 4.0) {
        crate::progress::poll().await;
        if pts.len() < 2 {
            continue;
        }
        let cubics = fit_cubics(&pts, FIT_TOL * u);
        let flat = flatten(&cubics);
        let mut best: Option<Lens> = None;
        for wd in [1.0, 2.0, 3.0, 4.5] {
            let outline = lens_outline(&flat, wd * u, FIT_TOL * u);
            for sg in [0.5, 1.0, 2.0] {
                let (win, g) = lens_alpha(&outline, sg * u, clip, w, h);
                let (a, gain) = white_fit(img, comp, win, &g, opt.tol());
                let gain = alpha_gain(win, &g, a).unwrap_or(gain);
                if a > 0.0 && best.as_ref().is_none_or(|b| gain > b.gain) {
                    best = Some(Lens {
                        outline: outline.clone(),
                        sigma: sg * u,
                        opacity: a,
                        gain,
                    });
                }
            }
        }
        if let Some(b) = best
            && opt.keeps(&b.mark(b.gain + credit(comp, &b)))
        {
            cands.push(b);
        }
    }
    cands.sort_by(|a, b| b.gain.total_cmp(&a.gain));
    let mut out = Vec::new();
    for mut l in cands {
        crate::progress::poll().await;
        let (win, g) = lens_alpha(&l.outline, l.sigma, clip, w, h);
        let (a, gain) = white_fit(img, comp, win, &g, opt.tol());
        let gain = alpha_gain(win, &g, a).unwrap_or(gain);
        if a <= 0.0 || !opt.keeps(&l.mark(gain + credit(comp, &l))) {
            continue;
        }
        for (i, &gi) in g.iter().enumerate() {
            if gi > 0.01 {
                let p = win.at(i, w);
                comp[p] += a * gi * (DVec3::ONE - comp[p]);
            }
        }
        l.opacity = a;
        l.gain = gain;
        out.push(l);
    }
    out
}
