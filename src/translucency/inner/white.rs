//! The inner layers above the layer: the white by opening, the tints, the full fit.

use crate::HashMap;

use super::{
    Class, DVec3, InnerFit, InnerInputs, MATCH_TOL, NO_FIT, NOISE, Open, PixelFit, Tint, VEIL_MAX,
    arange, assemble, cheb, classify_colors, open_at, with_interior,
};

/// The white above the layer read against a smooth beneath: per pixel its opacity and fit.
pub struct OpenedWhite {
    /// per pixel of the region's inside its fit (the scored pixels)
    pub fits: Vec<PixelFit>,
    /// per pixel of the region, rims included, the white's opacity (0 where none)
    pub alpha: Vec<f64>,
}

/// The opening's width: wider than the white highlights' cores, narrower than the
/// beneath's flat tones (5 px at the unit's reference width, 7 at twice it).
pub fn opening_width(width: usize) -> usize {
    2 * crate::translucency::raster::unit_of(width).ceil() as usize + 3
}

/// A white of varying opacity above the layer, against the beneath a grey opening of the
/// peel estimates (the white only brightens; outside the region held white). Per pixel t
/// solves `o = (1 - t) d + t`, kept in (0.02, 0.9] within twice the tolerance, with interior.
pub fn white_by_opening(inp: &InnerInputs, op: &Open, k: f64, m: DVec3) -> OpenedWhite {
    let (w, h) = (inp.width, inp.height);
    let size = opening_width(w);
    let r = size / 2;
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for (i, v) in inp.region.iter().enumerate() {
        if *v {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    let mut fits = vec![NO_FIT; w * h];
    if x1 < x0 {
        return OpenedWhite {
            fits,
            alpha: vec![0.0; w * h],
        };
    }
    let (x0, y0) = (x0.saturating_sub(r), y0.saturating_sub(r));
    let (x1, y1) = ((x1 + r).min(w - 1), (y1 + r).min(h - 1));
    let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
    let mut peel = vec![DVec3::ONE; bw * bh];
    for y in 0..bh {
        for x in 0..bw {
            let i = (y + y0) * w + x + x0;
            if inp.region[i] {
                peel[y * bw + x] = (inp.px[i] - m) / k;
            }
        }
    }
    // min then max over the square window, separably, per channel
    let filt = |src: &[DVec3], take_min: bool| -> Vec<DVec3> {
        let pick = |a: DVec3, b: DVec3| if take_min { a.min(b) } else { a.max(b) };
        let mut tmp = src.to_vec();
        for y in 0..bh {
            for x in 0..bw {
                let mut v = src[y * bw + x];
                for xx in x.saturating_sub(r)..=(x + r).min(bw - 1) {
                    v = pick(v, src[y * bw + xx]);
                }
                tmp[y * bw + x] = v;
            }
        }
        let mut out = tmp.clone();
        for y in 0..bh {
            for x in 0..bw {
                let mut v = tmp[y * bw + x];
                for yy in y.saturating_sub(r)..=(y + r).min(bh - 1) {
                    v = pick(v, tmp[yy * bw + x]);
                }
                out[y * bw + x] = v;
            }
        }
        out
    };
    let opened = filt(&filt(&peel, true), false);
    let tol = MATCH_TOL;
    let mut mask = vec![false; w * h];
    let mut alpha = vec![0.0f64; w * h];
    for y in 0..bh {
        for x in 0..bw {
            let i = (y + y0) * w + x + x0;
            if !inp.region[i] {
                continue;
            }
            let b = opened[y * bw + x].clamp(DVec3::ZERO, DVec3::ONE);
            let d = k * b + m;
            let v = DVec3::ONE - d;
            let vv = v.dot(v);
            if vv < 1e-12 {
                continue;
            }
            let o = inp.px[i];
            let t = (o - d).dot(v) / vv;
            let res = (o - d - t * v).abs().max_element();
            if t > 0.02 && t <= VEIL_MAX && res <= 2.0 * tol {
                let mat = op
                    .tree
                    .nearest(b, 1.0)
                    .map_or(usize::MAX, |(_, j)| op.bmat[j]);
                if inp.inside[i] {
                    fits[i] = PixelFit {
                        class: Class::White,
                        mat,
                        t,
                        d: res / 2.0,
                    };
                }
                alpha[i] = t;
                mask[i] = true;
            }
        }
    }
    let kept = with_interior(&mask, w, h);
    for ((f, a), k) in fits.iter_mut().zip(alpha.iter_mut()).zip(&kept) {
        if !*k {
            *f = NO_FIT;
            *a = 0.0;
        }
    }
    OpenedWhite {
        fits,
        alpha,
    }
}

/// A tint above the layer, and per distinct color its fit: the homothety from the layer's
/// images to the colors nothing else explains, by votes `mL = o - kL d` per kL in 0.5 to
/// 0.96 (opacity up to a half), the best cells refitted and scored by pixels explained.
pub fn fit_tint(inp: &InnerInputs, op: &Open, taken: &[PixelFit]) -> Option<(Tint, Vec<PixelFit>)> {
    let tol = MATCH_TOL;
    let cell = 2.0 * tol;
    let images = &op.images;
    let free: Vec<usize> = (0..inp.colors.len())
        .filter(|&e| op.scored[e] && taken[e].class == Class::None)
        .collect();
    if free.is_empty() || images.is_empty() {
        return None;
    }
    // per free color its best image under (kL, mL) within lim
    let matches = |kl: f64, ml: DVec3, lim: f64| -> Vec<(PixelFit, DVec3)> {
        let mut out = vec![(NO_FIT, DVec3::ZERO); inp.colors.len()];
        for &e in &free {
            let o = inp.colors[e];
            for &(q, r) in images {
                let pred = kl * q + ml;
                // the tint moves the image by more than the tolerance
                if cheb(pred, q) <= tol {
                    continue;
                }
                let d = (o - pred).abs().max_element();
                if d <= lim && d < out[e].0.d {
                    out[e] = (
                        PixelFit {
                            class: Class::Tint(0),
                            mat: r,
                            t: 1.0 - kl,
                            d,
                        },
                        q,
                    );
                }
            }
        }
        out
    };
    let mut best: Option<(f64, Tint, Vec<PixelFit>)> = None;
    for kl in arange(0.5, 0.961, 0.02) {
        let margin = NOISE / (1.0 - kl);
        let mut votes: HashMap<[i64; 3], (f64, usize)> = HashMap::default();
        for (fi, &e) in free.iter().enumerate() {
            let o = inp.colors[e];
            for &(q, _) in images {
                let ml = o - kl * q;
                let c = ml / (1.0 - kl);
                if c.min_element() < -margin || c.max_element() > 1.0 + margin {
                    continue;
                }
                let g = (ml / cell).floor();
                let v = votes
                    .entry([g.x as i64, g.y as i64, g.z as i64])
                    .or_insert((0.0, usize::MAX));
                if v.1 != fi {
                    v.0 += inp.weights[e];
                    v.1 = fi;
                }
            }
        }
        let mut cells: Vec<([i64; 3], f64)> = votes.into_iter().map(|(k, v)| (k, v.0)).collect();
        cells.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for (key, _) in cells.into_iter().take(3) {
            let mut ml = (DVec3::new(key[0] as f64, key[1] as f64, key[2] as f64) + 0.5) * cell;
            let mut fit = matches(kl, ml, cell);
            for _ in 0..2 {
                let (mut s, mut n) = (DVec3::ZERO, 0.0);
                for &e in &free {
                    if fit[e].0.class != Class::None {
                        s += inp.weights[e] * (inp.colors[e] - kl * fit[e].1);
                        n += inp.weights[e];
                    }
                }
                if n == 0.0 {
                    break;
                }
                ml = s / n;
                fit = matches(kl, ml, tol);
            }
            let c = ml / (1.0 - kl);
            if c.min_element() < -margin || c.max_element() > 1.0 + margin {
                continue;
            }
            let wsum: f64 = free
                .iter()
                .filter(|&&e| fit[e].0.class != Class::None)
                .map(|&e| inp.weights[e] * (1.0 - fit[e].0.d / tol))
                .sum();
            if best.as_ref().is_none_or(|b| wsum > b.0) {
                best = Some((
                    wsum,
                    Tint { k: kl, m: ml },
                    fit.into_iter().map(|f| f.0).collect(),
                ));
            }
        }
    }
    best.map(|(_, t, f)| (t, f))
}

/// The inner layers at a pinned relation: the layer alone, the white ([`white_by_opening`])
/// and up to `tints` tints, each fitted on the colors left (a color is the white's when most
/// of its pixels are).
pub fn fit_full(inp: &InnerInputs, k: f64, m: DVec3, tints: usize) -> InnerFit {
    let op = open_at(inp, k, m);
    let mut pc = classify_colors(inp, &op, k, m, false);
    let wb = white_by_opening(inp, &op, k, m);
    let mut white_px = vec![0.0f64; inp.colors.len()];
    for (f, &e) in wb.fits.iter().zip(&inp.index) {
        if f.class == Class::White && e != u32::MAX {
            white_px[e as usize] += 1.0;
        }
    }
    let mut taken = pc.clone();
    for (e, t) in taken.iter_mut().enumerate() {
        if t.class != Class::White && white_px[e] > 0.5 * inp.weights[e] {
            t.class = Class::White;
        }
    }
    let mut found = Vec::new();
    for ti in 0..tints {
        let Some((t, f)) = fit_tint(inp, &op, &taken) else {
            break;
        };
        let mut any = false;
        for ((p, q), g) in pc.iter_mut().zip(taken.iter_mut()).zip(&f) {
            if q.class == Class::None && g.class != Class::None {
                *p = PixelFit {
                    class: Class::Tint(ti),
                    ..*g
                };
                *q = *p;
                any = true;
            }
        }
        if !any {
            break;
        }
        found.push(t);
    }
    let mut f = assemble(inp, &op, k, m, &pc, found, Some((&wb.fits, true)));
    f.white = wb.alpha;
    f
}
