//! A line's paint: its coverage, colour and width fitted against the target's ground.

use super::*;

/// A window of the image: columns x0..x1, rows y0..y1.
#[derive(Clone, Copy, Debug)]
pub(super) struct Win {
    pub(super) x0: usize,
    pub(super) y0: usize,
    pub(super) x1: usize,
    pub(super) y1: usize,
}

impl Win {
    pub(super) fn around(paths: &[Vec<DVec2>], pad: f64, width: usize, height: usize) -> Win {
        let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
        for q in paths.iter().flatten() {
            lo = lo.min(*q);
            hi = hi.max(*q);
        }
        let x0 = (lo.x - pad).floor().max(0.0) as usize;
        let y0 = (lo.y - pad).floor().max(0.0) as usize;
        let x1 = ((hi.x + pad).ceil().max(0.0) as usize).min(width);
        let y1 = ((hi.y + pad).ceil().max(0.0) as usize).min(height);
        Win {
            x0,
            y0,
            x1: x1.max(x0),
            y1: y1.max(y0),
        }
    }
    pub(super) fn w(&self) -> usize {
        self.x1 - self.x0
    }
    pub(super) fn len(&self) -> usize {
        self.w() * (self.y1 - self.y0)
    }
    pub(super) fn at(&self, i: usize, width: usize) -> usize {
        (self.y0 + i / self.w()) * width + self.x0 + i % self.w()
    }
}

/// The anti-aliased coverage of a stroke of `width` along polylines over a window, the box
/// filter's approximation the vocabulary's strokes use (half the width less the pixel
/// centre's distance, plus a half, clipped), each segment visiting only its own box.
pub(super) fn coverage(paths: &[Vec<DVec2>], width: f64, win: Win) -> Vec<f64> {
    let mut d = vec![f64::INFINITY; win.len()];
    let r = width / 2.0 + 1.0;
    let ww = win.w();
    for p in paths {
        let segs: Vec<(DVec2, DVec2)> = if p.len() == 1 {
            vec![(p[0], p[0])]
        } else {
            p.windows(2).map(|s| (s[0], s[1])).collect()
        };
        for (a, b) in segs {
            let lo = a.min(b) - DVec2::splat(r);
            let hi = a.max(b) + DVec2::splat(r);
            let xa = (lo.x.floor().max(win.x0 as f64) as usize).min(win.x1);
            let xb = (hi.x.ceil().max(0.0) as usize).clamp(win.x0, win.x1);
            let ya = (lo.y.floor().max(win.y0 as f64) as usize).min(win.y1);
            let yb = (hi.y.ceil().max(0.0) as usize).clamp(win.y0, win.y1);
            let ab = b - a;
            for y in ya..yb {
                for x in xa..xb {
                    let q = DVec2::new(x as f64 + 0.5, y as f64 + 0.5);
                    let t = ((q - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
                    let k = (y - win.y0) * ww + x - win.x0;
                    d[k] = d[k].min((q - (a + t * ab)).length());
                }
            }
        }
    }
    d.into_iter()
        .map(|d| (width / 2.0 - d + 0.5).clamp(0.0, 1.0))
        .collect()
}

/// The colour and opacity of a stroke over the composite from its coverage, as the
/// vocabulary fits its strokes (o = d + a cov (C - d) by least squares); refused where the
/// paint lies outside the gamut by more than the noise.
pub(super) fn fit_paint(
    img: &Image,
    comp: &[DVec3],
    win: Win,
    cov: &[f64],
) -> Option<(DVec3, f64)> {
    fit_paint_by(img, win, cov, |i| comp[win.at(i, img.width)])
}

/// [`fit_paint`] over the ground `under` per window pixel.
pub(super) fn fit_paint_by(
    img: &Image,
    win: Win,
    cov: &[f64],
    under: impl Fn(usize) -> DVec3,
) -> Option<(DVec3, f64)> {
    let mut ata = DMat4::ZERO;
    let mut atb = DVec4::ZERO;
    let mut n = 0;
    for (i, &w) in cov.iter().enumerate() {
        if w <= 0.05 {
            continue;
        }
        n += 1;
        let p = win.at(i, img.width);
        let (o, d) = (img.px[p], under(i));
        for c in 0..3 {
            let mut row = DVec4::ZERO;
            row[c] = w;
            row[3] = -w * d[c];
            let y = o[c] - d[c];
            atb += row * y;
            ata += DMat4::from_cols(row * row[0], row * row[1], row * row[2], row * row[3]);
        }
    }
    if n < 3 {
        return None;
    }
    // over a ground of one colour only the paint a C is told, not a: a faint pull toward an
    // opaque line decides it there (an artist's line is opaque), lost under a varying ground
    let pull = 1e-4 * n as f64;
    ata.w_axis.w += pull;
    atb.w += pull;
    if ata.determinant().abs() < 1e-12 {
        return None;
    }
    let sol = ata.inverse() * atb;
    let a = sol[3].clamp(0.0, 1.0);
    if !a.is_finite() || a < 1e-3 {
        return None;
    }
    let (mut num, mut den) = (DVec3::ZERO, 0.0);
    for (i, &w) in cov.iter().enumerate() {
        if w <= 0.05 {
            continue;
        }
        let p = win.at(i, img.width);
        let aw = a * w;
        num += (img.px[p] - (1.0 - aw) * under(i)) * aw;
        den += aw * aw;
    }
    let raw = num / den.max(1e-12);
    let c = raw.clamp(DVec3::ZERO, DVec3::ONE);
    if a * (raw - c).abs().max_element() > NOISE {
        return None;
    }
    Some((c, a))
}

/// A dash row: the line through its gaps and each dash's span along it (arc lengths).
pub(super) type DashRow = (Vec<DVec2>, Vec<(f64, f64)>);

/// A stroke candidate's geometry and paint, fitted.
#[derive(Clone, Debug)]
pub(super) struct Fit {
    pub(super) paths: Vec<Vec<Bezier>>,
    pub(super) flat: Vec<Vec<DVec2>>,
    pub(super) width: f64,
    pub(super) colour: DVec3,
    pub(super) opacity: f64,
    pub(super) plain: f64,
    pub(super) edge: f64,
    pub(super) smooth: f64,
    pub(super) price: f64,
    /// for a dash row, the line through its gaps and each dash's span along it (arc lengths)
    pub(super) row: Option<DashRow>,
    /// what the stroke saves drawn over the target's own ground beside it (the straight mix
    /// of its sides), edge-weighted: what the line is worth where nothing stands in for it
    pub(super) own: f64,
}

/// What a candidate removes from the composite under the objective: the squared error, and
/// the edge term's extra weight at the target's edges on that saving, discounted by `smooth`.
pub(super) fn saving(
    img: &Image,
    comp: &[DVec3],
    field: &EdgeField,
    opt: &Options,
    paint: impl Iterator<Item = (usize, DVec3)>,
    smooth: f64,
) -> (f64, f64) {
    let (mut plain, mut edge) = (0.0, 0.0);
    for (p, n) in paint {
        let d = comp[p];
        let saved = field.residual(p, d - img.px[p]) - field.residual(p, n - img.px[p]);
        plain += saved;
        edge += (field.weight(p, opt.edge) as f64 - 1.0) * saved;
    }
    (plain, smooth * edge)
}

/// The paint of a fixed stroke geometry over the composite and what it saves; the pixels it
/// covers with their alpha.
pub(super) fn paint(
    img: &Image,
    comp: &[DVec3],
    field: &EdgeField,
    opt: &Options,
    f: &mut Fit,
) -> Option<Vec<(usize, f64)>> {
    let win = Win::around(&f.flat, f.width / 2.0 + 2.0, img.width, img.height);
    if win.len() == 0 {
        return None;
    }
    let cov = coverage(&f.flat, f.width, win);
    let (c, a) = fit_paint(img, comp, win, &cov)?;
    let alpha: Vec<(usize, f64)> = cov
        .iter()
        .enumerate()
        .filter(|(_, w)| **w > 0.0)
        .map(|(i, w)| (win.at(i, img.width), a * w))
        .collect();
    let (plain, edge) = saving(
        img,
        comp,
        field,
        opt,
        alpha.iter().map(|&(p, a)| (p, comp[p] + a * (c - comp[p]))),
        f.smooth,
    );
    f.colour = c;
    f.opacity = a;
    f.plain = plain;
    f.edge = edge;
    Some(alpha)
}

/// Unit normals of a polyline (left of the direction in image coordinates).
pub(super) fn normals(p: &[DVec2]) -> Vec<DVec2> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let d = p[(i + 1).min(n - 1)] - p[i.saturating_sub(1)];
            let d = d / d.length().max(1e-9);
            DVec2::new(-d.y, d.x)
        })
        .collect()
}

/// The widest line the detector finds: twice its far radius ([`colour_lines_at`]) less the
/// anti-aliasing reach, at least a pixel. A wider stroke is a tone correction, not a line.
pub fn widest_line(u: f64) -> f64 {
    (2.0 * (2.0 * u - REACH as f64)).max(1.0)
}

/// The width of the target's line along `paths`, read in colour (so a pale hem counts): per
/// point, the profile's distance from the mix of its two ends, its central peak's width at
/// half height; the median over points peaking above the noise, None where none does.
pub fn line_width(img: &Image, paths: &[Vec<DVec2>], u: f64) -> Option<f64> {
    let reach = 2.0 * u + REACH as f64;
    let steps = (2.0 * reach / 0.5).round() as usize;
    let mut found = Vec::new();
    for p in paths {
        let nrm = normals(p);
        for (q, n) in p.iter().zip(&nrm).step_by(2) {
            let prof: Vec<DVec3> = (0..=steps)
                .map(|k| {
                    let r = *q + (-reach + 0.5 * k as f64) * *n - DVec2::splat(0.5);
                    img.sample(r.x, r.y)
                })
                .collect();
            let (a, b) = (prof[0], prof[steps]);
            let e: Vec<f64> = prof
                .iter()
                .enumerate()
                .map(|(k, c)| {
                    let t = k as f64 / steps as f64;
                    (*c - (a + t * (b - a))).abs().max_element()
                })
                .collect();
            // the peak nearest the centre (the centreline is read to the pixel), and the run
            // about it at half its height: a broad bump of shading is wide there however its
            // flanks curve, a line is its own width
            let centre = steps / 2;
            let near = (2.0 * (1.0 + REACH as f64)) as usize;
            let (lo, hi) = (centre.saturating_sub(near), (centre + near).min(steps));
            let k = (lo..=hi)
                .max_by(|&a, &b| e[a].total_cmp(&e[b]))
                .unwrap_or(centre);
            let peak = e[k];
            if peak > NOISE {
                let (mut a, mut b) = (k, k);
                while a > 0 && e[a - 1] >= peak / 2.0 {
                    a -= 1;
                }
                while b < steps && e[b + 1] >= peak / 2.0 {
                    b += 1;
                }
                found.push(0.5 * (b - a + 1) as f64);
            }
        }
    }
    if found.is_empty() {
        return None;
    }
    found.sort_by(f64::total_cmp);
    Some(found[found.len() / 2])
}

/// The widths a line is fitted at: by half pixels from one pixel (a thinner line is a fainter
/// one-pixel line) to its [`line_width`] plus half, at most [`widest_line`]. None for a wider
/// ridge: it is a tone (a shading ridge, a highlight), which fills draw.
pub(super) fn widths(img: &Image, paths: &[Vec<DVec2>], u: f64) -> Vec<f64> {
    let top = widest_line(u);
    let Some(m) = line_width(img, paths, u).filter(|&m| m <= top) else {
        return Vec::new();
    };
    // every width from the anti-aliasing scale to the reading and half a pixel beyond, by
    // half pixels: the reading is blurred by the raster's pixels, so the line is no wider
    let n = ((m.min(top) + 0.5 - 1.0) / 0.5).floor().max(0.0) as usize;
    (0..=n).map(|k| (1.0 + 0.5 * k as f64).min(top)).collect()
}

/// The width out of `tried` at which a stroke along `flat` best draws the target over the
/// target's own ground (the mix of its two sides), and that gain: fitted against the
/// composite, least squares would widen a stroke to correct the composite's tone.
pub(super) fn target_width(
    img: &Image,
    comp: &[DVec3],
    field: &EdgeField,
    lambda: f64,
    flat: &[Vec<DVec2>],
    tried: &[f64],
) -> Option<(f64, f64)> {
    let top = tried.iter().copied().fold(1.0, f64::max);
    let win = Win::around(flat, top / 2.0 + 2.0, img.width, img.height);
    if win.len() == 0 {
        return None;
    }
    let r = top / 2.0 + REACH as f64 + 0.5;
    // per window pixel within the widest stroke's reach, the nearest point of the centreline
    // and its segment's direction, each segment visiting only its own box (as `coverage`)
    let reach = top / 2.0 + 1.0;
    let ww = win.w();
    let mut near = vec![(f64::INFINITY, DVec2::ZERO, DVec2::ZERO); win.len()];
    for p in flat {
        for s in p.windows(2) {
            let (a, b) = (s[0], s[1]);
            let lo = a.min(b) - DVec2::splat(reach);
            let hi = a.max(b) + DVec2::splat(reach);
            let xa = (lo.x.floor().max(win.x0 as f64) as usize).min(win.x1);
            let xb = (hi.x.ceil().max(0.0) as usize).clamp(win.x0, win.x1);
            let ya = (lo.y.floor().max(win.y0 as f64) as usize).min(win.y1);
            let yb = (hi.y.ceil().max(0.0) as usize).clamp(win.y0, win.y1);
            let ab = b - a;
            for y in ya..yb {
                for x in xa..xb {
                    let c = DVec2::new(x as f64 + 0.5, y as f64 + 0.5);
                    let t = ((c - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
                    let q = a + t * ab;
                    let k = (y - win.y0) * ww + x - win.x0;
                    let d = c.distance(q);
                    if d < near[k].0 {
                        near[k] = (d, q, ab);
                    }
                }
            }
        }
    }
    let ground: Vec<DVec3> = (0..win.len())
        .map(|i| {
            let (d, q, ab) = near[i];
            if !d.is_finite() {
                return comp[win.at(i, img.width)];
            }
            let c = DVec2::new(
                (win.x0 + i % ww) as f64 + 0.5,
                (win.y0 + i / ww) as f64 + 0.5,
            );
            let n = DVec2::new(-ab.y, ab.x).normalize_or_zero();
            let s = (c - q).dot(n);
            let left = q + r * n - DVec2::splat(0.5);
            let right = q - r * n - DVec2::splat(0.5);
            let (l, rr) = (img.sample(left.x, left.y), img.sample(right.x, right.y));
            let t = ((s + r) / (2.0 * r)).clamp(0.0, 1.0);
            rr + t * (l - rr)
        })
        .collect();
    // (gain, width, pixels drawn)
    let mut fits: Vec<(f64, f64, usize)> = Vec::new();
    // a width's coverage is that of the nearest centreline point, which `near` holds within
    // the widest stroke's reach (beyond it no width covers); a lone point has no segment there
    let lone = flat.iter().any(|p| p.len() == 1);
    for &width in tried {
        let cov: Vec<f64> = if lone {
            coverage(flat, width, win)
        } else {
            near.iter()
                .map(|n| (width / 2.0 - n.0 + 0.5).clamp(0.0, 1.0))
                .collect()
        };
        let Some((col, a)) = fit_paint_by(img, win, &cov, |i| ground[i]) else {
            continue;
        };
        let (mut gain, mut n) = (0.0, 0);
        for (i, &w) in cov.iter().enumerate() {
            if w <= 0.0 {
                continue;
            }
            n += 1;
            let p = win.at(i, img.width);
            let g = ground[i];
            let o = g + a * w * (col - g);
            gain += field.weight(p, lambda) as f64
                * (field.residual(p, g - img.px[p]) - field.residual(p, o - img.px[p]));
        }
        fits.push((gain, width, n));
    }
    let best = fits.iter().copied().max_by(|a, b| a.0.total_cmp(&b.0))?;
    // an artist's line is thin and opaque, and below the anti-aliasing scale a thinner, more
    // opaque line lays down the same paint: of the widths within the noise of the best, the
    // thinnest (only the thin one matches at the truth's 4x)
    let tolerance = best.2 as f64 * 3.0 * NOISE * NOISE;
    fits.iter()
        .filter(|f| f.0 >= best.0 - tolerance)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|f| (f.1, f.0))
}

/// The best stroke along `subpaths` (one path) over shifts across of up to half a pixel and
/// the [`widths`], its centreline fitted to within a pixel, by its net gain at the price of
/// one path, its cubics and one segment for width and paint.
pub(super) fn fit(
    img: &Image,
    comp: &[DVec3],
    field: &EdgeField,
    opt: &Options,
    subpaths: &[Vec<DVec2>],
    smooth: f64,
) -> Option<Fit> {
    let u = unit_of(img.width);
    let mut best: Option<Fit> = None;
    let tried = widths(img, subpaths, u);
    for sh in [-0.5, 0.0, 0.5] {
        let paths: Vec<Vec<Bezier>> = subpaths
            .iter()
            .filter(|p| p.len() >= 2)
            .map(|p| {
                let nrm = normals(p);
                let moved: Vec<DVec2> = p.iter().zip(&nrm).map(|(q, n)| *q + sh * *n).collect();
                fit_cubics(&moved, FIT_TOL)
            })
            .collect();
        if paths.is_empty() {
            return None;
        }
        let flat: Vec<Vec<DVec2>> = paths.iter().map(|c| flatten(c)).collect();
        let cubics: usize = paths.iter().map(Vec::len).sum();
        let price = opt.path_cost + (cubics as f64 + 1.0) * opt.segment_cost;
        let Some((width, own)) = target_width(img, comp, field, opt.edge, &flat, &tried) else {
            continue;
        };
        {
            let mut f = Fit {
                paths: paths.clone(),
                flat: flat.clone(),
                width,
                colour: DVec3::ZERO,
                opacity: 0.0,
                plain: 0.0,
                edge: 0.0,
                smooth,
                price,
                row: None,
                own,
            };
            if paint(img, comp, field, opt, &mut f).is_none() {
                continue;
            }
            let net = |f: &Fit| f.plain + f.edge - f.price;
            if best.as_ref().is_none_or(|b| net(&f) > net(b)) {
                best = Some(f);
            }
        }
    }
    best
}

/// The stretches of a line where the target differs from the composite by over 3 NOISE,
/// gaps of up to the reach bridged: a dash row's dashes (one chain can hold a row of
/// stitches, the gradient running on across their gaps).
pub(super) fn fragments(img: &Image, comp: &[DVec3], p: &[DVec2]) -> Vec<Vec<DVec2>> {
    let (w, h) = (img.width, img.height);
    let differs: Vec<bool> = p
        .iter()
        .map(|q| {
            let (x, y) = (q.x.floor(), q.y.floor());
            if x < 0.0 || y < 0.0 || x as usize >= w || y as usize >= h {
                return false;
            }
            let i = y as usize * w + x as usize;
            (img.px[i] - comp[i]).abs().max_element() > 3.0 * NOISE
        })
        .collect();
    let mut out: Vec<Vec<DVec2>> = Vec::new();
    let (mut start, mut last) = (None::<usize>, 0usize);
    for (i, &d) in differs.iter().enumerate() {
        if !d {
            continue;
        }
        match start {
            Some(s) if i - last - 1 > REACH => {
                out.push(p[s..=last].to_vec());
                start = Some(i);
            }
            None => start = Some(i),
            _ => {}
        }
        last = i;
    }
    if let Some(s) = start {
        out.push(p[s..=last].to_vec());
    }
    out.retain(|f| f.len() >= 2);
    out
}

/// The stroke candidates of one linked line: through its gaps, as a dash row of where it
/// differs from the composite, as one path of its members, and each member alone (a junction
/// may have joined lines of different colours).
pub(super) fn stroke_candidates(
    img: &Image,
    comp: &[DVec3],
    field: &EdgeField,
    weak: &[bool],
    opt: &Options,
    chains: &[Chain],
    line: &Linked,
) -> Vec<Fit> {
    let (parts, through) = polylines(chains, line);
    // the line's simplicity is judged on the whole linked line, gaps and all
    let smooth = discount(&through, field, weak);
    let mut out = Vec::new();
    out.extend(fit(
        img,
        comp,
        field,
        opt,
        std::slice::from_ref(&through),
        smooth,
    ));
    let dashes = fragments(img, comp, &through);
    if dashes.len() > 1 {
        // each dash's span along the line through the gaps, for drawing the row as one dashed
        // stroke (its pieces are copies of the line's points)
        let mut arc = vec![0.0];
        for q in through.windows(2) {
            arc.push(arc[arc.len() - 1] + q[0].distance(q[1]));
        }
        let find = |q: DVec2| through.iter().position(|t| *t == q).map(|i| arc[i]);
        let spans: Option<Vec<(f64, f64)>> = dashes
            .iter()
            .map(|d| Some((find(d[0])?, find(d[d.len() - 1])?)))
            .collect();
        out.extend(fit(img, comp, field, opt, &dashes, smooth).map(|mut f| {
            f.row = spans.map(|s| (through.clone(), s));
            f
        }));
    }
    if parts.len() > 1 {
        // the members linked across their gaps are a dash row too where they are collinear
        // pieces (the shoe's stitches, linked by the continuation test)
        let mut spans = Vec::new();
        let mut s0 = 0.0;
        for (k, p) in parts.iter().enumerate() {
            if k > 0 {
                s0 += parts[k - 1][parts[k - 1].len() - 1].distance(p[0]);
            }
            let l = arc_length(p);
            spans.push((s0, s0 + l));
            s0 += l;
        }
        out.extend(fit(img, comp, field, opt, &parts, smooth).map(|mut f| {
            f.row = Some((through.clone(), spans));
            f
        }));
        for p in &parts {
            let s = discount(p, field, weak);
            out.extend(fit(img, comp, field, opt, std::slice::from_ref(p), s));
        }
    }
    out
}
