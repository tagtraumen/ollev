//! Cuts: the flat areas either side of a step and their fills.

use super::*;

/// The flat areas a cut may take: the components, 4-connected, of the pixels off every edge
/// of the target and of the render, each with the edge pixels 4-adjacent to it (the boundary
/// a shape's fill reaches to).
pub(super) struct Areas {
    pub(super) label: Vec<u32>,
    pub(super) pixels: Vec<Vec<usize>>,
}

pub(super) fn areas(barrier: &[bool], width: usize, height: usize) -> Areas {
    let open: Vec<bool> = barrier.iter().map(|b| !*b).collect();
    let (lab, n) = label(&open, width, height, Conn::Four);
    let mut pixels = vec![Vec::new(); n + 1];
    for (i, &l) in lab.iter().enumerate() {
        if l > 0 {
            pixels[l as usize].push(i);
        } else {
            let (x, y) = (i % width, i / width);
            let mut near: Vec<u32> = Vec::new();
            if x > 0 {
                near.push(lab[i - 1]);
            }
            if x + 1 < width {
                near.push(lab[i + 1]);
            }
            if y > 0 {
                near.push(lab[i - width]);
            }
            if y + 1 < height {
                near.push(lab[i + width]);
            }
            near.retain(|&l| l > 0);
            near.sort_unstable();
            near.dedup();
            for l in near {
                pixels[l as usize].push(i);
            }
        }
    }
    Areas { label: lab, pixels }
}

/// The areas either side of a linked line: per side the area most of its samples land in, 2
/// px off the line (past its anti-aliased rim), or none where the two sides are one area.
pub(super) fn sides(
    areas: &Areas,
    through: &[DVec2],
    width: usize,
    height: usize,
) -> Option<[u32; 2]> {
    let nrm = normals(through);
    let mut votes: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    for (p, n) in through.iter().zip(&nrm).step_by(2) {
        for (s, sign) in [(0, 1.0), (1, -1.0)] {
            let q = *p + sign * 2.0 * *n;
            if q.x < 0.0 || q.y < 0.0 || q.x as usize >= width || q.y as usize >= height {
                continue;
            }
            let l = areas.label[q.y as usize * width + q.x as usize];
            if l > 0 {
                votes[s].push(l);
            }
        }
    }
    let mode = |v: &mut Vec<u32>| {
        v.sort_unstable();
        let mut best = (0, 0);
        let mut i = 0;
        while i < v.len() {
            let j = v[i..].iter().take_while(|&&l| l == v[i]).count();
            if j > best.1 {
                best = (v[i], j);
            }
            i += j;
        }
        best.0
    };
    let (a, b) = (mode(&mut votes[0]), mode(&mut votes[1]));
    (a > 0 && b > 0 && a != b).then_some([a, b])
}

/// A cut candidate: its fill and outline, what it saves and costs.
#[derive(Clone, Debug)]
pub(super) struct CutFit {
    pub(super) outline: Vec<Vec<Bezier>>,
    pub(super) fill: CutFill,
    pub(super) plain: f64,
    pub(super) edge: f64,
    pub(super) smooth: f64,
    pub(super) price: f64,
}

/// The fills of an area by least squares on the target: its mean, and the linear gradient
/// along the direction the colour changes most (the channels' gradients, summed, as one
/// direction: an SVG linear gradient has one), between the projections' extremes.
pub(super) fn area_fills(img: &Image, px: &[usize]) -> (CutFill, Option<CutFill>) {
    let w = img.width;
    let at = |i: usize| DVec2::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
    let n = px.len() as f64;
    let mean = px.iter().map(|&i| img.px[i]).sum::<DVec3>() / n;
    let flat = CutFill::Flat(mean);
    if px.len() < 6 {
        return (flat, None);
    }
    let centre = px.iter().map(|&i| at(i)).sum::<DVec2>() / n;
    // c = a + B (p - centre): per channel the 2 x 2 normal equations of the slopes
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    let (mut bx, mut by) = (DVec3::ZERO, DVec3::ZERO);
    for &i in px {
        let d = at(i) - centre;
        let e = img.px[i] - mean;
        sxx += d.x * d.x;
        sxy += d.x * d.y;
        syy += d.y * d.y;
        bx += d.x * e;
        by += d.y * e;
    }
    let m = DMat3::from_cols(
        DVec3::new(sxx, sxy, 0.0),
        DVec3::new(sxy, syy, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    );
    if m.determinant().abs() < 1e-9 {
        return (flat, None);
    }
    let inv = m.inverse();
    let slope = |c: usize| {
        let s = inv * DVec3::new(bx[c], by[c], 0.0);
        DVec2::new(s.x, s.y)
    };
    let g = [slope(0), slope(1), slope(2)];
    // one direction: the channels' slopes summed with their signs aligned to the largest
    let big = *g
        .iter()
        .max_by(|a, b| a.length().total_cmp(&b.length()))
        .unwrap();
    let dir: DVec2 = g
        .iter()
        .map(|v| if v.dot(big) < 0.0 { -*v } else { *v })
        .sum();
    if dir.length() < 1e-9 {
        return (flat, None);
    }
    let dir = dir.normalize();
    let ts: Vec<f64> = px.iter().map(|&i| (at(i) - centre).dot(dir)).collect();
    let tt: f64 = ts.iter().map(|t| t * t).sum();
    if tt < 1e-9 {
        return (flat, None);
    }
    let b = px
        .iter()
        .zip(&ts)
        .map(|(&i, &t)| t * (img.px[i] - mean))
        .sum::<DVec3>()
        / tt;
    let (lo, hi) = ts
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &t| {
            (l.min(t), h.max(t))
        });
    let linear = CutFill::Linear {
        from: centre + lo * dir,
        to: centre + hi * dir,
        c0: (mean + lo * b).clamp(DVec3::ZERO, DVec3::ONE),
        c1: (mean + hi * b).clamp(DVec3::ZERO, DVec3::ONE),
    };
    (flat, Some(linear))
}

/// A cut over one area: its outline, the better of its fills by net gain, the area's pixels
/// painted, at the price of one path, its outline's cubics and one segment more for a
/// gradient's direction.
pub(super) fn fit_cut(
    img: &Image,
    comp: &[DVec3],
    field: &EdgeField,
    opt: &Options,
    areas: &Areas,
    area: u32,
    smooth: f64,
) -> Option<CutFit> {
    let (w, h) = (img.width, img.height);
    let px = &areas.pixels[area as usize];
    if px.is_empty() {
        return None;
    }
    let mut mask = vec![false; w * h];
    for &i in px {
        mask[i] = true;
    }
    let outline = outline(&mask, w, h);
    let cubics: usize = outline.iter().map(Vec::len).sum();
    if cubics == 0 {
        return None;
    }
    let at = |i: usize| DVec2::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
    let (flat, linear) = area_fills(img, px);
    let mut best: Option<CutFit> = None;
    for (fill, extra) in [(Some(flat), 0.0), (linear, 1.0)] {
        let Some(fill) = fill else {
            continue;
        };
        let (plain, edge) = saving(
            img,
            comp,
            field,
            opt,
            px.iter().map(|&i| (i, fill.at(at(i)))),
            smooth,
        );
        let price = opt.path_cost + (cubics as f64 + extra) * opt.segment_cost;
        let c = CutFit {
            outline: outline.clone(),
            fill,
            plain,
            edge,
            smooth,
            price,
        };
        if best
            .as_ref()
            .is_none_or(|b| c.plain + c.edge - c.price > b.plain + b.edge - b.price)
        {
            best = Some(c);
        }
    }
    best
}

/// A candidate of either kind, for the one greedy pass.
pub(super) enum Candidate {
    Stroke(Fit),
    Cut(CutFit),
}

impl Candidate {
    pub(super) fn net(&self) -> f64 {
        match self {
            Candidate::Stroke(f) => f.plain + f.edge - f.price,
            Candidate::Cut(c) => c.plain + c.edge - c.price,
        }
    }
}
