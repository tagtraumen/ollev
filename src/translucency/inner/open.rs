//! The outside materials in the open per relation, and the colours they classify.

use crate::HashMap;

use super::{
    ChebGrid, Class, Conn, DVec3, InnerInputs, MATCH_TOL, NO_FIT, PixelFit, SHOW_THROUGH, VEIL_MAX,
    VOTE_GRID, cheb, label,
};

/// The outside under one relation: the colors that are the layer elsewhere, the materials
/// in the open, the sources and the inside colors scored.
pub struct Open {
    /// per material whether it is in the open
    pub open: Vec<bool>,
    pub(super) b: Vec<DVec3>,
    pub(super) bmat: Vec<usize>,
    /// the sources on a grid of cubes the side of the queries' bound, the tolerance over k
    pub(super) tree: ChebGrid,
    /// the sources' colors on the vote grid under the layer, with their material
    pub(super) images: Vec<(DVec3, usize)>,
    /// per inside color whether it is scored
    pub scored: Vec<bool>,
    pub total: f64,
}

/// The outside under a relation: outside colours peeling to other outside colours are the missed
/// layer; a material whose colours more region pixels read as other materials' images is no
/// source; an inside colour near the rest is background seen bare unless it is an image there.
pub fn open_at(inp: &InnerInputs, k: f64, m: DVec3) -> Open {
    let nm = inp.bmean.len();
    let tol = MATCH_TOL;
    let layer: Vec<bool> = inp
        .b
        .iter()
        .map(|&c| {
            !inp.btree.is_empty()
                && inp
                    .btree
                    .nearest((c - m) / k, tol / k)
                    .is_some_and(|(_, q)| cheb(c, inp.b[q]) > SHOW_THROUGH)
        })
        .collect();
    let mut first = build_open(inp, k, m, &layer, vec![true; nm], None);
    // Only the colours with an outside colour near them off the layer are classified: the
    // rules below read a colour's fit only through those, and most colours have none
    first.scored = inp
        .nearby
        .iter()
        .map(|near| near.iter().any(|&j| !layer[j as usize]))
        .collect();
    let near_of = |e: usize| {
        let mut near = vec![false; nm];
        for &j in &inp.nearby[e] {
            if !layer[j as usize] {
                near[inp.bmat[j as usize]] = true;
            }
        }
        near
    };
    // The alone test first: a colour it matches keeps that fit, the shading test only
    // reading the others. Per material the pixels read as other materials' images then lie
    // between the alone colours' count and that plus every colour left (pixel counts, exact)
    let mut pc = classify_colors(inp, &first, k, m, false);
    let (mut low, mut high) = (vec![0.0f64; nm], vec![0.0f64; nm]);
    for (e, f) in pc.iter().enumerate() {
        if !first.scored[e] {
            continue;
        }
        let near = near_of(e);
        for i in (0..nm).filter(|&i| near[i]) {
            if f.class == Class::None {
                high[i] += inp.weights[e];
            } else if i != f.mat {
                low[i] += inp.weights[e];
                high[i] += inp.weights[e];
            }
        }
    }
    // A material is open where its pixels outnumber those images: the bounds decide most;
    // the colours near an undecided one are tried over white and black
    let undecided: Vec<bool> = (0..nm)
        .map(|i| low[i] <= inp.bpx[i] && high[i] > inp.bpx[i])
        .collect();
    let lines = ShadingLines::new(&first.images);
    let mut tried = vec![false; pc.len()];
    let mut images = low;
    for e in 0..pc.len() {
        if !first.scored[e] || pc[e].class != Class::None {
            continue;
        }
        let near = near_of(e);
        if !(0..nm).any(|i| near[i] && undecided[i]) {
            continue;
        }
        pc[e] = shading_fit(&first, &lines, inp.colors[e]);
        tried[e] = true;
        if pc[e].class != Class::None {
            for i in (0..nm).filter(|&i| near[i] && i != pc[e].mat) {
                images[i] += inp.weights[e];
            }
        }
    }
    let open: Vec<bool> = (0..nm)
        .map(|i| {
            if undecided[i] {
                images[i] <= inp.bpx[i]
            } else {
                high[i] <= inp.bpx[i]
            }
        })
        .collect();
    // The scored rule reads a colour's fit only where it outweighs the open sources near it
    for e in 0..pc.len() {
        if !first.scored[e] || tried[e] || pc[e].class != Class::None {
            continue;
        }
        let near = open_near(inp, &open, &layer, e);
        if near > 0.0 && inp.weights[e] > near {
            pc[e] = shading_fit(&first, &lines, inp.colors[e]);
        }
    }
    build_open(inp, k, m, &layer, open, Some(&pc))
}

/// The pixels of the open sources near inside color `e`, the layer's colors left out.
fn open_near(inp: &InnerInputs, open: &[bool], layer: &[bool], e: usize) -> f64 {
    inp.nearby[e]
        .iter()
        .filter(|&&j| open[inp.bmat[j as usize]] && !layer[j as usize])
        .map(|&j| inp.bw[j as usize])
        .sum()
}

/// The outside with the given materials open: the sources (open, not the layer), their
/// images on the vote grid, and the inside colors scored (all with `pc` None, else by the
/// third rule of [`open_at`]).
pub(super) fn build_open(
    inp: &InnerInputs,
    k: f64,
    m: DVec3,
    layer: &[bool],
    open: Vec<bool>,
    pc: Option<&[PixelFit]>,
) -> Open {
    type Cell = ((usize, [i64; 3]), (DVec3, f64));
    let (mut b, mut bmat) = (Vec::new(), Vec::new());
    let mut grid: HashMap<(usize, [i64; 3]), (DVec3, f64)> = HashMap::default();
    for (j, (c, &r)) in inp.b.iter().zip(&inp.bmat).enumerate() {
        if open[r] && !layer[j] {
            b.push(*c);
            bmat.push(r);
            let g = (*c / VOTE_GRID).floor();
            let e = grid
                .entry((r, [g.x as i64, g.y as i64, g.z as i64]))
                .or_insert((DVec3::ZERO, 0.0));
            e.0 += inp.bw[j] * *c;
            e.1 += inp.bw[j];
        }
    }
    // the cells by material and place, for a deterministic order
    let mut cells: Vec<Cell> = grid.into_iter().collect();
    cells.sort_by_key(|c| c.0);
    let images = cells
        .into_iter()
        .map(|((r, _), (s, w))| (k * (s / w) + m, r))
        .collect();
    let scored: Vec<bool> = (0..inp.colors.len())
        .map(|e| match pc {
            None => true,
            Some(pc) => {
                let near = open_near(inp, &open, layer, e);
                near == 0.0 || (pc[e].class != Class::None && inp.weights[e] > near)
            }
        })
        .collect();
    let total = inp
        .weights
        .iter()
        .zip(&scored)
        .filter(|(_, s)| **s)
        .map(|(w, _)| *w)
        .sum();
    Open {
        open,
        tree: ChebGrid::new(&b, MATCH_TOL / k),
        b,
        bmat,
        images,
        scored,
        total,
    }
}

/// Per distinct inside color: the layer alone, else (with `shading`) white or black above
/// its image.
pub fn classify_colors(
    inp: &InnerInputs,
    op: &Open,
    k: f64,
    m: DVec3,
    shading: bool,
) -> Vec<PixelFit> {
    let tol = MATCH_TOL;
    // see [`ShadingLines`]: the images each color is tried against
    let lines = if shading {
        ShadingLines::new(&op.images)
    } else {
        ShadingLines::default()
    };
    inp.colors
        .iter()
        .zip(&op.scored)
        .map(|(&o, &scored)| {
            if !scored {
                return NO_FIT;
            }
            if !op.tree.is_empty()
                && let Some((d, j)) = op.tree.nearest((o - m) / k, tol / k)
                && cheb(o, op.b[j]) > SHOW_THROUGH
            {
                return PixelFit {
                    class: Class::Alone,
                    mat: op.bmat[j],
                    t: 0.0,
                    d: d * k,
                };
            }
            if !shading {
                return NO_FIT;
            }
            shading_fit(op, &lines, o)
        })
        .collect()
}

/// A colour as white or black above the nearest image the lines let through, the first of
/// equal residuals in the images' order; no fit where none is within the tolerance.
pub(super) fn shading_fit(op: &Open, lines: &ShadingLines, o: DVec3) -> PixelFit {
    let mut best = NO_FIT;
    let near = lines.near(o);
    for (i, &(q, r)) in op.images.iter().enumerate() {
        if !near[i] {
            continue;
        }
        if let Some((class, t, res)) = shading_residual(o, q)
            && res < best.d
        {
            best = PixelFit {
                class,
                mat: r,
                t,
                d: res,
            };
        }
    }
    best
}

/// The lines [`shading_residual`] moves each image along: a colour farther than root 3 times the
/// tolerance from both (the residual over white is at least `1 - o`'s distance from the line
/// along `1 - q`, over black `o`'s from the line along q) cannot match and is skipped exactly.
#[derive(Default)]
pub(super) struct ShadingLines {
    /// per image the unit directions of 1 - q and of q by channel, NaN where too short to
    /// normalize (those images are always tried)
    pub(super) white: [Vec<f32>; 3],
    pub(super) black: [Vec<f32>; 3],
    /// per image whether the color asked for last may match it
    pub(super) near: std::cell::RefCell<Vec<bool>>,
}

impl ShadingLines {
    pub(super) fn new(images: &[(DVec3, usize)]) -> Self {
        let unit = |v: DVec3| {
            let n = v.length();
            if v.dot(v) >= 1e-12 && n.is_finite() {
                (v / n).as_vec3()
            } else {
                glam::Vec3::NAN
            }
        };
        let split = |dirs: Vec<glam::Vec3>| {
            [
                dirs.iter().map(|d| d.x).collect(),
                dirs.iter().map(|d| d.y).collect(),
                dirs.iter().map(|d| d.z).collect(),
            ]
        };
        Self {
            white: split(images.iter().map(|(q, _)| unit(DVec3::ONE - *q)).collect()),
            black: split(images.iter().map(|(q, _)| unit(*q)).collect()),
            near: std::cell::RefCell::new(vec![false; images.len()]),
        }
    }

    /// Per image whether `o` may come within the tolerance of it over white or black.
    pub(super) fn near(&self, o: DVec3) -> std::cell::Ref<'_, Vec<bool>> {
        let limit =
            |aa: f32| (3.0 * MATCH_TOL * MATCH_TOL) as f32 * (1.0 + 1e-3) + 1e-5 * (1.0 + aa);
        let (w, b) = ((DVec3::ONE - o).as_vec3(), o.as_vec3());
        let (ww, bb) = (w.dot(w), b.dot(b));
        let (lw, lb) = (limit(ww), limit(bb));
        {
            let mut near = self.near.borrow_mut();
            let [wx, wy, wz] = &self.white;
            let [bx, by, bz] = &self.black;
            let lines = [wx, wy, wz, bx, by, bz].map(|v| &v[..]);
            let color = [w.x, w.y, w.z, ww, lw, b.x, b.y, b.z, bb, lb];
            // eight lanes where the processor has them: the test is a filter with a margin
            // far above any rounding, so how it rounds does not change what passes it
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: the processor has AVX2, checked just above
                unsafe { near_avx2(&mut near, lines, color) };
                drop(near);
                return self.near.borrow();
            }
            near_lanes(&mut near, lines, color);
        }
        self.near.borrow()
    }
}

/// [`ShadingLines::near`]'s test over every image: the directions by channel, white's then
/// black's, and the color's offsets from white and black, their squared lengths and limits.
#[inline(always)]
pub(super) fn near_lanes(near: &mut [bool], lines: [&[f32]; 6], color: [f32; 10]) {
    let n = near.len();
    let [wx, wy, wz, bx, by, bz] = lines.map(|v| &v[..n]);
    let [cwx, cwy, cwz, ww, lw, cbx, cby, cbz, bb, lb] = color;
    // within the limit, or NaN (from a direction too short to have one)
    let within = |x: f32, limit: f32| x <= limit || x.is_nan();
    for i in 0..n {
        let pw = cwx * wx[i] + cwy * wy[i] + cwz * wz[i];
        let pb = cbx * bx[i] + cby * by[i] + cbz * bz[i];
        near[i] = within(ww - pw * pw, lw) | within(bb - pb * pb, lb);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) fn near_avx2(near: &mut [bool], lines: [&[f32]; 6], color: [f32; 10]) {
    near_lanes(near, lines, color);
}

/// A color as white or black above an image `q` at an opacity t in (0, [`VEIL_MAX`]], the
/// inner layer moving the image by more than the tolerance (else it is the image itself at
/// a looser tolerance): the better of the two within the tolerance, its t and residual.
pub(super) fn shading_residual(o: DVec3, q: DVec3) -> Option<(Class, f64, f64)> {
    let tol = MATCH_TOL;
    let mut best: Option<(Class, f64, f64)> = None;
    for (class, w) in [(Class::White, DVec3::ONE), (Class::Black, DVec3::ZERO)] {
        let v = w - q;
        let vv = v.dot(v);
        if vv < 1e-12 {
            continue;
        }
        let t = (o - q).dot(v) / vv;
        if t <= 0.0 || t > VEIL_MAX || (t * v).abs().max_element() <= tol {
            continue;
        }
        let res = (o - q - t * v).abs().max_element();
        if res <= tol && best.is_none_or(|b| res < b.2) {
            best = Some((class, t, res));
        }
    }
    best
}

/// The components of a mask with an interior: a pixel whose eight neighbors (within the
/// frame) are all in the mask.
pub fn with_interior(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    let radius = crate::translucency::layer::interior_radius(width);
    // Found within the mask's box grown by the radius and a pixel, cut by the frame: every
    // pixel's neighborhood lies inside it or meets the frame where the image's does, so the
    // components kept are the ones the whole image keeps
    let (mut x0, mut y0, mut x1, mut y1) = (width, height, 0, 0);
    for (p, _) in mask.iter().enumerate().filter(|(_, v)| **v) {
        let (x, y) = (p % width, p / width);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
    }
    if x1 == 0 {
        return vec![false; mask.len()];
    }
    let pad = radius.max(1) + 1;
    let (x0, y0) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
    let (x1, y1) = ((x1 + pad).min(width), (y1 + pad).min(height));
    if (x0, y0, x1, y1) == (0, 0, width, height) {
        return with_interior_in(mask, width, height, radius);
    }
    let (w, h) = (x1 - x0, y1 - y0);
    let mut sub = Vec::with_capacity(w * h);
    for y in y0..y1 {
        sub.extend_from_slice(&mask[y * width + x0..y * width + x1]);
    }
    let kept = with_interior_in(&sub, w, h, radius);
    let mut out = vec![false; mask.len()];
    for y in y0..y1 {
        out[y * width + x0..y * width + x1].copy_from_slice(&kept[(y - y0) * w..(y - y0 + 1) * w]);
    }
    out
}

/// [`with_interior`] over the whole of `mask`, at the image's interior radius.
pub(super) fn with_interior_in(
    mask: &[bool],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<bool> {
    let (lab, n) = label(mask, width, height, Conn::Eight);
    let mut keep = vec![false; n + 1];
    if radius > 1 {
        for (p, d) in crate::translucency::raster::deep_in(mask, width, height, radius)
            .iter()
            .enumerate()
        {
            if *d {
                keep[lab[p] as usize] = true;
            }
        }
        keep[0] = false;
        return lab.iter().map(|&l| keep[l as usize]).collect();
    }
    for y in 0..height {
        for x in 0..width {
            let p = y * width + x;
            if !mask[p] || keep[lab[p] as usize] {
                continue;
            }
            let mut deep = true;
            'n: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let (yy, xx) = (y as isize + dy, x as isize + dx);
                    if yy < 0 || xx < 0 || yy >= height as isize || xx >= width as isize {
                        continue;
                    }
                    if !mask[yy as usize * width + xx as usize] {
                        deep = false;
                        break 'n;
                    }
                }
            }
            keep[lab[p] as usize] |= deep;
        }
    }
    keep[0] = false;
    lab.iter().map(|&l| keep[l as usize]).collect()
}
