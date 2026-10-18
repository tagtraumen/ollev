//! A relation's inner fit: the layer alone and its score.

use super::{
    Class, DVec3, InnerInputs, MATCH_TOL, NO_FIT, Open, PixelFit, SHOW_THROUGH, Tint, classify_colors, in_gamut_c, open_at,
    with_interior,
};

/// The fit of a relation with its inner layers: per pixel what explains it (inner layers
/// only within their kept regions), the tints, the materials and the score.
#[derive(Clone, Debug)]
pub struct InnerFit {
    pub k: f64,
    pub m: DVec3,
    pub px: Vec<PixelFit>,
    pub tints: Vec<Tint>,
    /// the materials counted, with the pixels each explains alone and through inner layers
    pub materials: Vec<(usize, f64, f64)>,
    /// the materials in the open
    pub open: Vec<bool>,
    /// the inside pixels scored
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))] // read by the lab
    pub total: f64,
    /// shares of the scored pixels: the layer alone, through white, black, tints;
    /// unexplained
    pub shares: [f64; 5],
    pub score: f64,
    /// the white's opacity per pixel of the region, rims included ([`fit_full`] only)
    pub white: Vec<f64>,
}

impl InnerFit {
    #[cfg(feature = "diagnostics")]
    pub fn n(&self) -> usize {
        self.materials.len()
    }
    pub fn c(&self) -> DVec3 {
        self.m / (1.0 - self.k)
    }
}

/// The per-color fits spread to the pixels, the inner layers kept where they have a region
/// (components with an interior), the materials counted and the score.
pub fn assemble(
    inp: &InnerInputs,
    op: &Open,
    k: f64,
    m: DVec3,
    per_color: &[PixelFit],
    tints: Vec<Tint>,
    per_pixel: Option<(&[PixelFit], bool)>,
) -> InnerFit {
    let (w, h) = (inp.width, inp.height);
    // Only the region's pixels off the rims can have a fit (every other pixel is no fit
    // throughout), so the passes below visit those alone, in the image's order: the sums
    // run over the same pixels in the same order as over the whole image
    let fitted: Vec<usize> = (0..inp.index.len())
        .filter(|&i| inp.index[i] != u32::MAX)
        .collect();
    let mut px = vec![NO_FIT; inp.index.len()];
    for &i in &fitted {
        let e = inp.index[i] as usize;
        if !op.scored[e] {
            continue;
        }
        let pc = per_color[e];
        px[i] = match per_pixel {
            // the per-pixel fit where the color has none, or first (over the layer alone)
            // when asked
            Some((pp, first))
                if pp[i].class != Class::None
                    && (pc.class == Class::None || (first && pc.class == Class::Alone)) =>
            {
                pp[i]
            }
            _ => pc,
        };
    }
    let mut classes = vec![Class::White, Class::Black];
    classes.extend((0..tints.len()).map(Class::Tint));
    let mut mask = vec![false; px.len()];
    for class in classes {
        for &i in &fitted {
            mask[i] = px[i].class == class;
        }
        let kept = with_interior(&mask, w, h);
        for &i in &fitted {
            if mask[i] && !kept[i] {
                px[i] = NO_FIT;
            }
        }
    }
    let nm = inp.bmean.len();
    let (mut alone, mut through) = (vec![0.0f64; nm], vec![0.0f64; nm]);
    let mut shares = [0.0f64; 5];
    let mut close = 0.0;
    let total = op.total.max(1e-9);
    for p in fitted.iter().map(|&i| &px[i]) {
        let slot = match p.class {
            Class::None => continue,
            Class::Alone => 0,
            Class::White => 1,
            Class::Black => 2,
            Class::Tint(_) => 3,
        };
        close += 1.0 - p.d / MATCH_TOL;
        shares[slot] += 1.0;
        if p.class == Class::Alone {
            alone[p.mat] += 1.0;
        } else {
            through[p.mat] += 1.0;
        }
    }
    shares[4] = op.total - shares[..4].iter().sum::<f64>();
    for s in &mut shares {
        *s /= total;
    }
    // A material counts by the pixels its image explains alone, since through a white or
    // black of free opacity every image reaches the pale colors; two materials are one where
    // their images lie within the show-through of each other
    let mut order: Vec<usize> = (0..nm).collect();
    order.sort_by(|&a, &b| alone[b].total_cmp(&alone[a]));
    let mut materials: Vec<(usize, f64, f64)> = Vec::new();
    for i in order {
        if alone[i] < inp.inside_palette.min_px {
            continue;
        }
        if materials
            .iter()
            .all(|c| (k * (inp.bmean[i] - inp.bmean[c.0])).abs().max_element() > SHOW_THROUGH)
        {
            materials.push((i, alone[i], through[i]));
        }
    }
    InnerFit {
        k,
        m,
        px,
        tints,
        score: close / total,
        materials,
        white: Vec::new(),
        open: op.open.clone(),
        total: op.total,
        shares,
    }
}

/// The layer's m refitted on the pixels it explains alone (each the color less k times its
/// open material's nearest color), every pixel weighed alike.
pub fn refit_alone(inp: &InnerInputs, op: &Open, k: f64, m: DVec3, pc: &[PixelFit]) -> DVec3 {
    let (mut s, mut n) = (DVec3::ZERO, 0.0);
    for (e, f) in pc.iter().enumerate() {
        if f.class != Class::Alone {
            continue;
        }
        let o = inp.colors[e];
        if let Some((_, j)) = op.tree.nearest((o - m) / k, MATCH_TOL / k) {
            s += inp.weights[e] * (o - k * op.b[j]);
            n += inp.weights[e];
        }
    }
    if n > 0.0 { s / n } else { m }
}

/// m refitted twice on the layer alone's pixels, in gamut, with the materials in the open at
/// the start (a refit moves m by a few levels).
fn refit_twice(inp: &InnerInputs, op: &Open, k: f64, m0: DVec3) -> DVec3 {
    let mut m = m0;
    for _ in 0..2 {
        let pc = classify_colors(inp, op, k, m, false);
        let m2 = refit_alone(inp, op, k, m, &pc);
        if !in_gamut_c(k, m2) {
            break;
        }
        m = m2;
    }
    m
}

/// The layer alone at (k, m), m refitted on the layer alone's pixels twice (for the lab).
#[cfg(any(test, feature = "diagnostics"))]
pub fn fit_at(inp: &InnerInputs, k: f64, m0: DVec3) -> InnerFit {
    let op = open_at(inp, k, m0);
    let m = refit_twice(inp, &op, k, m0);
    // where the refits left m where it started, the open materials are those already found
    // (`open_at` is a function of k and m alone)
    if m.to_array().map(f64::to_bits) == m0.to_array().map(f64::to_bits) {
        let pc = classify_colors(inp, &op, k, m, false);
        return assemble(inp, &op, k, m, &pc, Vec::new(), None);
    }
    fit_fixed(inp, k, m)
}

/// [`assemble`]'s score for the layer alone (no tints, no per-pixel fits), bit for bit,
/// without the per-pixel record, which [`scan`] builds only for each k's best relation.
pub(super) fn score_alone(inp: &InnerInputs, op: &Open, per_color: &[PixelFit]) -> f64 {
    let (w, h) = (inp.width, inp.height);
    let fitted: Vec<usize> = (0..inp.index.len())
        .filter(|&i| inp.index[i] != u32::MAX && op.scored[inp.index[i] as usize])
        .collect();
    let class_of = |i: usize| per_color[inp.index[i] as usize].class;
    let mut dropped = vec![false; fitted.len()];
    let mut mask = vec![false; inp.index.len()];
    for class in [Class::White, Class::Black] {
        for &i in &fitted {
            mask[i] = class_of(i) == class;
        }
        let kept = with_interior(&mask, w, h);
        for (n, &i) in fitted.iter().enumerate() {
            if mask[i] && !kept[i] {
                dropped[n] = true;
            }
            mask[i] = false;
        }
    }
    let mut close = 0.0;
    for (n, &i) in fitted.iter().enumerate() {
        let p = &per_color[inp.index[i] as usize];
        if dropped[n] || p.class == Class::None {
            continue;
        }
        close += 1.0 - p.d / MATCH_TOL;
    }
    close / op.total.max(1e-9)
}

/// [`fit_at`]'s relation and score, without the fit's record.
pub(super) fn fit_at_score(inp: &InnerInputs, k: f64, m0: DVec3) -> (DVec3, f64) {
    let op = open_at(inp, k, m0);
    let m = refit_twice(inp, &op, k, m0);
    let op = if m.to_array().map(f64::to_bits) == m0.to_array().map(f64::to_bits) {
        op
    } else {
        open_at(inp, k, m)
    };
    let pc = classify_colors(inp, &op, k, m, false);
    (m, score_alone(inp, &op, &pc))
}

/// The layer alone at (k, m) as given.
pub fn fit_fixed(inp: &InnerInputs, k: f64, m: DVec3) -> InnerFit {
    let op = open_at(inp, k, m);
    let pc = classify_colors(inp, &op, k, m, false);
    assemble(inp, &op, k, m, &pc, Vec::new(), None)
}
