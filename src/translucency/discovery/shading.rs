//! Shading layers: shadows and highlights as black and white layers.

use super::*;

/// A shadow `o = s b` per channel (black at opacity `1 - s`) or a highlight
/// `1 - o = s (1 - b)` (white): in the space `t = b`, or `t = 1 - b`, both are `t_o = s t_b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shade {
    Shadow,
    Highlight,
}

impl Shade {
    pub fn space(self, v: Vec3) -> Vec3 {
        match self {
            Shade::Shadow => v,
            Shade::Highlight => Vec3::ONE - v,
        }
    }

    /// The color of the layer: black or white.
    pub fn color(self) -> Vec3 {
        match self {
            Shade::Shadow => Vec3::ZERO,
            Shade::Highlight => Vec3::ONE,
        }
    }
}

/// A shading layer: its kind, its scale per channel, its evidence.
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    pub kind: Shade,
    pub s: Vec3,
    pub score: f32,
}

/// `slope_fits` with one scale per channel, and never steeper inside: beneath a shading layer
/// the gradient shows at `s < 1` times its slope.
pub(super) fn slope_fits_channels(inside: &Plane, outside: &Plane, s: Vec3) -> bool {
    let diff = ((inside.gx - s * outside.gx).length_squared()
        + (inside.gy - s * outside.gy).length_squared())
    .sqrt();
    let scaled = ((s * outside.gx).length_squared() + (s * outside.gy).length_squared()).sqrt();
    diff <= 0.3 * inside.slope().max(scaled) + SLOPE_MIN
        && inside.slope() <= outside.slope() + SLOPE_MIN
}

/// Per contact point the scale per channel, `t_o / t_b`; a channel with almost nothing to
/// scale (under 4 noise levels: black beneath a shadow, white beneath a highlight) says
/// nothing of its scale and takes the mean of the others. None where no channel can.
pub(super) fn shade_ratio(to: Vec3, tb: Vec3) -> Option<Vec3> {
    let meas: Vec<bool> = (0..3).map(|c| tb[c] >= 4. * NOISE).collect();
    let count = meas.iter().filter(|&&m| m).count();
    if count == 0 {
        return None;
    }
    let mean = (0..3)
        .filter(|&c| meas[c])
        .map(|c| to[c] / tb[c].max(1e-6))
        .sum::<f32>()
        / count as f32;
    Some(Vec3::from_array(std::array::from_fn(|c| {
        if meas[c] {
            to[c] / tb[c].max(1e-6)
        } else {
            mean
        }
    })))
}

/// Per directed pair: x is y under the shading layer at every contact point to the noise and
/// the planes' uncertainty, the two differ, the texture of x is s times y's and the slopes
/// are s times y's per channel.
pub(super) fn shade_pair_fits(lv: &Leaves, kind: Shade, s: Vec3) -> Vec<bool> {
    let sm = s.element_sum() / 3.;
    let unc = &lv.uncertainty;
    lv.pairs
        .iter()
        .map(|p| {
            let (x, y) = (p.x as usize, p.y as usize);
            let slack = 2. * NOISE + unc[x] + sm * unc[y];
            let points = &lv.points[p.points.clone()];
            lv.interior[x]
                && points
                    .iter()
                    .all(|q| (kind.space(q.o) - s * kind.space(q.b)).abs().max_element() <= slack)
                && points
                    .iter()
                    .any(|q| (q.o - q.b).abs().max_element() > 2. * NOISE)
                && (lv.texture[x] - sm * lv.texture[y]).abs() <= slack
                && slope_fits_channels(&lv.plane[x], &lv.plane[y], s)
        })
        .collect()
}

/// How many shading hypotheses are refined and cut, and connected groups taken per relation.
pub(super) const SHADE_TOP: usize = 40;
pub(super) const SHADE_GROUPS: usize = 6;

/// Shading layers: a Hough over per-channel scales (steps of 0.025) of contact points moved
/// one way in every channel, refined by least squares through the origin, verified by the
/// cut, each connected group a layer, scored by evidence times the mean scale.
pub fn shading_pass(lv: &Leaves) -> Vec<Layer> {
    let graph = Graph::of(lv);
    let mut raw: Vec<(Shade, Vec3, f32)> = Vec::new();
    for kind in [Shade::Shadow, Shade::Highlight] {
        let mut cells: HashMap<IVec3, (HashSet<IVec3>, bool, f32, Vec3, usize)> =
            HashMap::default();
        let mut order: Vec<IVec3> = Vec::new();
        let mut ok = 0;
        for p in &lv.pairs {
            if !lv.interior[p.x as usize] {
                continue;
            }
            let (pa, pb) = (&lv.plane[p.x as usize], &lv.plane[p.y as usize]);
            for q in &lv.points[p.points.clone()] {
                if (q.o - q.b).abs().max_element() <= 2. * NOISE {
                    continue;
                }
                let (to, tb) = (kind.space(q.o), kind.space(q.b));
                let Some(s) = shade_ratio(to, tb) else {
                    continue;
                };
                // One direction in every channel, and something left of the beneath in each
                if (0..3).any(|c| to[c] > tb[c] + NOISE)
                    || !(0..3).any(|c| tb[c] - to[c] > 2. * NOISE)
                    || (0..3).any(|c| !(s[c] > 0.02 && s[c] < 0.98))
                    || !slope_fits_channels(pa, pb, s)
                {
                    continue;
                }
                ok += 1;
                let key = (s / 0.025).round().as_ivec3();
                let graded = (pb.gx.length_squared() + pb.gy.length_squared()).sqrt() > SLOPE_MIN;
                let cell = cells.entry(key).or_insert_with(|| {
                    order.push(key);
                    (HashSet::default(), false, 0., Vec3::ZERO, 0)
                });
                cell.0.insert(bin(tb, 4.));
                cell.1 |= graded;
                cell.2 += q.w;
                cell.3 += s;
                cell.4 += 1;
            }
        }
        if ok < 2 {
            continue;
        }
        for key in order {
            let (bs, graded, l, sum, n) = &cells[&key];
            if bs.len() < 2 && !graded {
                continue;
            }
            raw.push((kind, *sum / *n as f32, *l));
        }
    }
    raw.sort_by(|a, b| b.2.total_cmp(&a.2));
    // Merged within a Hough step only: layers with close scales (the jellyfish's three
    // bells) would refit to a mean that fits none
    let mut merged: Vec<(Shade, Vec3)> = Vec::new();
    for (kind, s, _) in raw {
        if !merged
            .iter()
            .any(|&(k2, s2)| k2 == kind && (s - s2).abs().max_element() <= 0.025)
        {
            merged.push((kind, s));
        }
    }
    let mut out: Vec<(Layer, Vec<u32>)> = Vec::new();
    for &(kind, s0) in merged.iter().take(SHADE_TOP) {
        let mut s = s0;
        let keep = |p: &crate::translucency::leaves::Pair,
                    q: &crate::translucency::leaves::Point| {
            lv.interior[p.x as usize] && (q.o - q.b).abs().max_element() > 2. * NOISE
        };
        for _ in 0..2 {
            let fits = shade_pair_fits(lv, kind, s);
            let (mut num, mut den, mut count) = (Vec3::ZERO, Vec3::ZERO, 0);
            for (p, &f) in lv.pairs.iter().zip(&fits) {
                if !f {
                    continue;
                }
                for q in &lv.points[p.points.clone()] {
                    let (to, tb) = (kind.space(q.o), kind.space(q.b));
                    if keep(p, q)
                        && (to - s * tb).abs().max_element()
                            <= 2. * NOISE + lv.uncertainty[p.x as usize]
                    {
                        num += q.w * to * tb;
                        den += q.w * tb * tb;
                        count += 1;
                    }
                }
            }
            if count < 2 {
                break;
            }
            let mut s2 = num / den.max(Vec3::splat(1e-9));
            // A channel with nothing beneath to scale keeps the others' mean
            let weak: Vec<bool> = (0..3).map(|c| den[c] < 1e-6).collect();
            let strong: Vec<f32> = (0..3).filter(|&c| !weak[c]).map(|c| s2[c]).collect();
            if !strong.is_empty() {
                let mean = strong.iter().sum::<f32>() / strong.len() as f32;
                for c in 0..3 {
                    if weak[c] {
                        s2[c] = mean;
                    }
                }
            }
            if (0..3).any(|c| !(s2[c] > 0.02 && s2[c] < 0.98)) {
                break;
            }
            s = s2;
        }
        let fits = shade_pair_fits(lv, kind, s);
        let mut sup: Vec<Support> = Vec::new();
        for (p, &f) in lv.pairs.iter().zip(&fits) {
            if !f {
                continue;
            }
            let pts = &lv.points[p.points.clone()];
            let n = pts.len() as f32;
            let x = p.x as usize;
            sup.push(Support {
                a: p.x,
                b: p.y,
                len: p.len as f32,
                bg: pts.iter().map(|q| q.b).sum::<Vec3>() / n,
                img: pts.iter().map(|q| q.o).sum::<Vec3>() / n,
                span: lv.plane[x].slope() * (lv.size[x] as f32).sqrt(),
            });
        }
        // Every connected group under the relation is a layer of its own: the best cut's
        // group, then the best of what is left
        for _ in 0..SHADE_GROUPS {
            if sup.is_empty() {
                break;
            }
            let (ev, members) = best_cut(&sup, &graph);
            if ev <= 0. || members.is_empty() {
                break;
            }
            let group: HashSet<u32> = members.iter().map(|p| p.a).collect();
            sup.retain(|p| !group.contains(&p.a));
            if let Some(layer) = shade_group(lv, kind, s, &members, ev) {
                let mut under: Vec<u32> = group.into_iter().collect();
                under.sort_unstable();
                out.push((layer, under));
            }
        }
    }
    out.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
    // The same layer found from two cells: one kind, close scales and shared leaves
    let mut final_: Vec<(Layer, Vec<u32>)> = Vec::new();
    for (layer, under) in out {
        if final_.iter().any(|(l2, u2)| {
            l2.kind == layer.kind
                && (layer.s - l2.s).abs().max_element() <= 0.04
                && under.iter().any(|x| u2.contains(x))
        }) {
            continue;
        }
        final_.push((layer, under));
    }
    final_.into_iter().map(|(l, _)| l).collect()
}

/// One group of leaves under a shading relation as a layer, if the backgrounds show through:
/// they vary past the noise after the layer's scale, and the images vary at least half as
/// much as the scaled backgrounds.
pub(super) fn shade_group(
    lv: &Leaves,
    kind: Shade,
    s: Vec3,
    members: &[Support],
    ev: f32,
) -> Option<Layer> {
    let chosen: HashSet<(u32, u32)> = members.iter().map(|p| (p.a, p.b)).collect();
    let mut tb: [Vec<f32>; 3] = Default::default();
    let mut to: [Vec<f32>; 3] = Default::default();
    for p in &lv.pairs {
        if !chosen.contains(&(p.x, p.y)) {
            continue;
        }
        for q in &lv.points[p.points.clone()] {
            let (o, b) = (kind.space(q.o), kind.space(q.b));
            for c in 0..3 {
                tb[c].push(b[c]);
                to[c].push(o[c]);
            }
        }
    }
    if tb[0].len() < 2 {
        return None;
    }
    let spread_of = |v: &mut [Vec<f32>; 3]| -> f32 {
        (0..3)
            .map(|c| {
                let hi = percentile(&mut v[c], 95.);
                let lo = percentile(&mut v[c], 5.);
                hi - lo
            })
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let spread = spread_of(&mut tb);
    let ospread = spread_of(&mut to);
    let sm = s.element_sum() / 3.;
    if sm * spread <= SHOW_THROUGH || ospread <= 0.5 * sm * spread {
        return None;
    }
    Some(Layer {
        kind,
        s,
        score: ev * sm,
    })
}
