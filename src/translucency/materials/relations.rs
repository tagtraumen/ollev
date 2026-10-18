//! Relations proposed by votes and refitted, and the palettes of a region.

use glam::DVec3;

use super::{
    Matcher, Palette, argsort_desc, clear_of, min_material_px, sorted_unique, unique_counts,
};
use crate::translucency::raster::{ChebTree, Conn, Image, SHOW_THROUGH, dilate};

/// The best cells of m at one k, as (votes, m): votes `m = o - k b` of inside samples against
/// outside samples, per cell the best inside share per outside material summed; and the
/// cells most of the inside's pixels reach.
pub fn vote_cells(mt: &Matcher, k: f64, per_k: usize) -> Vec<(f64, DVec3)> {
    let (ins, outs) = (mt.inside, mt.outside);
    if ins.materials().is_empty() || outs.materials().is_empty() {
        return Vec::new();
    }
    let cell = 2.0 * mt.tol;
    let s = mt.samples();
    let (no, nb) = (s.so.len() as i64, s.sb.len());
    let nmat = outs.materials().len() as i64;
    let nji = ins.materials().len() as i64;
    let mut full = Vec::with_capacity(s.so.len() * nb);
    for (oi, o) in s.so.iter().enumerate() {
        for (bi, b) in s.sb.iter().enumerate() {
            if mt.exclude_self && mt.same[s.sj[oi]] == mt.same[s.si[bi]] {
                continue;
            }
            let key = ((*o - k * *b) / cell).floor();
            let code =
                ((key.x as i64 + 600) * 1200 + (key.y as i64 + 600)) * 1200 + (key.z as i64 + 600);
            full.push((code * nmat + s.si[bi] as i64) * no + oi as i64);
        }
    }
    // distinct (cell, outside material, inside sample)
    let full = sorted_unique(full);
    // per (cell, outside material, inside material) the count of samples
    let key2: Vec<i64> = full
        .iter()
        .map(|f| (f / no) * nji + s.sj[(f % no) as usize] as i64)
        .collect();
    let (u2, c2) = unique_counts(key2);
    // per (cell, outside material) the max over inside materials, then summed per cell
    let mut cells: Vec<i64> = Vec::new();
    let mut tot: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < u2.len() {
        let cmat = u2[i] / nji;
        let mut best = f64::NEG_INFINITY;
        let mut j = i;
        while j < u2.len() && u2[j] / nji == cmat {
            best = best.max(c2[j] as f64 / s.nj[(u2[j] % nji) as usize]);
            j += 1;
        }
        let c = cmat / nmat;
        if cells.last() == Some(&c) {
            *tot.last_mut().unwrap() += best;
        } else {
            cells.push(c);
            tot.push(best);
        }
        i = j;
    }
    let mut best: Vec<usize> = argsort_desc(&tot).into_iter().take(per_k).collect();
    // and the cells most of the inside's pixels reach, through any material: a large
    // material's image holds many pixels where small flat colors hold a few
    let pc = sorted_unique(full.iter().map(|f| f / (nmat * no) * no + f % no).collect());
    let mut u5: Vec<i64> = Vec::new();
    let mut pv: Vec<f64> = Vec::new();
    for p in pc {
        let c = p / no;
        if u5.last() == Some(&c) {
            *pv.last_mut().unwrap() += s.sw[(p % no) as usize];
        } else {
            u5.push(c);
            pv.push(s.sw[(p % no) as usize]);
        }
    }
    for o in argsort_desc(&pv).into_iter().take(per_k) {
        let pos = cells.partition_point(|&c| c < u5[o]);
        if !best.contains(&pos) {
            best.push(pos);
        }
    }
    best.into_iter()
        .map(|b| {
            let cc = cells[b];
            let kz = cc.rem_euclid(1200) - 600;
            let ky = cc.div_euclid(1200).rem_euclid(1200) - 600;
            let kx = cc.div_euclid(1_440_000) - 600;
            (
                tot[b],
                (DVec3::new(kx as f64, ky as f64, kz as f64) + 0.5) * cell,
            )
        })
        .collect()
}

/// The pixels of `mask` whose color is, within the show-through, a color of one of the
/// outside palette's materials.
pub fn shared_colors(img: &Image, mask: &[bool], outside: &Palette) -> Vec<bool> {
    let b: Vec<DVec3> = outside
        .colors
        .iter()
        .zip(&outside.label)
        .filter(|(_, l)| outside.materials().contains(*l))
        .map(|(c, _)| *c)
        .collect();
    let mut out = vec![false; mask.len()];
    if b.is_empty() {
        return out;
    }
    let t = ChebTree::new(&b);
    for (i, m) in mask.iter().enumerate() {
        if *m {
            out[i] = t.nearest(img.px[i], SHOW_THROUGH).is_some();
        }
    }
    out
}

/// The palette of a region, with `drop_shared` without the colors within the show-through of
/// an outside material: those are the material seen bare, and would join the inside clusters
/// into one.
pub fn inside_palette(
    img: &Image,
    region: &[bool],
    outside: &Palette,
    rim: &[bool],
    drop_shared: bool,
) -> Palette {
    if drop_shared {
        let shared = shared_colors(img, region, outside);
        let r: Vec<bool> = region.iter().zip(&shared).map(|(r, s)| *r && !*s).collect();
        Palette::new(img, &r, rim, min_material_px(img))
    } else {
        Palette::new(img, region, rim, min_material_px(img))
    }
}

/// The inside palette of a region and the palette of what is outside it (the whole image
/// off the region, two pixels away from it past its rim, or the mask `outside`), the outside
/// clustered by place and color.
pub fn region_palettes(
    img: &Image,
    region: &[bool],
    rim: &[bool],
    drop_shared: bool,
    outside: Option<&[bool]>,
) -> (Palette, Palette) {
    let po = outside_palette(img, region, rim, outside);
    (inside_palette(img, region, &po, rim, drop_shared), po)
}

/// The outside palette of [`region_palettes`].
pub fn outside_palette(
    img: &Image,
    region: &[bool],
    rim: &[bool],
    outside: Option<&[bool]>,
) -> Palette {
    let near = dilate(region, img.width, img.height, Conn::Four, clear_of(img));
    let out: Vec<bool> = match outside {
        None => near.iter().map(|n| !n).collect(),
        Some(o) => o.iter().zip(&near).map(|(o, n)| *o && !*n).collect(),
    };
    Palette::spatial(img, &out, rim, min_material_px(img), 0.5)
}
