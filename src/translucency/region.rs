//! What a layer's relation says of each pixel, for the region of the layer
//! ([`super::region_graph`]): the pixels the relation explains bare are the layer's core, the
//! pixels on an outside material's dichromatic polyline are that material seen bare.

use crate::HashMap;

use super::materials::{MATCH_TOL, Palette};
use super::raster::{ChebTree, Conn, DVec3, Image, SHOW_THROUGH, label};

/// What the relation says of each pixel: the core (explained bare, cleaned of specks under a
/// material's size), the pixels an outside material seen bare (on its dichromatic polyline),
/// and each material's base colour (the densest cell of its colours).
pub struct Evidence {
    pub core: Vec<bool>,
    pub bare: Vec<bool>,
    pub bases: Vec<DVec3>,
}

/// The [`Evidence`] of the relation `k`, `c` over the materials of `outside`; none when
/// `outside` has no material. `min_px` is the least pixels of a material at this width.
pub fn evidence(img: &Image, outside: &Palette, k: f64, c: DVec3, min_px: f64) -> Option<Evidence> {
    let (w, h) = (img.width, img.height);
    let n = w * h;
    let m = (1.0 - k) * c;
    // the images of the outside materials' colours, and every outside colour for the bare test
    let mut is_material = vec![false; outside.clusters.len()];
    for ci in outside.materials() {
        is_material[ci] = true;
    }
    let (mut mat_colors, mut images) = (Vec::new(), Vec::new());
    for (col, &l) in outside.colors.iter().zip(&outside.label) {
        if is_material[l] {
            mat_colors.push(*col);
            images.push(k * *col + m);
        }
    }
    if images.is_empty() {
        return None;
    }
    let image_tree = ChebTree::new(&images);
    // A pixel is an outside material seen bare where it lies within the show-through of the
    // material's polyline: its base (densest) colour with its highlight toward white and shade
    // toward black. Any of its colours would be too loose, as a shaded material covers the cube.
    let bases: Vec<DVec3> = outside
        .materials()
        .map(|ci| {
            let cl = &outside.clusters[ci];
            let x = &outside.colors[cl.start..cl.end];
            let w = &outside.weights[cl.start..cl.end];
            // the densest cell of three levels
            let mut cells: HashMap<(i64, i64, i64), f64> = HashMap::default();
            for (c, ww) in x.iter().zip(w) {
                let q = (*c * 255.0).round();
                *cells
                    .entry((q.x as i64 / 3, q.y as i64 / 3, q.z as i64 / 3))
                    .or_insert(0.0) += ww;
            }
            let best = cells
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k)
                .unwrap_or((0, 0, 0));
            let (mut sum, mut tot) = (DVec3::ZERO, 0.0);
            for (c, ww) in x.iter().zip(w) {
                let q = (*c * 255.0).round();
                if (q.x as i64 / 3, q.y as i64 / 3, q.z as i64 / 3) == best {
                    sum += *c * *ww;
                    tot += ww;
                }
            }
            sum / tot.max(1e-9)
        })
        .collect();
    let on_segment = |p: DVec3, a: DVec3, b: DVec3| -> f64 {
        let d = b - a;
        let dd = d.dot(d).max(1e-12);
        let t = ((p - a).dot(d) / dd).clamp(0.0, 1.0);
        (p - (a + t * d)).abs().max_element()
    };
    let mut core = vec![false; n];
    let mut bare = vec![false; n];
    // both readings are functions of the colour: asked once per distinct colour
    let mut memo: HashMap<[u64; 3], (bool, bool)> = HashMap::default();
    for i in 0..n {
        let p = img.px[i];
        (core[i], bare[i]) = *memo
            .entry(p.to_array().map(f64::to_bits))
            .or_insert_with(|| {
                let core = image_tree
                    .nearest(p, MATCH_TOL)
                    .is_some_and(|(_, j)| (p - mat_colors[j]).abs().max_element() > SHOW_THROUGH);
                let bare = bases.iter().any(|&b| {
                    on_segment(p, b, DVec3::ONE) <= SHOW_THROUGH
                        || on_segment(p, b, DVec3::ZERO) <= SHOW_THROUGH
                });
                (core, bare)
            });
    }
    // the core cleaned of specks
    let (lab, count) = label(&core, w, h, Conn::Four);
    let mut sizes = vec![0usize; count + 1];
    for &l in &lab {
        sizes[l as usize] += 1;
    }
    for i in 0..n {
        if core[i] && (sizes[lab[i] as usize] as f64) < min_px {
            core[i] = false;
        }
    }
    Some(Evidence { core, bare, bases })
}

/// `inside` with its holes filled: what is not inside and not connected to the frame is
/// enclosed.
pub fn fill_holes(inside: &[bool], w: usize, h: usize) -> Vec<bool> {
    let n = w * h;
    let outside_mask: Vec<bool> = inside.iter().map(|&v| !v).collect();
    let (olab, ocount) = label(&outside_mask, w, h, Conn::Four);
    let mut touches_frame = vec![false; ocount + 1];
    for i in 0..n {
        let (x, y) = (i % w, i / w);
        if outside_mask[i] && (x == 0 || y == 0 || x + 1 == w || y + 1 == h) {
            touches_frame[olab[i] as usize] = true;
        }
    }
    (0..n)
        .map(|i| inside[i] || !touches_frame[olab[i] as usize])
        .collect()
}
