//! Ranking the proposal's families by how many materials' images meet inside the region
//! (under a real layer several do), read per pixel; ties go to the inner score times the
//! pixels times `fresh` ([`Evidence::key`]).

use super::inner::{Class, InnerFit, InnerInputs};
use super::raster::{Conn, DVec3, SHOW_THROUGH, dilate};

/// The reach within which two images count as meeting: 3 px at 330 (outlines and
/// anti-aliased rims lie between the images).
pub fn reach(width: usize) -> usize {
    super::raster::length_px(width, 3.0)
}

/// Per pixel the counted material whose image the layer alone explains there, `None`
/// elsewhere: an uncounted material is the counted one whose image is nearest, when within
/// the show-through.
pub fn material_labels(inp: &InnerInputs, fit: &InnerFit) -> Vec<Option<usize>> {
    let counted: Vec<usize> = fit.materials.iter().map(|m| m.0).collect();
    let to: Vec<Option<usize>> = (0..inp.bmean.len())
        .map(|m| {
            if counted.contains(&m) {
                return Some(m);
            }
            counted
                .iter()
                .map(|&c| {
                    (
                        c,
                        (fit.k * (inp.bmean[m] - inp.bmean[c])).abs().max_element(),
                    )
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .filter(|(_, d)| *d <= SHOW_THROUGH)
                .map(|(c, _)| c)
        })
        .collect();
    fit.px
        .iter()
        .map(|p| (p.class == Class::Alone).then(|| to[p.mat]).flatten())
        .collect()
}

/// The mask dilated by a square of side `2 r + 1` (a Chebyshev reach).
fn dilate_square(mask: &[bool], width: usize, height: usize, r: usize) -> Vec<bool> {
    dilate(mask, width, height, Conn::Eight, r)
}

/// Where the images of the inner fit's materials meet.
#[derive(Clone, Copy, Debug, Default)]
pub struct Contact {
    /// the best pair's strength: the smaller of its two sides' pixels within reach of the
    /// other image
    pub pair: usize,
    /// the materials in pairs that meet (both sides above zero)
    pub meeting: usize,
}

/// Per pair of counted materials the pixels of each one's image within `reach` of the
/// other's.
pub fn contact(inp: &InnerInputs, fit: &InnerFit, reach: usize) -> Contact {
    contact_of(&material_labels(inp, fit), inp.width, inp.height, reach)
}

/// [`contact`] counting only the materials that continue out of the region: some of the
/// image lies within `reach` of an outside pixel of the material's own colour, the beneath
/// running on past the layer's edge. One pixel is enough, so the rule has no size.
pub fn contact_continued(inp: &InnerInputs, fit: &InnerFit, reach: usize) -> Contact {
    let (w, h) = (inp.width, inp.height);
    let labels = continued_labels(
        material_labels(inp, fit),
        &inp.region,
        &inp.px,
        &inp.bmean,
        (w, h),
        reach,
    );
    contact_of(&labels, w, h, reach)
}

/// The labels of the materials whose image lies within `reach` of an outside pixel of their
/// own mean color (within the show-through), the others' cleared.
fn continued_labels(
    mut labels: Vec<Option<usize>>,
    region: &[bool],
    px: &[DVec3],
    bmean: &[DVec3],
    (w, h): (usize, usize),
    reach: usize,
) -> Vec<Option<usize>> {
    let mut mats: Vec<usize> = labels.iter().flatten().copied().collect();
    mats.sort_unstable();
    mats.dedup();
    for m in mats {
        let own: Vec<bool> = (0..w * h)
            .map(|i| !region[i] && (px[i] - bmean[m]).abs().max_element() <= SHOW_THROUGH)
            .collect();
        let near = dilate_square(&own, w, h, reach);
        let continued = labels.iter().zip(&near).any(|(l, n)| *l == Some(m) && *n);
        if !continued {
            for l in labels.iter_mut().filter(|l| **l == Some(m)) {
                *l = None;
            }
        }
    }
    labels
}

/// [`contact`] over given per-pixel material labels.
fn contact_of(labels: &[Option<usize>], w: usize, h: usize, reach: usize) -> Contact {
    let mut mats: Vec<usize> = labels.iter().flatten().copied().collect();
    mats.sort_unstable();
    mats.dedup();
    let near: Vec<Vec<bool>> = mats
        .iter()
        .map(|&m| {
            let mask: Vec<bool> = labels.iter().map(|l| *l == Some(m)).collect();
            dilate_square(&mask, w, h, reach)
        })
        .collect();
    let side = |m: usize, other: usize| {
        labels
            .iter()
            .zip(&near[other])
            .filter(|(l, n)| **l == Some(m) && **n)
            .count()
    };
    let mut out = Contact::default();
    let mut meets = vec![false; mats.len()];
    for (i, &a) in mats.iter().enumerate() {
        for (j, &b) in mats.iter().enumerate().skip(i + 1) {
            let s = side(a, j).min(side(b, i));
            out.pair = out.pair.max(s);
            if s > 0 {
                meets[i] = true;
                meets[j] = true;
            }
        }
    }
    out.meeting = meets.iter().filter(|m| **m).count();
    out
}

/// What a family is ranked by.
#[derive(Clone, Copy, Debug)]
pub struct Evidence {
    /// the inner score times the region's pixels
    pub score_px: f64,
    pub fresh: f64,
    pub contact: Contact,
}

impl Evidence {
    /// The order: the materials whose images meet, then the inner score times pixels times
    /// `fresh`.
    pub fn key(&self) -> (usize, f64) {
        (self.contact.meeting, self.score_px * self.fresh)
    }
}

/// Indices of `ev` from the first-ranked family to the last (ties in their order).
pub fn order(ev: &[Evidence]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..ev.len()).collect();
    idx.sort_by(|&a, &b| {
        let (ka, kb) = (ev[a].key(), ev[b].key());
        kb.0.cmp(&ka.0).then(kb.1.total_cmp(&ka.1))
    });
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_dilation_reaches_diagonals() {
        let (w, h) = (7, 7);
        let mut m = vec![false; w * h];
        m[3 * w + 3] = true;
        let d = dilate_square(&m, w, h, 2);
        assert!(d[w + 1] && d[5 * w + 5] && d[3 * w + 1]);
        assert!(!d[0] && !d[6 * w + 6] && !d[3 * w]);
        assert_eq!(d.iter().filter(|v| **v).count(), 25);
    }

    /// A white ground (material 0) above a tan (material 1) and a layer over the right half
    /// from row `top`, the images labelled by what lies beneath; with `blob`, a white piece
    /// beneath the layer wholly inside its region over the tan.
    fn labelled(top: usize, blob: bool) -> (usize, usize) {
        let (w, h) = (40, 30);
        let (white, tan) = (DVec3::ONE, DVec3::new(0.95, 0.84, 0.72));
        let (mut labels, mut region, mut px) = (Vec::new(), Vec::new(), Vec::new());
        for y in 0..h {
            for x in 0..w {
                let inside = x >= 20 && y >= top;
                let m = usize::from(
                    !(y < 10 || (blob && (26..32).contains(&x) && (18..24).contains(&y))),
                );
                let b = [white, tan][m];
                region.push(inside);
                labels.push(inside.then_some(m));
                px.push(if inside {
                    0.6 * b + 0.4 * DVec3::new(0.8, 1.0, 0.8)
                } else {
                    b
                });
            }
        }
        let all = contact_of(&labels, w, h, 3).meeting;
        let kept = continued_labels(labels, &region, &px, &[white, tan], (w, h), 3);
        (all, contact_of(&kept, w, h, 3).meeting)
    }

    /// Both materials run on out of a layer over the ground and the tan: both count.
    #[test]
    fn materials_beneath_continue_out_of_the_region() {
        assert_eq!(labelled(0, false), (2, 2));
    }

    /// A white piece enclosed by the tan's image, the white ground beyond the reach of the
    /// region, meets the tan but does not continue out of the region: the tan alone is left.
    #[test]
    fn an_enclosed_second_material_does_not_count() {
        assert_eq!(labelled(15, true), (2, 0));
    }

    #[test]
    fn more_materials_meeting_first() {
        let ev = |score_px, fresh, meeting| Evidence {
            score_px,
            fresh,
            contact: Contact { pair: 1, meeting },
        };
        let ev = [
            ev(1000.0, 0.9, 0),
            ev(400.0, 0.5, 3),
            ev(800.0, 0.6, 2),
            ev(300.0, 0.9, 2),
        ];
        assert_eq!(order(&ev), vec![1, 2, 3, 0]);
    }
}
