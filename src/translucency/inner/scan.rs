//! The pin: the k scan and its rows.

use super::{
    DVec3, InnerFit, InnerInputs, MATCH_TOL, Matcher, NOISE, arange, cheb, fit_at_score,
    fit_fixed, in_gamut_c, vote_cells,
};

/// One k of the scan: the best fit's score over the point matcher's candidates of m, and the
/// fit itself where the row leads the scan.
pub struct ScanRow {
    pub k: f64,
    pub score: Option<f64>,
    pub fit: Option<InnerFit>,
}

/// The scan: per k of the grid the point matcher's vote cells refitted (as the pin does)
/// give the candidates of m, each fitted with the layer alone; the best by score. Each row's
/// fit is shown to `look` and kept only on the leading row (the last of equal scores).
pub async fn scan(
    inp: &InnerInputs,
    ks: &[f64],
    mut look: impl FnMut(&InnerFit),
) -> Vec<ScanRow> {
    let mts = scan_matchers(inp);
    let mut rows: Vec<ScanRow> = Vec::with_capacity(ks.len());
    for &k in ks {
        crate::progress::poll().await;
        let fit = row_best(inp, &mts, k).map(|(m, _)| fit_fixed(inp, k, m));
        let score = fit.as_ref().map(|f| f.score);
        if let Some(f) = &fit {
            look(f);
        }
        let leads = score.is_some_and(|s| rows.iter().filter_map(|r| r.score).all(|b| s >= b));
        if leads {
            rows.iter_mut().for_each(|r| r.fit = None);
        }
        rows.push(ScanRow {
            k,
            score,
            fit: fit.filter(|_| leads),
        });
    }
    rows
}

/// The point matchers whose vote cells propose a scan's m: the inside palette with and
/// without its shared colours against the outside.
pub(super) fn scan_matchers(inp: &InnerInputs) -> Vec<Matcher<'_>> {
    [&inp.inside_palette, &inp.inside_all]
        .into_iter()
        .map(|pi| Matcher::new(pi, &inp.outside_palette, MATCH_TOL, false, false))
        .collect()
}

/// One k of the scan: the best candidate's m and score, each candidate scored without its
/// record (the score [`fit_fixed`] gives the same m, `open_at` being a function of k and m
/// alone).
pub(super) fn row_best(inp: &InnerInputs, mts: &[Matcher], k: f64) -> Option<(DVec3, f64)> {
    let mut best: Option<(DVec3, f64)> = None;
    let mut tried: Vec<DVec3> = Vec::new();
    for (mt, m0) in mts
        .iter()
        .flat_map(|mt| vote_cells(mt, k, 6).into_iter().map(move |c| (mt, c.1)))
    {
        let (m1, s) = mt.refit_m(k, m0, 3);
        // candidates within a level of one tried refit alike
        if s.n == 0 || !in_gamut_c(k, m1) || tried.iter().any(|t| cheb(*t, m1) < NOISE / 2.0) {
            continue;
        }
        tried.push(m1);
        let f = fit_at_score(inp, k, m1);
        if best.is_none_or(|b| f.1 > b.1) {
            best = Some(f);
        }
    }
    best
}

/// The pin's k grid: 0.30 to 0.90 by 0.01.
pub fn pin_grid() -> Vec<f64> {
    arange(0.30, 0.905, 0.01)
}

/// The pin's k grid at a coarse step, every fifth k of [`pin_grid`] (0.30 to 0.90 by 0.05): a
/// first look at whether any relation of the grid explains the region as a layer at all (see
/// [`crate::translucency::families::ranked_candidates`]), whose rows the pin then reuses.
pub fn coarse_grid() -> Vec<f64> {
    pin_grid().into_iter().step_by(5).collect()
}

/// How far from its last pin a region re-cut at that pin's relation is pinned again: two
/// coarse steps, as the cut moves the region's evidence little.
pub const REPIN_REACH: f64 = 0.10;

/// The pin again over the ks of [`pin_grid`] within [`REPIN_REACH`] of `k`, the last pin's.
pub async fn repin(inp: &InnerInputs, k: f64) -> Option<InnerFit> {
    let mut grid = pin_grid();
    grid.retain(|g| (g - k).abs() <= REPIN_REACH + 1e-9);
    pin_over(inp, grid, Vec::new()).await
}

/// The pin over [`pin_grid`], the rows of `done` (such as the coarse scan) taken as scanned
/// where their k equals a grid k to the bit; the best row wins, the last of equal scores in
/// grid order.
pub async fn pin_with_rows(inp: &InnerInputs, done: Vec<ScanRow>) -> Option<InnerFit> {
    pin_over(inp, pin_grid(), done).await
}

async fn pin_over(inp: &InnerInputs, mut grid: Vec<f64>, done: Vec<ScanRow>) -> Option<InnerFit> {
    grid.retain(|&k| !done.iter().any(|r| r.k.to_bits() == k.to_bits()));
    // the rest of the grid with the rows done, in the grid's order (the matchers built once)
    let mts = scan_matchers(inp);
    let mut rows: Vec<(f64, Option<ScanRow>)> = done.into_iter().map(|r| (r.k, Some(r))).collect();
    rows.extend(grid.into_iter().map(|k| (k, None)));
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    // the leading score, and the fit of the row that last took the lead with its fit kept: a
    // done row without its fit never ends the pin's best, as the scan's leader outscores it
    let (mut lead, mut best): (Option<f64>, Option<InnerFit>) = (None, None);
    for (k, done) in rows {
        crate::progress::poll().await;
        let (score, fit) = match done {
            Some(row) => (row.score, row.fit),
            // a row of the rest scored without its record, its fit built only where it takes
            // the lead: the score is the fit's, bit for bit (`the_score_alone_matches_the_fit`),
            // so the rows that do not lead need none
            None => {
                let fit = row_best(inp, &mts, k)
                    .filter(|&(_, s)| lead.is_none_or(|b| s >= b))
                    .map(|(m, _)| fit_fixed(inp, k, m));
                (fit.as_ref().map(|f| f.score), fit)
            }
        };
        if let Some(s) = score
            && lead.is_none_or(|b| s >= b)
        {
            lead = Some(s);
            if fit.is_some() {
                best = fit;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::super::{
        Image, ShadingLines, fit_at, shading_residual, with_interior, with_interior_in,
    };
    use super::*;

    /// The components kept within the mask's box are those kept over the whole image, at an
    /// interior radius of 1 and above, for blobs inside the image and against its frame.
    #[test]
    fn interior_in_the_box_matches_the_image() {
        let mut seed = 7u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as usize
        };
        for width in [330, 700, 1100] {
            let height = 40;
            let radius = crate::translucency::layer::interior_radius(width);
            for _ in 0..30 {
                let mut mask = vec![false; width * height];
                for _ in 0..=next() % 6 {
                    let (cx, cy) = (next() % width, next() % height);
                    let (rx, ry) = (1 + next() % 9, 1 + next() % 9);
                    for y in cy.saturating_sub(ry)..(cy + ry).min(height) {
                        for x in cx.saturating_sub(rx)..(cx + rx).min(width) {
                            mask[y * width + x] = next() % 13 != 0;
                        }
                    }
                }
                assert_eq!(
                    with_interior(&mask, width, height),
                    with_interior_in(&mask, width, height, radius)
                );
            }
        }
    }

    /// Every color and image the shading residual matches is one the lines let through,
    /// over random pairs and pairs placed on the segments toward white and black.
    #[test]
    fn shading_lines_keep_every_match() {
        let mut seed = 99u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut matched = 0;
        for _ in 0..20000 {
            let q = DVec3::new(next(), next(), next()) * 1.1 - 0.05;
            let images = [(q, 0usize)];
            let lines = ShadingLines::new(&images);
            let t = next() * 0.95;
            let w = if next() < 0.5 {
                DVec3::ONE
            } else {
                DVec3::ZERO
            };
            let jitter = DVec3::new(next(), next(), next()) * 2.0 - 1.0;
            for o in [
                DVec3::new(next(), next(), next()),
                q + t * (w - q) + jitter * MATCH_TOL,
            ] {
                if shading_residual(o, q).is_some() {
                    matched += 1;
                    assert!(lines.near(o)[0], "{o:?} {q:?}");
                }
            }
        }
        assert!(matched > 1000);
    }

    /// A layer at k 0.6 over a white ground and a shaded tan, with a white streak above:
    /// the score without the record is the fit's, and the relation the same, bit for bit.
    #[test]
    fn the_score_alone_matches_the_fit() {
        let (img, region, _) = scene();
        let rim = vec![false; img.len()];
        let inp = InnerInputs::new(&img, &region, &rim, None);
        let mut scored = 0;
        for k in pin_grid() {
            for dm in [-0.03, 0.0, 0.02] {
                let m = (1.0 - k) * DVec3::new(0.8, 1.0, 0.8) + dm;
                if !in_gamut_c(k, m) {
                    continue;
                }
                let full = fit_at(&inp, k, m);
                let (m2, score) = fit_at_score(&inp, k, m);
                assert_eq!(full.score.to_bits(), score.to_bits(), "k {k}");
                assert_eq!(full.m.to_array(), m2.to_array());
                scored += usize::from(score > 0.0);
            }
        }
        assert!(scored > 10);
    }

    fn scene() -> (Image, Vec<bool>, Vec<bool>) {
        let (w, h) = (80, 60);
        let (k, c) = (0.6, DVec3::new(0.8, 1.0, 0.8));
        let q = |v: DVec3| (v * 255.0).round() / 255.0;
        let mut px = Vec::with_capacity(w * h);
        let (mut region, mut streak) = (vec![false; w * h], vec![false; w * h]);
        for y in 0..h {
            for x in 0..w {
                let b = if y < 20 {
                    DVec3::ONE
                } else {
                    // the tan, shaded along y
                    DVec3::new(0.95, 0.84, 0.72) - (y - 20) as f64 * DVec3::new(0.002, 0.003, 0.004)
                };
                let mut o = b;
                if x >= 40 {
                    region[y * w + x] = true;
                    o = k * b + (1.0 - k) * c;
                    if (56..60).contains(&x) && y >= 30 {
                        streak[y * w + x] = true;
                        o = 0.6 * o + 0.4 * DVec3::ONE;
                    }
                }
                px.push(q(o));
            }
        }
        (
            Image {
                width: w,
                height: h,
                px,
            },
            region,
            streak,
        )
    }

    #[test]
    fn a_ring_of_pixels_has_no_interior() {
        let (w, h) = (9, 9);
        let ring: Vec<bool> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                (x == 2 || x == 6 || y == 2 || y == 6)
                    && (2..=6).contains(&x)
                    && (2..=6).contains(&y)
            })
            .collect();
        assert!(!with_interior(&ring, w, h).iter().any(|v| *v));
        let block: Vec<bool> = (0..w * h).map(|i| (i % w) < 4 && (i / w) < 4).collect();
        assert!(with_interior(&block, w, h)[0]);
    }
}
