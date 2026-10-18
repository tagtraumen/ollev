//! Continuation of lines as evidence of a translucent layer: a line beneath (an arm's
//! contour, grout) that runs on under a layer keeps its geometry while each side's color b
//! becomes `k b + m`, so two linked parts of one smooth line pin the relation.

use glam::{DVec2, DVec3};

use super::raster::{
    ChebTree, Conn, Image, NOISE, arange, cheb, di_zenzo, dilate, label, length_px, line_pixels_at,
    linspace, map1, median, median_rows, np_sum, radians, rows_mean, skeletonize, sum_all3,
    unit_of,
};

mod chains;
mod edge_map;
mod fit;
mod links;
mod witnesses;

pub use chains::*;
pub use edge_map::*;
pub use fit::*;
pub use links::*;
pub use witnesses::*;

/// Gradients at the raster's own anti-aliasing scale at 330. The detector's lengths are in
/// pixels at 330 and scale by [`super::raster::unit_of`] (so hair strands at 1052 do not
/// become lines); its thresholds in levels stay.
pub const EDGE_SIGMA: f64 = 1.0;
/// A chain is fitted to within a pixel: the subpixel edge positions are good to a few
/// tenths, a pixel leaves room for the anti-aliasing's own wobble.
pub const FIT_TOL: f64 = 1.0;
/// Widths of ridges alike: within the slack of the wider, or the quantum in pixels.
pub const WIDTH_SLACK: f64 = 0.15;
pub const WIDTH_QUANTUM: f64 = 2.0;
/// The width of a step edge: its anti-aliasing and the gradient's sigma.
pub const EDGE_WIDTH: f64 = 2.0;
/// Suppression drops the weaker of two crossing edges within the stronger's gradient lobe,
/// about 3 sigma plus a pixel on either side, so a line crossing a layer's edge breaks over
/// twice that.
pub const JUNCTION_GAP: f64 = 2.0 * (3.0 * EDGE_SIGMA + 2.0);
/// The shortest chain kept: one with fewer pixels has no direction to continue.
pub const MIN_CHAIN: usize = 5;
/// Colors are read this far beyond the line's edge on each side: past its anti-aliased rim.
pub const SIDE_OFFSET: f64 = 2.0;
/// The tolerance of the crossing test.
pub const TOL: f64 = 3.0 * NOISE;
/// The window a corner is judged over, in pixels: past the staircase of the pixel grid.
pub const CORNER_REACH: usize = 3;

/// The ridge test's reaches: 1 and 2 px at 330, in the image's unit.
fn ridge_radii(u: f64) -> Vec<isize> {
    let mut r: Vec<isize> = [1.0, 2.0]
        .iter()
        .map(|v| ((v * u).round() as isize).max(1))
        .collect();
    r.dedup();
    r
}

/// A visible kink.
pub fn kink() -> f64 {
    radians(20.0)
}

/// A cubic's four control points.
pub type Bezier = [DVec2; 4];

// ----------------------------------------------------------------------------- one side

/// The pixel a sample point falls on, rounded half to even and kept in the frame.
pub fn pixel_of(img: &Image, q: DVec2) -> usize {
    let x = (q.x.round_ties_even() as i64).clamp(0, img.width as i64 - 1) as usize;
    let y = (q.y.round_ties_even() as i64).clamp(0, img.height as i64 - 1) as usize;
    y * img.width + x
}

/// The region proposal's seeds: a line running on into a layer drawn along it (a body's
/// contour into the dress) changes one side only and cannot pin k, but its changed inside
/// samples, unlike every outside one within the tolerance, lie under the layer.
pub struct OneSide {
    pub seeds: Vec<bool>,
    pub chains: Vec<Chain>,
    pub links: Vec<Link>,
    pub witnesses: Vec<Witness>,
    pub passing: Vec<usize>,
}

pub fn one_side_seeds(img: &Image) -> OneSide {
    let chains = build_chains(img);
    let links = all_links(&chains);
    let w = line_witnesses(img, &chains, &links);
    let mut seeds = vec![false; img.len()];
    for i in (0..w.len()).filter(|&i| w[i].why == Some("one side")) {
        let outside: Vec<DVec3> = w[i]
            .outside
            .iter()
            .map(|q| img.px[pixel_of(img, *q)])
            .collect();
        let tree = ChebTree::new(&outside);
        for q in &w[i].inside {
            let p = pixel_of(img, *q);
            if tree.nearest_dist(img.px[p]) > TOL {
                seeds[p] = true;
            }
        }
    }
    let passing = (0..w.len())
        .filter(|&i| w[i].why.is_none() && w[i].solved)
        .collect();
    OneSide {
        seeds,
        chains,
        links,
        witnesses: w,
        passing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tiny_image_has_chains_of_two_points_or_more() {
        // an icon of 32 px: the shortest chain rounds to one pixel and a corner cut leaves
        // pieces of a single point, which have no direction and no cubic
        let (w, h) = (32, 32);
        let mut seed = 7u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as f64 / 255.0
        };
        let blocks: Vec<DVec3> = (0..64)
            .map(|_| DVec3::new(next(), next(), next()))
            .collect();
        let px = (0..w * h)
            .map(|i| blocks[(i / w / 4) * 8 + (i % w) / 4])
            .collect();
        let img = Image {
            width: w,
            height: h,
            px,
        };
        for c in build_chains(&img) {
            assert!(c.p.len() >= 2 && !c.cubics.is_empty());
        }
    }

    #[test]
    fn a_line_of_points_is_one_cubic() {
        let p: Vec<DVec2> = (0..20)
            .map(|i| DVec2::new(i as f64, 0.5 * i as f64))
            .collect();
        let c = fit_cubics(&p, FIT_TOL);
        assert_eq!(c.len(), 1);
        assert!((c[0][3] - p[19]).length() < 1e-12);
    }

    #[test]
    fn a_right_angle_is_cut() {
        let mut p: Vec<DVec2> = (0..15).map(|i| DVec2::new(i as f64, 0.0)).collect();
        p.extend((1..15).map(|i| DVec2::new(14.0, i as f64)));
        assert_eq!(split_at_corners(&p, CORNER_REACH).len(), 2);
    }

    #[test]
    fn a_tint_over_a_line_solves() {
        // a line between 200 and 40 beneath a layer of k 0.5 over grey 100
        let (k, m) = (0.5, DVec3::splat(50.0 / 255.0));
        let a = [
            DVec3::new(200.0, 120.0, 40.0) / 255.0,
            DVec3::new(40.0, 160.0, 220.0) / 255.0,
        ];
        let b = [k * a[0] + m, k * a[1] + m];
        let s = solve_relation(&a, &b, 0.0);
        assert_eq!(s.why, None);
        assert!((s.k.unwrap() - 0.5).abs() < 1e-12);
    }
}
