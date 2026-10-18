//! The dress loop's inputs: the homothety's relation points on the proposal and its largest
//! components, grouped into colour families before any region is grown (a region per
//! family, k pinned on it), and the `fresh` ranker of a family's region.

use super::homothety;
use super::layer::Explained;
use super::lines::{OneSide, one_side_seeds};
use super::materials::{Palette, clear_of, min_material_px, outside_palette, region_palettes};
use super::raster::{ChebTree, Conn, DVec3, Image, SHOW_THROUGH, area_px, cheb, dilate, label};
use super::segment::equation_regions_default;

/// Relation points join a family when their color is within this of its first member's:
/// 12 levels.
pub const FAMILY_LEVELS: f64 = 12.0 / 255.0;
/// Components of the proposal tried on their own, the largest first, and their least size
/// in pixels at 330 (an area: by the unit squared at the image's width).
pub const COMPONENTS: usize = 6;
pub const COMPONENT_MIN_PX: usize = 30;
/// A family whose region holds fewer pixels (at 330, by the unit squared) has no region.
pub const REGION_MIN_PX: usize = 30;

/// What no iteration of the loop changes: the rims, the one-side seeds, the planar regions,
/// the proposal (the planar regions off the border holding a seed), its components largest
/// first.
pub struct FixedInputs {
    pub rim: Vec<bool>,
    pub one_side: OneSide,
    pub proposal: Vec<bool>,
    pub components: Vec<Vec<bool>>,
}

impl FixedInputs {
    /// With `rim` the picture's [`super::materials::rim_of`].
    pub fn new(img: &Image, rim: Vec<bool>) -> Self {
        let (w, h) = (img.width, img.height);
        let one_side = one_side_seeds(img);
        // no seed holds a planar region, so the proposal is empty without them
        if !one_side.seeds.iter().any(|s| *s) {
            return FixedInputs {
                rim,
                one_side,
                proposal: vec![false; w * h],
                components: Vec::new(),
            };
        }
        let eq = equation_regions_default(img);
        let n = eq.iter().max().map_or(0, |&l| l as usize + 1);
        let mut border = vec![false; n.max(1)];
        border[0] = true;
        for x in 0..w {
            border[eq[x] as usize] = true;
            border[eq[(h - 1) * w + x] as usize] = true;
        }
        for y in 0..h {
            border[eq[y * w] as usize] = true;
            border[eq[y * w + w - 1] as usize] = true;
        }
        let mut held = vec![false; n.max(1)];
        for (s, &l) in one_side.seeds.iter().zip(&eq) {
            if *s && !border[l as usize] {
                held[l as usize] = true;
            }
        }
        let proposal: Vec<bool> = eq.iter().map(|&l| held[l as usize]).collect();
        let (lab, nc) = label(&proposal, w, h, Conn::Four);
        let mut sizes = vec![0usize; nc];
        for &l in &lab {
            if l > 0 {
                sizes[l as usize - 1] += 1;
            }
        }
        let mut order: Vec<usize> = (0..nc).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(sizes[i]));
        let components = order
            .into_iter()
            .take(COMPONENTS)
            .filter(|&i| sizes[i] as f64 >= area_px(w, COMPONENT_MIN_PX as f64))
            .map(|i| lab.iter().map(|&l| l as usize == i + 1).collect())
            .collect();
        FixedInputs {
            rim,
            one_side,
            proposal,
            components,
        }
    }
}

/// A relation point: k, the color, the score over the best of its region's, its source.
#[derive(Clone, Debug)]
pub struct Point {
    pub k: f64,
    pub c: DVec3,
    pub score: f64,
    pub src: String,
}

/// The homothety's relation points on the proposal and each component where
/// [`homothety::gate`] passes, each list scaled to its best (shared colours kept), with the
/// proposal's outside palette (none where the proposal is empty).
pub fn material_points(img: &Image, fixed: &FixedInputs) -> (Vec<Point>, Option<Palette>) {
    let mut out = Vec::new();
    let mut proposal_outside = None;
    let regions = std::iter::once(("proposal".to_string(), &fixed.proposal)).chain(
        fixed
            .components
            .iter()
            .enumerate()
            .map(|(j, c)| (format!("component{}", j + 1), c)),
    );
    let debug = crate::diagnostics::debug();
    for (name, r) in regions {
        // an empty region has no cluster, so no pair for the gate
        if !r.iter().any(|v| *v) {
            continue;
        }
        let clock = crate::timing::Clock::now();
        // the inside first: where it holds nothing the gate could pair, no outside is read
        let pi = Palette::new(img, r, &fixed.rim, min_material_px(img));
        if !homothety::may_pass(&pi) {
            continue;
        }
        let po = outside_palette(img, r, &fixed.rim, None);
        let t_palettes = clock.elapsed().as_secs_f64();
        let gate = homothety::gate(&pi, &po, Some(&homothety::Ring::of(img, r, &fixed.rim)));
        if debug {
            eprintln!(
                "points: {name} gate {} in {:.2}s: {}",
                if gate.passes() { "passes" } else { "rejects" },
                clock.elapsed().as_secs_f64() - t_palettes,
                gate.describe()
            );
        }
        if std::ptr::eq(r, &fixed.proposal) {
            proposal_outside = Some(po);
        }
        if !gate.passes() {
            continue;
        }
        // The gate's fitted relations head the families: the region's joint relation first,
        // then each passing pair's own fit
        if let Some(rel) = &gate.relation {
            out.push(Point {
                k: rel.k,
                c: rel.c(),
                score: 2.0,
                src: name.clone(),
            });
        }
        for &i in &gate.passing {
            let f = &gate.pairs[i].fit;
            out.push(Point {
                k: f.k,
                c: f.c(),
                score: 1.5,
                src: name.clone(),
            });
        }
    }
    (out, proposal_outside)
}

/// Relation points of one color: every point within [`FAMILY_LEVELS`] of its first
/// member's color, members by score, and their summed score.
#[derive(Clone, Debug)]
pub struct Family {
    pub c: DVec3,
    pub members: Vec<Point>,
    pub sum: f64,
}

/// The points grouped by color alone, a point joining the first family whose color is
/// near; families by their summed score (ties in their order).
pub fn colour_families(points: &[Point]) -> Vec<Family> {
    let mut sorted: Vec<&Point> = points.iter().collect();
    sorted.sort_by(|a, b| (-a.score).total_cmp(&-b.score));
    let mut fams: Vec<Family> = Vec::new();
    for p in sorted {
        match fams.iter_mut().find(|f| cheb(f.c, p.c) <= FAMILY_LEVELS) {
            Some(f) => {
                f.members.push(p.clone());
                f.sum += p.score;
            }
            None => fams.push(Family {
                c: p.c,
                members: vec![p.clone()],
                sum: p.score,
            }),
        }
    }
    fams.sort_by(|a, b| (-a.sum).total_cmp(&-b.sum));
    fams
}

/// A region's palettes by color.
pub fn palettes(
    img: &Image,
    region: &[bool],
    rim: &[bool],
    drop_shared: bool,
) -> (Palette, Palette) {
    region_palettes(img, region, rim, drop_shared, None)
}

/// The share of the region's pixels whose color is no outside material's within the
/// show-through, the outside taken without the pixels the relation explains and two pixels
/// clear of the region, as the layer elsewhere (a missed part, a trim) is no open material.
pub fn fresh(img: &Image, region: &[bool], explained: &Explained, rim: &[bool]) -> f64 {
    let near = dilate(region, img.width, img.height, Conn::Four, clear_of(img));
    let open: Vec<bool> = explained
        .mask
        .iter()
        .zip(&near)
        .map(|(e, n)| !*e && !*n)
        .collect();
    let po = Palette::spatial(img, &open, rim, min_material_px(img), 0.5);
    let b: Vec<DVec3> = po
        .colors
        .iter()
        .zip(&po.label)
        .filter(|(_, l)| po.materials().contains(*l))
        .map(|(c, _)| *c)
        .collect();
    if b.is_empty() {
        return 0.0;
    }
    let tree = ChebTree::new(&b);
    // a function of the color: asked once per distinct color
    let mut memo: crate::HashMap<[u64; 3], bool> = crate::HashMap::default();
    let (mut n, mut hit) = (0usize, 0usize);
    for (p, r) in img.px.iter().zip(region) {
        if *r {
            n += 1;
            let key = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
            hit += *memo
                .entry(key)
                .or_insert_with(|| tree.nearest_dist(*p) > SHOW_THROUGH)
                as usize;
        }
    }
    hit as f64 / n.max(1) as f64
}
