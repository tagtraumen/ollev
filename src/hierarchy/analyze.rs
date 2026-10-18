//! The first read of a picture: its context and region hierarchy.

use crate::context::Context;
use crate::timing::Timings;

/// Share of the usual segmentation boundary cost for the leaves of the hierarchy, small enough
/// that details are still separate regions.
const LEAF_BOUNDARY_COST: f64 = 0.05;

/// The context and region hierarchy of an image.
pub fn analyze(
    rgb: &[u8],
    width: usize,
    height: usize,
    detail: f64,
    timings: &mut Timings,
) -> (Context, super::Hierarchy) {
    let mut ctx = Context::new(rgb, width, height, detail);
    let mut params = crate::tracer::Params::with_detail(detail);
    // The segmentation only gives the leaves; the hierarchy decides what merges beyond them
    params.segment.boundary_cost *= LEAF_BOUNDARY_COST;
    params.segment.cancelled = || crate::progress::poll();
    crate::progress::stage("analyze");
    let colors = timings.time("smooth", || {
        let colors = crate::tracer::segmentation_colors(rgb, width, height, &params);
        if let Some(antialias) = &params.antialias {
            ctx.unmixed = Some(std::sync::Arc::new(
                crate::tracer::unmixed_colors(rgb, width, height, antialias)
                    .iter()
                    .map(|c| c.as_vec3())
                    .collect(),
            ));
        }
        colors
    });
    // Merging starts from the runs of equal colors, which every merge would keep together
    let (seeds, count) = timings.time("seed", || {
        crate::tracer::segment::equal_color_regions(rgb, width, height)
    });
    let mut labels = timings.time("segment", || {
        crate::tracer::segment::segment_seeded(&colors, width, height, &params.segment, &seeds, count)
            .labels
    });
    // A cancelled trace goes on with one region, which costs nothing to build on
    if crate::progress::cancelled() {
        labels.iter_mut().for_each(|l| *l = 0);
    }
    let count = labels.iter().max().map_or(0, |&l| l as usize + 1);
    let count = timings.time("dissolve", || {
        super::dissolve_blends(&mut labels, count, &colors, width, height)
    });
    let hierarchy = timings.time("tree", || {
        super::Hierarchy::build(labels, count, &colors, width, height)
    });
    (ctx, hierarchy)
}
