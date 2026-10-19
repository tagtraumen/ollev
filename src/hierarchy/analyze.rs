//! The first read of a picture: its context and region hierarchy.

use crate::context::Context;
use crate::timing::Timings;

/// Share of the usual segmentation boundary cost for the leaves of the hierarchy, small enough
/// that details are still separate regions.
const LEAF_BOUNDARY_COST: f64 = 0.05;

/// The context and region hierarchy of an image.
pub async fn analyze(
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
    crate::progress::stage("analyze").await;
    let colors = timings
        .time("smooth", async || {
            let colors = crate::tracer::segmentation_colors(rgb, width, height, &params).await;
            if let Some(antialias) = &params.antialias {
                ctx.unmixed = Some(std::sync::Arc::new(
                    crate::tracer::unmixed_colors(rgb, width, height, antialias)
                        .await
                        .iter()
                        .map(|c| c.as_vec3())
                        .collect(),
                ));
            }
            colors
        })
        .await;
    // Merging starts from the runs of equal colors, which every merge would keep together
    let (seeds, count) = timings
        .time("seed", async || {
            crate::tracer::segment::equal_color_regions(rgb, width, height)
        })
        .await;
    let segment = &params.segment;
    let mut labels = timings
        .time("segment", async || {
            crate::tracer::segment::segment_seeded(&colors, width, height, segment, &seeds, count)
                .await
                .labels
        })
        .await;
    let count = labels.iter().max().map_or(0, |&l| l as usize + 1);
    let count = timings
        .time("dissolve", async || {
            super::dissolve_blends(&mut labels, count, &colors, width, height)
        })
        .await;
    let hierarchy = timings
        .time("tree", async || {
            super::Hierarchy::build(labels, count, &colors, width, height).await
        })
        .await;
    (ctx, hierarchy)
}
