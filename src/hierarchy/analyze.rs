//! The first read of a picture: its context and region hierarchy.

// Panic ratchet: the lints this module has yet to meet, each removed as it is cleared
// clean of these, which the parent module still expects
#![cfg_attr(not(test), deny(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::disallowed_macros,
    clippy::indexing_slicing,
    clippy::too_many_lines,
))]

use crate::context::Context;
use crate::error::TraceError;
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
) -> Result<(Context, super::Hierarchy), TraceError> {
    let mut ctx = Context::new(rgb, width, height, detail)?;
    let mut params = crate::tracer::Params::with_detail(detail);
    // The segmentation only gives the leaves; the hierarchy decides what merges beyond them
    params.segment.boundary_cost *= LEAF_BOUNDARY_COST;
    crate::progress::stage("analyze").await;
    let work = crate::plan::Work::new(width.saturating_mul(height));
    crate::progress::enter(&work.analyze(None), true);
    crate::progress::step("smooth").await;
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
    crate::progress::step("seed").await;
    // Merging starts from the runs of equal colors, which every merge would keep together
    let (seeds, count) = timings
        .time("seed", async || {
            crate::tracer::segment::equal_color_regions(rgb, width, height)
        })
        .await;
    // the merging takes the time of the runs it starts from
    let [_, _, segment, _, _] = work.analyze(Some(count));
    crate::progress::weigh(&[segment, ("analyze", work.reading(Some(count)))]);
    crate::progress::step("segment").await;
    let segment = &params.segment;
    let mut labels = timings
        .time("segment", async || {
            crate::tracer::segment::segment_seeded(&colors, width, height, segment, &seeds, count)
                .await
                .labels
        })
        .await;
    let count = labels.iter().max().map_or(0, |&l| l as usize + 1);
    crate::progress::step("dissolve").await;
    let count = timings
        .time("dissolve", async || {
            super::dissolve_blends(&mut labels, count, &colors, width, height)
        })
        .await;
    crate::progress::step("tree").await;
    let hierarchy = timings
        .time("tree", async || {
            super::Hierarchy::build(labels, count, &colors, width, height).await
        })
        .await?;
    crate::progress::leave();
    Ok((ctx, hierarchy))
}
