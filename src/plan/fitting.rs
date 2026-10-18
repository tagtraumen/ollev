//! How a planned shape is fitted: its simplification tolerances and completion.

use super::*;

/// The simplification tolerances: [`SIMPLIFY`], scaled down on images smaller than
/// [`TYPICAL_SIZE`] (8 px is a tenth of a 72 px emoji).
pub(super) fn tolerances(ctx: &Context) -> Vec<f32> {
    let scale = (ctx.frame / TYPICAL_SIZE).min(1.);
    SIMPLIFY.iter().map(|t| t * scale).collect()
}

/// Image size, as the square root of its area, at and above which the simplification
/// tolerances are what [`SIMPLIFY`] says.
pub(super) const TYPICAL_SIZE: f32 = 256.;

/// Spread per pixel below which a region is one color, as far as primitives are concerned:
/// about one 8-bit level per channel of noise.
pub(super) const FLAT_SPREAD: f64 = 0.01;

/// What completion proposes for a shape: primitives for smooth single-color objects, to
/// within half the scale they are smooth at (see [`smooth_at`]), and bridges across every
/// hidden stretch.
pub(super) fn completion(smooth: Option<i32>, allowance: f32, primitives: bool) -> Completion {
    Completion {
        primitives: primitives && smooth.is_some(),
        primitive_tolerance: crate::complete::PRIMITIVE_TOLERANCE
            .max(smooth.unwrap_or(1) as f32 / 2.),
        primitive_allowance: allowance,
    }
}

/// The error a region's blocks at scale `smooth` cost against the smooth shape they render,
/// granted to a primitive ([`Completion::primitive_allowance`]): perimeter times a quarter
/// block times the squared contrast with what it is drawn on. Nothing at scale one.
pub(super) fn block_allowance(
    ctx: &Context,
    hierarchy: &Hierarchy,
    node: u32,
    parent: Option<usize>,
    plan: &[Drawn],
    smooth: Option<i32>,
) -> f32 {
    let Some(scale) = smooth.filter(|&s| s > 1) else {
        return 0.;
    };
    let n = &hierarchy.nodes[node as usize];
    let own = ctx.rgb(n.stats.mean());
    let under = parent.map_or(own, |p| ctx.rgb(plan[p].color));
    n.perimeter as f32 * scale as f32 / 4. * own.distance_squared(under)
}

/// How a region's edge reads: the scale its rim is anti-aliased at (see
/// [`Hierarchy::antialias_scale`]), and that scale as `smooth` where the band along its outline
/// is also one colour within [`FLAT_SPREAD`], so a primitive may stand for it.
pub(super) fn smooth_at(
    ctx: &Context,
    hierarchy: &Hierarchy,
    node: u32,
    covered: &[u32],
) -> EdgeReading {
    let mean = ctx
        .rgb(hierarchy.nodes[node as usize].stats.mean())
        .as_dvec3();
    let scale = hierarchy.antialias_scale(node, &ctx.target, mean, ANTIALIASED);
    let Some(scale) = scale else {
        return EdgeReading {
            antialiased: None,
            smooth: None,
        };
    };
    // Anti-aliased: the band along the outline, past the blur. In blocks: the outermost block,
    // past its own edge pixel, since the next block in is the next step of the gradient
    let (band, radius) = if scale > 1 {
        (scale, 1)
    } else {
        (
            FLAT_BAND,
            hierarchy.nodes[node as usize].blur.ceil().max(1.) as i32,
        )
    };
    let stats = hierarchy.own_rim_stats(node, covered, radius, band);
    let spread = (stats.sse() / stats.n.max(1.)).sqrt();
    EdgeReading {
        antialiased: Some(scale),
        smooth: (spread < FLAT_SPREAD).then_some(scale),
    }
}

/// How a region's edge reads, see [`smooth_at`].
pub(super) struct EdgeReading {
    /// The scale the rim is anti-aliased at, `None` for a crisp edge; strokes use it whatever
    /// the colours inside (the wall's grid under glassware's glass).
    pub(super) antialiased: Option<i32>,
    /// That scale where the region is one colour along its outline too, so a primitive may
    /// stand for it.
    pub(super) smooth: Option<i32>,
}

/// Share of a region's rim that has to be mixed pixels for its edge to count as anti-aliased:
/// pixel art measures 0, anti-aliased edges 0.5 to 1.0, a pixel-art coin's banded rings 0.28
/// to 0.31 at the scale of their blocks, see [`Hierarchy::mixed_rim`].
pub(super) const ANTIALIASED: f64 = 0.2;

/// How far from the outline the band judged for flatness reaches, in pixels.
pub(super) const FLAT_BAND: i32 = 4;
