//! Outlines of regions as cubics.

use crate::tracer::Cubic;
use crate::tracer::fit::FitParams;
use crate::tracer::planar::{self, ChainUse};
use glam::{IVec2, Vec2};

use crate::HashMap;
use crate::hierarchy::Hierarchy;

/// Turning angle above which an outline keeps a sharp corner.
pub const CORNER_ANGLE: f32 = 70f32.to_radians();
pub const FIT: FitParams = FitParams {
    corner_angle: CORNER_ANGLE,
    tolerance: 0.5,
    corner_window: 4,
    separate_turns: false,
};
/// [`FIT`] keeping corners of separate turns within the window apart, see
/// [`FitParams::separate_turns`].
pub const FIT_TURNS: FitParams = FitParams {
    separate_turns: true,
    ..FIT
};
/// How far a simplified outline may stray from the fitted one; refinement pulls it back where
/// the image needs it.
const SIMPLIFY: f32 = 1.;

/// The outline of hierarchy node `node` fitted with `fit`, with the positions of the sharp
/// corners the trace found.
pub fn outline_with(
    hierarchy: &Hierarchy,
    node: u32,
    height: usize,
    fit: &FitParams,
) -> (Vec<Vec<Cubic>>, Vec<Vec2>) {
    let n = &hierarchy.nodes[node as usize];
    outline_corners(n.min, n.max, hierarchy.width(), height, fit, |i| {
        hierarchy.contains(node, i)
    })
}

/// The outline of the pixels within `min..max` that are `inside`, as closed loops of cubics.
pub fn outline_of(
    min: IVec2,
    max: IVec2,
    width: usize,
    height: usize,
    fit: &FitParams,
    inside: impl Fn(usize) -> bool,
) -> Vec<Vec<Cubic>> {
    outline_corners(min, max, width, height, fit, inside).0
}

/// [`outline_of`] with the positions of the sharp corners the trace found: the turns within
/// the chains, and the joints between chains that turn by the corner angle.
pub fn outline_corners(
    min: IVec2,
    max: IVec2,
    width: usize,
    height: usize,
    fit: &FitParams,
    inside: impl Fn(usize) -> bool,
) -> (Vec<Vec<Cubic>>, Vec<Vec2>) {
    // A pixel of margin, except at the image border, so the region only meets the corners of the
    // crop where those are corners of the image too
    let min = (min - 1).max(IVec2::ZERO);
    let max = (max + 1).min(IVec2::new(width as i32, height as i32));
    let size = (max - min).as_uvec2();
    let (w, h) = (size.x as usize, size.y as usize);
    let labels: Vec<u32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| u32::from(inside((min.y as usize + y) * width + min.x as usize + x)))
        .collect();

    let map = planar::build(&labels, w, h, 2);
    let mut fitted: HashMap<usize, Vec<Cubic>> = HashMap::default();
    let mut corners: Vec<Vec2> = Vec::new();
    let offset = min.as_vec2();
    let loops: Vec<Vec<Cubic>> = map.loops[1]
        .iter()
        .map(|uses| {
            uses.iter()
                .flat_map(|&ChainUse { chain, reversed }| {
                    let cubics = fitted.entry(chain).or_insert_with(|| {
                        let c = &map.chains[chain];
                        let (cubics, found) =
                            crate::tracer::fit::fit_chain_corners(&c.points, c.is_closed(), fit);
                        corners.extend(found.iter().map(|&p| p + offset));
                        crate::tracer::beautify::beautify_chain(&cubics, SIMPLIFY, fit.corner_angle)
                            .into_iter()
                            .map(|c| c.map(|p: Vec2| p + offset))
                            .collect()
                    });
                    let oriented: Vec<Cubic> = if reversed {
                        cubics
                            .iter()
                            .rev()
                            .map(|c| [c[3], c[2], c[1], c[0]])
                            .collect()
                    } else {
                        cubics.clone()
                    };
                    oriented
                })
                .collect()
        })
        .collect();
    // Joints between chains that turn sharply are corners too
    for cubics in &loops {
        for j in 0..cubics.len() {
            let sharp =
                crate::complete::joint_turn(cubics, j).is_some_and(|t| t.abs() > fit.corner_angle);
            if sharp && !corners.iter().any(|c| c.distance(cubics[j][0]) < 0.5) {
                corners.push(cubics[j][0]);
            }
        }
    }
    (loops, corners)
}
