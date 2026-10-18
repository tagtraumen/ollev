//! Linework for the target edges no shape of a traced scene draws (the contour between two
//! thighs on skin): a ridge becomes a stroke along its centreline, a step a cut that splits
//! the shape spanning it, each kept when its gain, edges weighted, exceeds its price.

use glam::{DMat3, DMat4, DVec2, DVec3, DVec4};

use crate::edges::{EdgeField, REACH, SIGMA, edges_of, line_centres};
use crate::translucency::lines::{
    Bezier, Chain, FIT_TOL, Kind, LinkMode, MIN_CHAIN, SIDE_OFFSET, find_links, fit_cubics,
    make_chain, trace_chains,
};
use crate::translucency::raster::{
    Conn, Image, NOISE, di_zenzo, dilate, label, length_px, map1, skeletonize, unit_of,
};
use crate::translucency::vocabulary::{colour_lines_at, flatten, outline};

mod chains;
mod closing;
mod cuts;
mod paint;
mod planner;

pub use chains::*;
use closing::*;
use cuts::*;
pub use paint::*;
pub use planner::*;

/// The price of a segment and a path in squared RGB error (`Context::segment_cost` and
/// `path_cost`), the edge term's weight `lambda` (0 for the plain error).
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub segment_cost: f64,
    pub path_cost: f64,
    pub edge: f64,
}

/// A cut's fill: flat, or linear along `from` to `to` (user space) between two colours.
#[derive(Clone, Copy, Debug)]
pub enum CutFill {
    Flat(DVec3),
    Linear {
        from: DVec2,
        to: DVec2,
        c0: DVec3,
        c1: DVec3,
    },
}

impl CutFill {
    pub fn at(&self, p: DVec2) -> DVec3 {
        match *self {
            CutFill::Flat(c) => c,
            CutFill::Linear { from, to, c0, c1 } => {
                let d = to - from;
                let t = ((p - from).dot(d) / d.length_squared().max(1e-12)).clamp(0.0, 1.0);
                c0 + t * (c1 - c0)
            }
        }
    }
}

/// The target's edges the render does not reproduce within the anti-aliasing reach: lines
/// with no render line near, steps (off the lines' flanks) with no render edge of either
/// kind near; and every edge of either image, which bounds a cut.
pub struct Missing {
    pub lines: Vec<bool>,
    pub steps: Vec<bool>,
    pub barrier: Vec<bool>,
}

pub fn missing(target: &Image, render: &Image) -> Missing {
    let (w, h) = (target.width, target.height);
    let u = unit_of(w);
    let t_step = edges_of(target, SIGMA);
    let t_line = line_centres(target, u);
    let r_step = edges_of(render, SIGMA);
    let r_line = line_centres(render, u);
    let near_line = dilate(&r_line, w, h, Conn::Eight, REACH);
    let r_any: Vec<bool> = r_step.iter().zip(&r_line).map(|(a, b)| *a || *b).collect();
    let near_any = dilate(&r_any, w, h, Conn::Eight, REACH);
    // a hairline's gradient has a lobe either side of its ridge: those step edges are the
    // line's flanks, not boundaries of their own
    let flank = dilate(&t_line, w, h, Conn::Eight, REACH);
    Missing {
        lines: (0..w * h).map(|i| t_line[i] && !near_line[i]).collect(),
        steps: (0..w * h)
            .map(|i| t_step[i] && !flank[i] && !near_any[i])
            .collect(),
        barrier: (0..w * h)
            .map(|i| t_step[i] || t_line[i] || r_any[i])
            .collect(),
    }
}

/// Strokes and cuts for the target edges the scene does not draw, each tried at its
/// [`positions`] under the objective (edge term at `lambda`, discounted) and kept by `keep`,
/// best net first, refitted on what was taken.
pub fn into_scene(
    ctx: &crate::Context,
    scene: &mut crate::Scene,
    field: &EdgeField,
    lambda: f64,
    keep: crate::translucency::vocabulary::Keep,
) -> Planned {
    let (w, h) = (ctx.width, ctx.height);
    let target = Image::from_target(&ctx.target, w);
    let full = crate::Window::full(w, h);
    let comp: Vec<DVec3> = scene
        .render(full, 0..scene.shapes.len(), [1.0; 3])
        .iter()
        .map(|v| v.as_dvec3())
        .collect();
    let render = Image {
        width: w,
        height: h,
        px: comp.clone(),
    };
    let weak = weak_edges(&target);
    let mut miss = missing(&target, &render);
    let mut out = Planned {
        missing_lines: miss.lines.iter().filter(|m| **m).count(),
        missing_steps: miss.steps.iter().filter(|m| **m).count(),
        ..Planned::default()
    };
    // the missing lines followed along their ridges before anything is priced, so that a fold
    // is one candidate, not its fragments
    let followed = follow_ridges(&target, &miss.lines);
    // the followed lines bound a cut as the detected ones do
    for (b, f) in miss.barrier.iter_mut().zip(&followed) {
        *b |= *f;
    }
    let mut chains = chains_of(&target, &miss.lines, true);
    let steps = chains_of(&target, &miss.steps, false);
    chains.extend(steps);
    let lines = linked(&chains);
    // the followed lines are candidates beside the detected ones, not in their place (a row
    // of stitches is followed through its gaps into one solid line); only the chains the
    // following changed
    let seen: crate::HashSet<(i64, i64, i64)> = chains
        .iter()
        .map(|c| (c.p[0].x as i64, c.p[0].y as i64, (c.length * 4.0) as i64))
        .collect();
    let followed_chains: Vec<Chain> = chains_of(&target, &followed, true)
        .into_iter()
        .filter(|c| !seen.contains(&(c.p[0].x as i64, c.p[0].y as i64, (c.length * 4.0) as i64)))
        .collect();
    let followed_lines = linked(&followed_chains);
    let opt = Options {
        segment_cost: ctx.segment_cost,
        path_cost: ctx.path_cost,
        edge: lambda,
    };
    // candidates by the line they answer: a line's alternatives (through its gaps, as its
    // dashes, its members together or alone) compete with each other, the best taken first
    // and the rest refitted with it drawn
    let mut groups: Vec<Vec<Candidate>> = Vec::new();
    for l in lines.iter().filter(|l| l.ridge) {
        if crate::progress::poll() {
            break;
        }
        let g: Vec<Candidate> = stroke_candidates(&target, &comp, field, &weak, &opt, &chains, l)
            .into_iter()
            .map(Candidate::Stroke)
            .collect();
        groups.push(g);
    }
    for l in &followed_lines {
        if crate::progress::poll() {
            break;
        }
        let g: Vec<Candidate> =
            stroke_candidates(&target, &comp, field, &weak, &opt, &followed_chains, l)
                .into_iter()
                .map(Candidate::Stroke)
                .collect();
        groups.push(g);
    }
    let cuts: Vec<&Linked> = lines.iter().filter(|l| !l.ridge).collect();
    if !cuts.is_empty() {
        let mut a = areas(&miss.barrier, w, h);
        let mut closed = 0;
        let found = closing_cut_candidates(
            &target,
            &comp,
            field,
            &weak,
            &opt,
            &chains,
            &cuts,
            &mut a,
            &mut closed,
        );
        groups.extend(found.into_iter().map(|c| vec![Candidate::Cut(c)]));
        out.closed = closed;
    }
    out.proposed = groups.iter().map(Vec::len).sum();
    // a candidate that pays only over the target's own ground (`Fit::own`) had a shape stand
    // in for it, and is weighed with that shape left out or refitted ([`joint_move`])
    let mark_price = |c: &Candidate| match c {
        Candidate::Stroke(f) => {
            (f.paths.iter().map(Vec::len).sum::<usize>() as f64 + 1.0) * ctx.segment_cost
        }
        Candidate::Cut(_) => f64::INFINITY,
    };
    let stands_in = |c: &Candidate| match c {
        Candidate::Stroke(f) => f.own > mark_price(c),
        Candidate::Cut(_) => false,
    };
    for g in &mut groups {
        g.retain(|c| c.net() > 0.0 || stands_in(c));
    }
    groups.retain(|g| !g.is_empty());
    out.paying = groups.iter().map(Vec::len).sum();
    let best = |g: &[Candidate]| g.iter().map(Candidate::net).fold(f64::MIN, f64::max);
    groups.sort_by(|a, b| best(b).total_cmp(&best(a)));
    // the areas a line's cut takes ([`line_cuts`]), made on the first that asks
    let mut line_areas: Option<Areas> = None;
    // each area's cut, fitted once: many lines bound the same area
    let mut fitted_cuts: crate::HashMap<u32, Option<CutFit>> = Default::default();
    for mut g in groups {
        if crate::progress::poll() {
            break;
        }
        loop {
            // (index, choice, the variant of the scene it goes with)
            let mut chosen: Option<(usize, Choice, Variant)> = None;
            let mut seen: Option<(usize, Choice)> = None;
            for (i, c) in g.iter().enumerate() {
                let choice = match c {
                    Candidate::Stroke(f) => {
                        try_stroke(ctx, scene, &target, field, lambda, f, Variant::Same)
                    }
                    Candidate::Cut(cf) => try_cut(ctx, scene, &target, field, lambda, cf),
                };
                let Some(ch) = choice else {
                    continue;
                };
                if ch.kept(keep, ctx.segment_cost)
                    && chosen.as_ref().is_none_or(|b| ch.net > b.1.net)
                {
                    chosen = Some((i, ch, Variant::Same));
                } else if seen.as_ref().is_none_or(|b| ch.net > b.1.net) {
                    seen = Some((i, ch));
                }
            }
            // the joint moves of the candidate worth most over its own ground
            let eligible =
                g.iter()
                    .enumerate()
                    .filter(|(_, c)| stands_in(c))
                    .max_by(|a, b| match (a.1, b.1) {
                        (Candidate::Stroke(x), Candidate::Stroke(y)) => x.own.total_cmp(&y.own),
                        _ => std::cmp::Ordering::Equal,
                    });
            if let Some((i, Candidate::Stroke(f))) = eligible {
                let alone = match (&chosen, &seen) {
                    (Some((j, ch, _)), _) if *j == i => Some(ch),
                    (_, Some((j, ch))) if *j == i => Some(ch),
                    _ => None,
                };
                let window = window_of(&alone.map_or_else(Vec::new, |a| a.shapes.clone()), w, h);
                let window = if alone.is_some() {
                    window
                } else {
                    let win = Win::around(&f.flat, f.width / 2.0 + 2.0, w, h);
                    crate::Window {
                        x: win.x0,
                        y: win.y0,
                        width: win.w(),
                        height: win.y1 - win.y0,
                    }
                };
                let joints: Vec<(Choice, Variant)> = stand_ins(scene, window)
                    .into_iter()
                    .filter_map(|k| joint_move(ctx, scene, &target, field, lambda, f, alone, k))
                    .collect();
                for (ch, v) in joints {
                    if ch.kept(keep, ctx.segment_cost)
                        && chosen.as_ref().is_none_or(|b| ch.net > b.1.net)
                    {
                        chosen = Some((i, ch, v));
                    }
                }
            }
            // the line's best form over a cut along it, the cut on either side ([`line_cuts`])
            let line = match (&chosen, &seen) {
                (Some((i, ch, Variant::Same)), Some((j, s))) => {
                    if s.net > ch.net {
                        Some((*j, s))
                    } else {
                        Some((*i, ch))
                    }
                }
                (Some((i, ch, Variant::Same)), None) => Some((*i, ch)),
                (Some(_), Some((j, s))) | (None, Some((j, s))) => Some((*j, s)),
                _ => None,
            };
            let mut cuts: Vec<(usize, Choice)> = Vec::new();
            if let Some((i, line)) = line
                && let Candidate::Stroke(f) = &g[i]
                && line.kind == crate::translucency::vocabulary::MarkKind::Line
            {
                let a = line_areas.get_or_insert_with(|| areas(&miss.barrier, w, h));
                for ch in line_cuts(
                    ctx,
                    scene,
                    &target,
                    &comp,
                    field,
                    &opt,
                    f,
                    line,
                    a,
                    &mut fitted_cuts,
                ) {
                    cuts.push((i, ch));
                }
            }
            {
                for (i, ch) in cuts {
                    if ch.kept(keep, ctx.segment_cost)
                        && chosen.as_ref().is_none_or(|b| ch.net > b.1.net)
                    {
                        chosen = Some((i, ch, Variant::Same));
                    }
                }
            }
            let Some((i, ch, v)) = chosen else {
                break;
            };
            match ch.kind {
                crate::translucency::vocabulary::MarkKind::Cut => out.cuts += 1,
                _ => out.strokes += 1,
            }
            let mut position = ch.position;
            match v {
                Variant::Same => {}
                Variant::Drop(k) => {
                    scene.shapes.remove(k);
                    scene.change();
                    if k < position {
                        position -= 1;
                    }
                    out.dropped += 1;
                }
                Variant::Paint(k, fill) => {
                    scene.set_fill_at(k, fill);
                    out.refitted += 1;
                }
            }
            if position < scene.shapes.len() {
                out.below += 1;
            }
            insert_at(scene, position, ch.shapes);
            g.remove(i);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb8(px: &[DVec3]) -> Vec<u8> {
        px.iter()
            .flat_map(|c| c.to_array().map(|v| (v * 255.0).round() as u8))
            .collect()
    }

    fn rectangle(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Vec<crate::tracer::Cubic>> {
        let c = [
            glam::Vec2::new(x0, y0),
            glam::Vec2::new(x1, y0),
            glam::Vec2::new(x1, y1),
            glam::Vec2::new(x0, y1),
        ];
        vec![
            (0..4)
                .map(|k| {
                    let (a, b) = (c[k], c[(k + 1) % 4]);
                    [a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.), b]
                })
                .collect(),
        ]
    }

    /// The planner's pass over a scene of one skin-coloured background, for the dark line
    /// on skin it lacks (the thighs' contour): one stroke goes into the stack, kept by the
    /// plan's keep rule, and the render shows the line.
    #[test]
    fn the_planner_draws_a_missing_line_into_the_stack() {
        let (w, h) = (330, 40);
        let skin = DVec3::new(0.95, 0.80, 0.70);
        let dark = DVec3::new(0.55, 0.35, 0.30);
        let px: Vec<DVec3> = (0..w * h)
            .map(|i| {
                if i % w == 30 && (5..35).contains(&(i / w)) {
                    dark
                } else {
                    skin
                }
            })
            .collect();
        let ctx = crate::Context::new(&rgb8(&px), w, h, 1.0);
        let mut scene = crate::Scene::with_background(w, h, skin.as_vec3());
        let field = EdgeField::of_target(&ctx.target, w, h);
        let out = into_scene(&ctx, &mut scene, &field, 1.0, crate::plan::keep_mark);
        assert_eq!(out.strokes, 1, "{out:?}");
        assert_eq!(scene.shapes.len(), 2);
        assert!(scene.shapes[1].stroke.is_some());
        let r = scene.render(crate::Window::full(w, h), 0..2, [1.0; 3]);
        assert!(
            (r[20 * w + 30].as_dvec3() - dark).length() < 0.15,
            "{:?}",
            r[20 * w + 30]
        );
    }

    /// A band of the line's darkness spread wide, the trace's stand-in for a thin dark line
    /// on skin (the thighs' crotch wedge): the line does not pay over it, and with the band
    /// left out or refitted it does; the joint move is taken.
    #[test]
    fn a_shape_that_stood_in_for_a_line_goes() {
        let (w, h) = (330, 40);
        let skin = DVec3::new(0.95, 0.80, 0.70);
        let dark = DVec3::new(0.45, 0.25, 0.20);
        let px: Vec<DVec3> = (0..w * h)
            .map(|i| {
                if i % w == 30 && (5..35).contains(&(i / w)) {
                    dark
                } else {
                    skin
                }
            })
            .collect();
        let ctx = crate::Context::new(&rgb8(&px), w, h, 1.0);
        let mut scene = crate::Scene::with_background(w, h, skin.as_vec3());
        // the line's darkness over a band seven pixels wide
        let band = skin + (dark - skin) / 7.0;
        scene.push(crate::Shape::new(
            rectangle(27.0, 5.0, 34.0, 35.0),
            band.as_vec3(),
        ));
        let field = EdgeField::of_target(&ctx.target, w, h);
        let out = into_scene(&ctx, &mut scene, &field, 1.0, crate::plan::keep_mark);
        assert_eq!(out.strokes, 1, "{out:?}");
        assert_eq!(out.dropped + out.refitted, 1, "{out:?}");
        let r = scene.render(crate::Window::full(w, h), 0..scene.shapes.len(), [1.0; 3]);
        for x in [28, 32] {
            let i = 20 * w + x;
            assert!((r[i].as_dvec3() - skin).length() < 0.05, "{x}: {:?}", r[i]);
        }
    }

    /// A line seen through a translucent layer over half its length goes below the layer:
    /// drawn on top it could match only one half's colour (the thighs' contour under the
    /// skirt).
    #[test]
    fn a_line_under_a_layer_goes_below_it() {
        let (w, h) = (330, 40);
        let skin = DVec3::new(0.95, 0.80, 0.70);
        let dark = DVec3::new(0.55, 0.35, 0.30);
        let veil = DVec3::new(0.6, 0.9, 0.6);
        let px: Vec<DVec3> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let under = if x == 30 && (5..35).contains(&y) {
                    dark
                } else {
                    skin
                };
                if y < 20 {
                    0.5 * under + 0.5 * veil
                } else {
                    under
                }
            })
            .collect();
        let ctx = crate::Context::new(&rgb8(&px), w, h, 1.0);
        let mut scene = crate::Scene::with_background(w, h, skin.as_vec3());
        let layer = crate::Shape::new(
            rectangle(0.0, 0.0, w as f32, 20.0),
            crate::Fill {
                color: veil.as_vec3(),
                gradient: None,
                opacity: 0.5,
            },
        );
        scene.push(layer);
        let field = EdgeField::of_target(&ctx.target, w, h);
        let out = into_scene(&ctx, &mut scene, &field, 1.0, crate::plan::keep_mark);
        assert!(out.strokes >= 1, "{out:?}");
        assert!(out.below >= 1, "{out:?}");
        assert!(
            scene.shapes[1].stroke.is_some(),
            "the line is below the layer"
        );
        let r = scene.render(crate::Window::full(w, h), 0..scene.shapes.len(), [1.0; 3]);
        for y in [10, 28] {
            let i = y * w + 30;
            assert!((r[i].as_dvec3() - px[i]).length() < 0.15, "{y}: {:?}", r[i]);
        }
    }
}
