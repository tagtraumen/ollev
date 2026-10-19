//! Even-odd loops (of fills and clips) for a host that fills by the nonzero rule only: the
//! loops as they turn where that covers alike, else each turned to have the filled side on its
//! left, so every hole cancels the loop round it. Loops that cross have no such winding: they
//! are turned so, or their even-odd coverage is traced into loops that do not cross.

use crate::render::FillRule;
use glam::DVec2;

use super::Lowering;
use crate::model::{Clip, Curve, Geometry, Object};

pub(super) async fn lower(l: &mut Lowering<'_>, o: &Object) -> Option<Vec<Object>> {
    let Geometry::Fill {
        loops,
        rule: FillRule::EvenOdd,
    } = &o.geometry
    else {
        return None;
    };
    l.report.wound += 1;
    Some(vec![Object {
        geometry: Geometry::Fill {
            loops: if l.quick {
                wind(loops)
            } else {
                nonzero(l, loops).await
            },
            rule: FillRule::NonZero,
        },
        ..o.clone()
    }])
}

/// A clip's loops for the nonzero rule.
pub(super) async fn clip(l: &Lowering<'_>, c: &mut Clip) {
    if c.rule == FillRule::EvenOdd {
        c.loops = if l.quick {
            wind(&c.loops)
        } else {
            nonzero(l, &c.loops).await
        };
        c.rule = FillRule::NonZero;
    }
}

/// Loops whose nonzero coverage is the even-odd coverage of `loops`: they themselves as they
/// turn, else turned to the filled side; for loops that cross, the turned ones within the
/// price, else the coverage traced afresh where that misses less, each drawn in black; a
/// checkpoint between the steps of a trace afresh.
async fn nonzero(l: &Lowering<'_>, loops: &[Vec<Curve>]) -> Vec<Vec<Curve>> {
    let geometry = |loops: &[Vec<Curve>], rule| Geometry::Fill {
        loops: loops.to_vec(),
        rule,
    };
    // The loops' coverage by either rule from one rasterization
    let path = |loops: &[Vec<Curve>]| crate::scene::loops_path(&crate::model::loops32(loops));
    let [even_odd, as_is] =
        crate::scene::coverage_rasters(&path(loops), [FillRule::EvenOdd, FillRule::NonZero]);
    if difference(&even_odd, &as_is) <= 1. / 256. {
        return loops.to_vec();
    }
    let wound = wind(loops);
    let turned = crate::scene::coverage_raster(&path(&wound), FillRule::NonZero);
    if difference(&even_odd, &turned) <= 1. / 256. {
        return wound;
    }
    let fill = |loops: &[Vec<Curve>], rule| Object {
        geometry: geometry(loops, rule),
        paint: crate::model::Paint::Solid(glam::DVec3::ZERO),
        opacity: 1.,
        blur: 0.,
        clip: None,
    };
    let plain = fill(loops, FillRule::EvenOdd);
    let miss =
        |c: &[Vec<Curve>]| l.miss(std::slice::from_ref(&plain), &[fill(c, FillRule::NonZero)]);
    // The turned loops are the coarser form: kept within the price, as every rewrite is
    let kept = miss(&wound);
    if l.price.is_some_and(|p| kept <= p) {
        return wound;
    }
    crate::progress::poll().await;
    let field = super::layers::Field::of(l, &plain, false, false, false);
    crate::progress::poll().await;
    let traced = wind(&field.loops(0.5));
    crate::progress::poll().await;
    if miss(&traced) < kept { traced } else { wound }
}

/// The loops flattened to polygons, finely enough to tell inside from outside.
fn polygon(cubics: &[Curve]) -> Vec<DVec2> {
    const STEPS: usize = 16;
    cubics
        .iter()
        .flat_map(|c| (0..STEPS).map(move |s| super::dashes::eval(c, s as f64 / STEPS as f64)))
        .collect()
}

fn contains(polygon: &[DVec2], p: DVec2) -> bool {
    let mut inside = false;
    for k in 0..polygon.len() {
        let (a, b) = (polygon[k], polygon[(k + 1) % polygon.len()]);
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
            inside = !inside;
        }
    }
    inside
}

/// Each loop turned so the side the even-odd rule fills lies to its left, probed just off the
/// middle of its longest cubic: every loop then winds the filled side the same way, so the
/// nonzero rule fills what the even-odd rule does wherever the loops do not cross.
fn wind(loops: &[Vec<Curve>]) -> Vec<Vec<Curve>> {
    let polygons: Vec<Vec<DVec2>> = loops.iter().map(|l| polygon(l)).collect();
    let filled = |p: DVec2| polygons.iter().filter(|q| contains(q, p)).count() % 2 == 1;
    let reach = |c: &Curve| c[0].distance(c[1]) + c[1].distance(c[2]) + c[2].distance(c[3]);
    loops
        .iter()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let c = l
                .iter()
                .max_by(|a, b| reach(a).total_cmp(&reach(b)))
                .expect("not empty");
            let middle = (c[0] + 3. * c[1] + 3. * c[2] + c[3]) / 8.;
            let tangent = (c[3] + c[2] - c[1] - c[0]).normalize_or(DVec2::X);
            // A hundredth of a pixel to the left
            if filled(middle + tangent.perp() * 0.01) {
                l.clone()
            } else {
                l.iter().rev().map(|c| [c[3], c[2], c[1], c[0]]).collect()
            }
        })
        .collect()
}

/// The largest difference of two coverages over any pixel.
fn difference(ra: &crate::Raster, rb: &crate::Raster) -> f32 {
    let (x0, y0) = (ra.x0.min(rb.x0), ra.y0.min(rb.y0));
    let (x1, y1) = (
        (ra.x0 + ra.w).max(rb.x0 + rb.w),
        (ra.y0 + ra.h).max(rb.y0 + rb.h),
    );
    let width = x1 + 1;
    let mut worst = 0f32;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = y * width + x;
            worst = worst.max((ra.at(i, width) - rb.at(i, width)).abs());
        }
    }
    worst
}
