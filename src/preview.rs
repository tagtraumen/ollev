//! First looks at a picture for the host to show while the trace reads it: the picture
//! downsampled to [`FIRST`] pixels, then four and sixteen times as many, each segmented into
//! flat regions drawn in their mean colours over the picture's mean, the regions' outlines
//! the pixel cracks between them as polygons (the finest level's fitted with cubics).

use glam::{DVec2, DVec3, IVec2};

use crate::model::{Curve, FillRule, Geometry, Item, Model, Object, Paint};
use crate::tracer::planar::{ChainUse, PlanarMap};

/// The pixels of the coarsest first look.
pub const FIRST: usize = 1 << 12;

/// Shows the host each first look at the `width` by `height` picture segmented at `detail`:
/// the levels of [`FIRST`] pixels times a power of four up to a third of the picture's.
pub(crate) async fn preview(rgb: &[u8], width: usize, height: usize, detail: f64) {
    let pixels = width * height;
    let levels: Vec<usize> = std::iter::successors(Some(FIRST), |p| Some(4 * p))
        .take_while(|&p| 3 * p <= pixels)
        .collect();
    let Some(&finest) = levels.last() else {
        return;
    };
    let size = |p: usize| {
        let s = (pixels as f64 / p as f64).sqrt();
        let w = ((width as f64 / s) as usize).max(1);
        let h = ((height as f64 / s) as usize).max(1);
        (w, h, width as f64 / w as f64)
    };
    // The finest level from the picture, the coarser from it
    let (fw, fh, fs) = size(finest);
    let fine = crate::input::downsampled(rgb, width, height, fw, fh, fs);
    for &p in &levels {
        crate::progress::poll().await;
        let (w, h, s) = size(p);
        let small = if p == finest {
            fine.clone()
        } else {
            crate::input::downsampled(&fine, fw, fh, w, h, fw as f64 / w as f64)
        };
        let model = regions(&small, w, h, s, (width, height), detail, p == finest).await;
        crate::progress::preview(model);
    }
}

/// The flat regions of the `w` by `h` picture `small`, drawn at `s` times its size over a
/// `frame`, outlined by fitted cubics where `fit`.
async fn regions(
    small: &[u8],
    w: usize,
    h: usize,
    s: f64,
    frame: (usize, usize),
    detail: f64,
    fit: bool,
) -> Model {
    let params = crate::tracer::Params::with_detail(detail);
    let colors = crate::tracer::segmentation_colors(small, w, h, &params).await;
    let (seeds, count) = crate::tracer::segment::equal_color_regions(small, w, h);
    let labels =
        crate::tracer::segment::segment_seeded(&colors, w, h, &params.segment, &seeds, count)
            .await
            .labels;
    let regions = labels.iter().max().map_or(0, |&l| l as usize + 1);
    let (mut sum, mut size) = (vec![DVec3::ZERO; regions], vec![0usize; regions]);
    for (i, &l) in labels.iter().enumerate() {
        let c = &small[3 * i..][..3];
        sum[l as usize] += DVec3::new(c[0].into(), c[1].into(), c[2].into()) / 255.;
        size[l as usize] += 1;
    }
    let map = crate::tracer::planar::build(&labels, w, h, regions);
    let chains = outlines(&map, s, fit).await;
    let loop_of = |uses: &[ChainUse]| -> Vec<Curve> {
        uses.iter()
            .flat_map(|u| {
                let cubics = &chains[u.chain];
                let oriented: Vec<Curve> = if u.reversed {
                    cubics.iter().rev().map(|c| [c[3], c[2], c[1], c[0]]).collect()
                } else {
                    cubics.clone()
                };
                oriented
            })
            .collect()
    };
    let (width, height) = (frame.0 as f64, frame.1 as f64);
    let corners = [
        DVec2::ZERO,
        DVec2::new(width, 0.),
        DVec2::new(width, height),
        DVec2::new(0., height),
    ];
    let ground = (0..4).map(|k| line(corners[k], corners[(k + 1) % 4])).collect();
    let object = |loops, colour| {
        Item::Object(Box::new(Object {
            // A region lies to the left of each of its loops, so its holes wind against it
            geometry: Geometry::Fill {
                loops,
                rule: FillRule::NonZero,
            },
            paint: Paint::Solid(colour),
            opacity: 1.,
            blur: 0.,
            clip: None,
        }))
    };
    let mean = sum.iter().sum::<DVec3>() / labels.len().max(1) as f64;
    let mut order: Vec<usize> = (0..regions).filter(|&r| size[r] > 0).collect();
    order.sort_by_key(|&r| std::cmp::Reverse(size[r]));
    let mut items = vec![object(vec![ground], mean)];
    for r in order {
        let loops: Vec<Vec<Curve>> = map.loops[r].iter().map(|l| loop_of(l)).collect();
        items.push(object(loops, sum[r] / size[r] as f64));
    }
    Model {
        width: frame.0,
        height: frame.1,
        items,
        clips: Vec::new(),
        definitions: Vec::new(),
        fidelity: None,
        clip: None,
    }
}

/// Each chain of the map at `s` times its size: fitted with cubics where `fit`, else the
/// polygon of its turns.
async fn outlines(map: &PlanarMap, s: f64, fit: bool) -> Vec<Vec<Curve>> {
    let mut chains = Vec::with_capacity(map.chains.len());
    for (k, chain) in map.chains.iter().enumerate() {
        if k % 256 == 0 {
            crate::progress::poll().await;
        }
        let at = |p: IVec2| p.as_dvec2() * s;
        chains.push(if fit {
            let (cubics, _) = crate::tracer::fit::fit_chain_corners(
                &chain.points,
                chain.is_closed(),
                &crate::outline::FIT,
            );
            cubics.iter().map(|c| c.map(|p| p.as_dvec2() * s)).collect()
        } else {
            let p = &chain.points;
            let turns = (1..p.len().saturating_sub(1)).filter(|&i| p[i] - p[i - 1] != p[i + 1] - p[i]);
            let mut corners: Vec<IVec2> = std::iter::once(p[0]).chain(turns.map(|i| p[i])).collect();
            corners.push(p[p.len() - 1]);
            corners.windows(2).map(|w| line(at(w[0]), at(w[1]))).collect()
        });
    }
    chains
}

/// The straight cubic from `a` to `b`.
fn line(a: DVec2, b: DVec2) -> Curve {
    [a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.), b]
}
