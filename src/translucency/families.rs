//! The proposal's families ([`super::proposal`]) as [`Candidate`]s of the plan: the region as
//! mask, the pinned relation as `k` and color, decided by the plan's objective. They are read
//! after the opaque trace, whose shapes serve as units.

use glam::Vec3;

use super::boundary::{continuity as continuous, open_beyond};
use super::discovery::Shade;
use super::homothety;
use super::inner::{self, InnerInputs};
use super::layer::{EXPLAIN_TOL, explained_pixels};
use super::materials::{MATCH_TOL, min_material_px, outside_palette};
use super::proposal::{
    Family, FixedInputs, Point, REGION_MIN_PX, colour_families, fresh, material_points,
};
use super::rank::{self, Evidence};
use super::raster::{Conn, Image, area_px, dilate, length_px};
use super::region_graph;
use super::{Candidate, OPACITY_MARGIN, SHOW_THROUGH, percentile};
use crate::{Context, Scene};

/// Families the loop grows a region for, the loop's own default.
pub const FAMILIES: usize = 6;

/// The scene's shapes as a label map: per pixel the topmost shape covering it by half or more,
/// each visible part split into four-connected pieces numbered from 1 in raster order. A unit
/// is a place; a material over two places is joined again by its colors in the palette.
pub fn shape_labels(scene: &Scene, width: usize, height: usize) -> Vec<u32> {
    let top: Vec<u32> = super::cost::top_shapes(scene, scene.shapes.len(), width, height)
        .into_iter()
        .map(|s| s.unwrap_or(0))
        .collect();
    let mut out = vec![0u32; width * height];
    let mut next = 0u32;
    let mut stack = Vec::new();
    for start in 0..width * height {
        if out[start] != 0 {
            continue;
        }
        next += 1;
        out[start] = next;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % width, i / width);
            let mut visit = |j: usize| {
                if out[j] == 0 && top[j] == top[i] {
                    out[j] = next;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < width {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - width);
            }
            if y + 1 < height {
                visit(i + width);
            }
        }
    }
    out
}

/// A family's region at a relation `k`, `c` as a candidate of the plan, or none where the
/// opacity is out of range or the region shows no background through the layer.
fn candidate_of(
    ctx: &Context,
    k: f32,
    c: Vec3,
    region: &[bool],
    index: usize,
) -> Option<Candidate> {
    if !(OPACITY_MARGIN..=1. - OPACITY_MARGIN).contains(&k) {
        return None;
    }
    // The show-through rule: the spread of the region's pixels (5th to 95th percentile, the
    // widest channel), k times the backgrounds' spread, must pass the noise. A layer over one
    // flat color cannot be told from an opaque shape and is drawn opaque
    let spread = (0..3)
        .map(|ch| {
            let mut v: Vec<f32> = ctx
                .target
                .iter()
                .zip(region)
                .filter(|(_, m)| **m)
                .map(|(c, _)| c[ch])
                .collect();
            percentile(&mut v, 95.) - percentile(&mut v, 5.)
        })
        .fold(f32::NEG_INFINITY, f32::max);
    if spread <= SHOW_THROUGH {
        if crate::diagnostics::debug() {
            eprintln!("families: family {index} shows no background ({spread:.4})");
        }
        return None;
    }
    let mut m = (1. - k) * c;
    // A translucent black or white is a shading layer, as the discovery's candidates are
    let shade = Shade::of(c, k);
    if let Some(kind) = shade {
        m = (1. - k) * kind.color();
    }
    let bounds = super::mask_bounds(region, ctx.width);
    let m = m.clamp(Vec3::ZERO, Vec3::ONE);
    Some(Candidate::new(k, m, region.to_vec(), bounds, shade))
}

/// A family as a candidate, with the ranking's evidence on it.
#[derive(Clone)]
pub struct Ranked {
    pub candidate: Candidate,
    /// the loop's family (1-based, as the loop numbers them)
    pub index: usize,
    pub evidence: Evidence,
    /// the materials meeting among those that continue out of the region, see
    /// [`rank::contact_continued`]
    pub meeting_beneath: usize,
    /// how far the pin moves its main material's image off the head's relation, see
    /// [`line_apart`]
    pub apart: f64,
}

/// The proposal's families as candidates in [`rank::order`]: each region cut, placed by the
/// beneath's continuity, pinned where the coarse scan finds `least_meeting` materials meeting
/// and its best fit on the head's relation's line ([`line_apart`]), and held to both again.
pub async fn ranked_candidates(
    ctx: &Context,
    scene: &Scene,
    leaves: &[u32],
    least_meeting: usize,
) -> Vec<Ranked> {
    let debug = crate::diagnostics::debug() || crate::diagnostics::levels();
    let (width, height) = (ctx.width, ctx.height);
    let img = Image::from_target(&ctx.target, width);
    let clock = crate::timing::Clock::now();
    let rim = super::materials::rim_of(&img);
    crate::progress::poll().await;
    let fixed = FixedInputs::new(&img, rim).await;
    if debug {
        eprintln!(
            "families: fixed inputs in {:.2}s: proposal {} px, {} components, {} one-side \
             chains, {} links, {} witnesses, {} passing",
            clock.elapsed().as_secs_f64(),
            fixed.proposal.iter().filter(|v| **v).count(),
            fixed.components.len(),
            fixed.one_side.chains.len(),
            fixed.one_side.links.len(),
            fixed.one_side.witnesses.len(),
            fixed.one_side.passing.len()
        );
    }
    // the picture's objects (the leaves joined where they are one material or no contour
    // parts them), which the veil pairs are read on
    crate::progress::poll().await;
    let pieces = super::veil::pieces(&img, leaves, &fixed.rim);
    // The proposal's relation points and families
    crate::progress::poll().await;
    let (mut points, proposal_outside) = material_points(&img, &fixed);
    // The layers the contours propose ([`super::veil`]) head families as fitted relations do:
    // a contour continued at k times its contrast is the gate's evidence, read along it.
    // Their outside is the picture off their own objects
    crate::progress::poll().await;
    let veil = super::veil::proposals(&img, &pieces);
    for p in &veil {
        if debug {
            eprintln!(
                "families: veil proposal k {:.3} c {:?}, {} px, {} pairs",
                p.k,
                (p.c() * 255.).round().to_array(),
                p.px,
                p.votes
            );
        }
        points.push(Point {
            k: p.k,
            c: p.c(),
            score: 2.0,
            src: VEIL_SRC.to_string(),
        });
    }
    let families = colour_families(&points);
    if debug {
        eprintln!(
            "families: fixed inputs and points in {:.2}s",
            clock.elapsed().as_secs_f64()
        );
    }
    // The rest reads on the families' relations: with none, no layer is proposed
    if families.is_empty() {
        return Vec::new();
    }
    // the picture's materials off the proposal, the relations' outside for the regions built
    // from the fitted relations, and off the veil's objects for the veil's
    let outside_all = proposal_outside
        .unwrap_or_else(|| outside_palette(&img, &fixed.proposal, &fixed.rim, None));
    let veil_union: Vec<bool> = (0..width * height)
        .map(|i| veil.iter().any(|p| p.mask[i]))
        .collect();
    let outside_veil = outside_palette(&img, &veil_union, &fixed.rim, None);
    // the opaque scene's shapes with the blends and slivers dissolved ([`super::mixed`]),
    // and the thin objects over the beneath on them ([`super::above`]), for the holes of a
    // pinned family: shapes, not leaves, which blur breaks into thin slivers
    let shapes_thin = std::sync::OnceLock::new();
    let shapes_thin = || {
        shapes_thin.get_or_init(|| {
            let labels = shape_labels(scene, width, height);
            let (shapes, _mixed) = super::mixed::clean_labels(&labels, &img, width, height);
            let thin = super::above::thin_objects(&img, &shapes);
            (shapes, thin)
        })
    };
    let reach = rank::reach(width);
    let top: Vec<(usize, &Family)> = families.iter().take(FAMILIES).enumerate().collect();
    // Each family's region grown: the minimum cut of the segmentation's leaves under the
    // head's relation, with its size and the time it took
    let mut regions: Vec<Option<(Vec<bool>, usize, f64)>> = Vec::with_capacity(top.len());
    for &(_, f) in &top {
        crate::progress::poll().await;
        let p0 = &f.members[0];
        let clock = crate::timing::Clock::now();
        let region = region_graph::region(
            &img,
            outside_of(p0, &outside_all, &outside_veil),
            p0.k,
            p0.c,
            &[],
            leaves,
            min_material_px(&img),
        );
        let px = region.iter().filter(|v| **v).count();
        let t = clock.elapsed().as_secs_f64();
        regions.push(((px as f64) >= area_px(width, REGION_MIN_PX as f64)).then_some((region, px, t)));
    }
    // A region an earlier family already has would pin to the same fit and be dropped as a
    // duplicate, so it is not pinned at all
    let pinned: Vec<bool> = (0..regions.len())
        .map(|a| {
            regions[a].as_ref().is_some_and(|(r, ..)| {
                !regions[..a]
                    .iter()
                    .any(|o| o.as_ref().is_some_and(|(o, ..)| o == r))
            })
        })
        .collect();
    // Each family pinned in turn, the plan's gates read on the coarse grid before the pin
    let coarse_grid = inner::coarse_grid();
    let pin_family = async |i: usize, f: &Family, grown: &(Vec<bool>, usize, f64)| {
        let (img, fixed) = (&img, &fixed);
        let p0 = &f.members[0];
        let (region, grown_px, t_grow) = (&grown.0, grown.1, grown.2);
        let clock = crate::timing::Clock::now();
        let inp = InnerInputs::new(img, region, &fixed.rim, None);
        // A family is scanned only where its region holds the image of an outside material
        // under a relation the pin could take ([`homothety::gate`] on the inside palette
        // against the outside), since the coarse scan is costly
        let gate = homothety::gate(
            &inp.inside_all,
            &inp.outside_palette,
            Some(&homothety::Ring::of(img, region, &fixed.rim)),
        );
        let t_gate = clock.elapsed().as_secs_f64();
        // A veil head's evidence is the pairs that proposed it, which the region's palette
        // cannot repeat: the region passes where it holds at least half of its seeds. Each
        // proposal of a member's relation is a seed once, as several groups share one
        let seeds: Vec<&super::veil::Proposal> = veil
            .iter()
            .filter(|v| {
                f.members
                    .iter()
                    .any(|p| p.src == VEIL_SRC && v.k == p.k && v.c() == p.c)
            })
            .collect();
        let seeds_kept = (!seeds.is_empty()).then(|| {
            let total: usize = seeds.iter().map(|v| v.px).sum();
            let kept = region
                .iter()
                .enumerate()
                .filter(|(j, r)| **r && seeds.iter().any(|v| v.mask[*j]))
                .count();
            (kept, total)
        });
        let passes = match seeds_kept {
            Some((kept, total)) => 2 * kept >= total,
            None => gate.passes(),
        };
        if debug {
            eprintln!(
                "families: family {} gate {} in {t_gate:.2}s ({} px): {}{}",
                i + 1,
                if passes { "passes" } else { "rejects" },
                region.iter().filter(|v| **v).count(),
                gate.describe(),
                seeds_kept.map_or(String::new(), |(kept, total)| format!(
                    "; veil seeds {kept} of {total} px in the region"
                ))
            );
        }
        if !passes {
            return None;
        }
        // The plan's gates read first on the coarse grid: the materials meeting at any of its
        // relations, as the re-cut region can meet more than this one; the line at its best,
        // the last of equal scores as the pin takes it, since the pin keeps the relation's line
        let t_inputs = clock.elapsed().as_secs_f64();
        let rows = inner::scan(&inp, &coarse_grid).await;
        crate::progress::poll().await;
        let t_scan = clock.elapsed().as_secs_f64() - t_inputs;
        let best = rows
            .iter()
            .filter_map(|r| r.fit.as_ref())
            .fold(None::<&inner::InnerFit>, |b, f| {
                if b.is_none_or(|b| f.score >= b.score) {
                    Some(f)
                } else {
                    b
                }
            });
        let meeting = rows
            .iter()
            .filter_map(|r| r.fit.as_ref())
            .map(|f| rank::contact(&inp, f, reach).meeting)
            .max()
            .unwrap_or(0);
        let apart = best.map_or(f64::INFINITY, |f| line_apart(&inp, f, p0.k, p0.c));
        let on_line = apart <= f64::from(SHOW_THROUGH);
        let gated = meeting < least_meeting || !on_line;
        if debug {
            let by_k: Vec<String> = rows
                .iter()
                .filter_map(|r| r.fit.as_ref())
                .map(|f| format!("{:.2}:{:.0}", f.k, line_apart(&inp, f, p0.k, p0.c) * 255.))
                .collect();
            eprintln!(
                "families: family {} grown at k {:.3} c {:?} ({} px, {} colours; inputs \
                 {t_inputs:.2}s, coarse scan {t_scan:.2}s) on the coarse grid: {meeting} \
                 materials meeting, the best {:.1} levels off the growth relation; by k:levels \
                 {}{}",
                i + 1,
                p0.k,
                (p0.c * 255.).round().to_array(),
                region.iter().filter(|v| **v).count(),
                inp.colors.len(),
                apart * 255.,
                by_k.join(" "),
                if gated { ", not pinned" } else { "" }
            );
        }
        if gated {
            return None;
        }
        let fit = inner::pin_with_rows(&inp, rows).await;
        crate::progress::poll().await;
        // a region re-cut at the pin's relation is pinned again near its k
        let pin = async |region: &[bool], k: f64| {
            let inp = InnerInputs::new(img, region, &fixed.rim, None);
            let fit = inner::repin(&inp, k).await;
            (inp, fit)
        };
        let (mut inp, mut fit) = (inp, fit);
        let (placed, recut): (Vec<bool>, Vec<bool>);
        let mut region = region;
        // A fitted family's region is cut at its head's relation; where the pin moves a
        // material's image past the match tolerance, the cut had no evidence there, so it is
        // cut and pinned again at the pin's relation
        if let Some(f) = &fit
            && inp.bmean.iter().any(|b| {
                ((f.k * *b + f.m) - (p0.k * *b + (1. - p0.k) * p0.c))
                    .abs()
                    .max_element()
                    > MATCH_TOL
            })
        {
            recut = region_graph::region(
                img,
                outside_of(p0, &outside_all, &outside_veil),
                f.k,
                f.c(),
                &[],
                leaves,
                min_material_px(img),
            );
            if recut != *region
                && recut.iter().filter(|v| **v).count() as f64
                    >= area_px(width, REGION_MIN_PX as f64)
            {
                region = &recut;
                crate::progress::poll().await;
                (inp, fit) = pin(region, f.k).await;
                crate::progress::poll().await;
            }
        }
        // The region placed by the beneath's continuity past what the full fit reads above
        if let Some(f) = &fit {
            let full = inner::fit_full(&inp, f.k, f.m, 2);
            crate::progress::poll().await;
            let above: Vec<bool> = full
                .px
                .iter()
                .map(|p| {
                    matches!(
                        p.class,
                        inner::Class::White | inner::Class::Black | inner::Class::Tint(_)
                    )
                })
                .collect();
            let open = open_beyond(img, region, super::boundary::reach(img));
            placed = continuous(img, region, f.k, f.c(), &above, &open);
            crate::progress::poll().await;
            region = &placed;
            (inp, fit) = pin(region, f.k).await;
        }
        crate::progress::poll().await;
        let fit = fit.as_ref()?;
        let t_pin = clock.elapsed().as_secs_f64();
        let px = region.iter().filter(|v| **v).count();
        let contact = rank::contact(&inp, fit, reach);
        let meeting_beneath = rank::contact_continued(&inp, fit, reach).meeting;
        let apart = line_apart(&inp, fit, p0.k, p0.c);
        let c = fit.c().as_vec3();
        // The plan's gates at the pin: `least_meeting` materials meeting and the pin on the
        // head's relation's line; most false families meet none
        if contact.meeting < least_meeting || apart > f64::from(SHOW_THROUGH) {
            if debug {
                eprintln!(
                    "families: refined family {} k {:.3} c {:?}, {px} px (grown {grown_px}), {} \
                     materials meeting ({meeting_beneath} beneath), main image {:.1} levels off \
                     the growth relation, gated (pinned {:.2}s)",
                    i + 1,
                    fit.k,
                    (c * 255.).round().to_array(),
                    contact.meeting,
                    apart * 255.,
                    clock.elapsed().as_secs_f64()
                );
            }
            return None;
        }
        crate::progress::poll().await;
        let explained = explained_pixels(img, fit.k, fit.c(), None, EXPLAIN_TOL);
        let evidence = Evidence {
            score_px: fit.score * px as f64,
            fresh: fresh(img, region, &explained, &fixed.rim),
            contact,
        };
        if debug {
            eprintln!(
                "families: refined family {} k {:.3} c {:?}, {px} px (grown {grown_px}), {} \
                 materials meeting ({meeting_beneath} beneath), main image {:.1} levels off the \
                 growth relation, score x px {:.1}, fresh {:.2} (grown in {t_grow:.2}s, pinned \
                 {t_pin:.2}s)",
                i + 1,
                fit.k,
                (c * 255.).round().to_array(),
                evidence.contact.meeting,
                apart * 255.,
                evidence.score_px,
                evidence.fresh,
            );
            eprintln!(
                "families: refined family {} pin: materials {:?} (open {}), shares \
                 alone/white/black/tints/none {:?}, {} tints",
                i + 1,
                fit.materials
                    .iter()
                    .map(|(m, alone, through)| (
                        *m,
                        (inp.bmean[*m] * 255.).round().as_ivec3().to_array(),
                        *alone as usize,
                        *through as usize
                    ))
                    .collect::<Vec<_>>(),
                fit.open.iter().filter(|o| **o).count(),
                fit.shares.map(|s| (s * 100.).round() as i32),
                fit.tints.len()
            );
            eprintln!(
                "families: refined family {} outside materials (mean, px, open): {:?}",
                i + 1,
                inp.bmean
                    .iter()
                    .zip(&inp.bpx)
                    .zip(&fit.open)
                    .map(|((mean, px), open)| (
                        (*mean * 255.).round().as_ivec3().to_array(),
                        *px as usize,
                        *open
                    ))
                    .collect::<Vec<_>>()
            );
        }
        let mut candidate = candidate_of(ctx, fit.k as f32, c, region, i + 1)?;
        crate::progress::poll().await;
        let (shapes, thin_above) = shapes_thin();
        let holes = Holes {
            img,
            index: i + 1,
            shapes,
            thin_above,
            seeds: &seeds,
            debug,
        };
        holes
            .carry(&mut candidate, &inp, fit, &explained.mask)
            .await;
        Some(Ranked {
            candidate,
            index: i + 1,
            evidence,
            meeting_beneath,
            apart,
        })
    };
    let mut grown: Vec<Ranked> = Vec::new();
    for ((&(i, f), region), _) in top.iter().zip(&regions).zip(&pinned).filter(|(_, p)| **p) {
        crate::progress::poll().await;
        if let Some(region) = region {
            grown.extend(pin_family(i, f, region).await);
        }
    }
    let ev: Vec<Evidence> = grown.iter().map(|r| r.evidence).collect();
    // Families pinned to one relation are one layer, decided once: the first by rank
    let mut out: Vec<Ranked> = Vec::new();
    for i in rank::order(&ev) {
        if out
            .iter()
            .all(|o| !same_relation(&o.candidate, &grown[i].candidate))
        {
            out.push(grown[i].clone());
        } else if debug {
            eprintln!(
                "families: family {} pinned to the relation of one ranked before it",
                grown[i].index
            );
        }
    }
    out
}

/// Whether two candidates are one relation: k at one step of the pin's grid, the colour
/// within the matcher's tolerance.
pub fn same_relation(a: &Candidate, b: &Candidate) -> bool {
    let step = (inner::pin_grid()[1] - inner::pin_grid()[0]) as f32;
    (a.k - b.k).abs() < step / 2.
        && (a.fill().color - b.fill().color).abs().max_element() <= MATCH_TOL as f32
}

/// The source name of the relation points the veil pairs propose ([`super::veil`]).
const VEIL_SRC: &str = "veil";

/// The outside palette a fitted head's region is cut against: the picture off the veil's
/// proposed objects for a veil head, off the loop's proposal for the gate's.
fn outside_of<'a>(
    head: &Point,
    outside_all: &'a super::materials::Palette,
    outside_veil: &'a super::materials::Palette,
) -> &'a super::materials::Palette {
    if head.src == VEIL_SRC {
        outside_veil
    } else {
        outside_all
    }
}

/// How far `fit` moves its main material's image (widest channel) from where the growth
/// relation `k`, `c` maps it: the pin should choose along that relation's line of solutions,
/// a move past the show-through is a different reading. Infinite where it counts no material.
pub fn line_apart(
    inp: &InnerInputs,
    fit: &inner::InnerFit,
    k: f64,
    c: super::raster::DVec3,
) -> f64 {
    let Some(&(main, _, _)) = fit.materials.iter().max_by(|a, b| a.1.total_cmp(&b.1)) else {
        return f64::INFINITY;
    };
    let b = inp.bmean[main];
    let grown = k * b + (1.0 - k) * c;
    let pinned = fit.k * b + fit.m;
    (grown - pinned).abs().max_element()
}

/// What the full fit reads above the layer, per pixel of the frame.
fn above_of(full: &inner::InnerFit) -> super::Above {
    super::Above {
        white: full.white.iter().map(|t| *t as f32).collect(),
        tint: full
            .px
            .iter()
            .map(|p| match p.class {
                inner::Class::Tint(ti) => Some(ti as u8),
                _ => None,
            })
            .collect(),
        tints: full
            .tints
            .iter()
            .map(|t| (t.k as f32, t.m.as_vec3()))
            .collect(),
    }
}

/// What a pinned family's holes are read on: the scene's shapes with the blends dissolved,
/// the thin objects over them, and the veil proposals of its head.
struct Holes<'a> {
    img: &'a Image,
    /// the family, 1-based
    index: usize,
    shapes: &'a [u32],
    thin_above: &'a [bool],
    seeds: &'a [&'a super::veil::Proposal],
    debug: bool,
}

impl Holes<'_> {
    /// Sets what the full fit at `fit` reads above the layer and the holes on the candidate,
    /// and carries its region (`inp`'s) under the parts above ([`super::extent`]).
    async fn carry(
        &self,
        candidate: &mut Candidate,
        inp: &InnerInputs,
        fit: &inner::InnerFit,
        explained: &[bool],
    ) {
        let (width, height) = (self.img.width, self.img.height);
        // what the full fit reads above the layer travels with the candidate: the peel takes
        // it off and the vocabulary draws it
        let full = inner::fit_full(inp, fit.k, fit.m, 2);
        candidate.above = Some(std::sync::Arc::new(above_of(&full)));
        let region = &inp.region[..];
        let (shapes, thin_above) = (self.shapes, self.thin_above);
        let n = width * height;
        // The holes: what the full fit leaves unexplained in the region, and the thin objects
        // over the frame off the layer's rim band (the anti-aliasing scale each side of its
        // edge), whose thin pieces are the layer's edge
        let rim = {
            let reach = length_px(width, 1.5);
            let outside: Vec<bool> = region.iter().map(|m| !m).collect();
            let grown = dilate(region, width, height, Conn::Eight, reach);
            let shrunk = dilate(&outside, width, height, Conn::Eight, reach);
            grown.iter().zip(&shrunk).map(|(a, b)| *a && *b).collect::<Vec<bool>>()
        };
        // And for a veil family the objects its pairs' contours run under, in front of both
        // sides whole, even where not thin; and the objects its pairs read bare
        let fronts: Vec<bool> = (0..n)
            .map(|j| self.seeds.iter().any(|v| v.fronts[j]))
            .collect();
        let bare: Vec<bool> = (0..n)
            .map(|j| self.seeds.iter().any(|v| v.bare[j]))
            .collect();
        let n_objects = shapes.iter().max().map_or(0, |&l| l as usize + 1);
        // per object, whether most of its thin pixels where `is` holds are in `of`
        let majority = |is: &dyn Fn(usize) -> bool, of: &[bool]| -> Vec<bool> {
            let mut px = vec![0usize; n_objects];
            let mut ex = vec![0usize; n_objects];
            for i in 0..n {
                if thin_above[i] && is(i) {
                    px[shapes[i] as usize] += 1;
                    if of[i] {
                        ex[shapes[i] as usize] += 1;
                    }
                }
            }
            (0..n_objects).map(|l| 2 * ex[l] > px[l]).collect()
        };
        // A thin object the relation explains mostly is of the layer, not above it, and joins
        // the region
        let of_layer: Vec<bool> = {
            let yes = majority(&|i| !region[i], explained);
            (0..n)
                .map(|i| thin_above[i] && !region[i] && yes[shapes[i] as usize])
                .collect()
        };
        // A thin object the veil pairs read bare continues under the layer, so it lies beneath
        // and is no hole; one no pair reads stays above
        let seen_through: Vec<bool> = {
            let yes = majority(&|_| true, &bare);
            (0..n)
                .map(|i| thin_above[i] && yes[shapes[i] as usize])
                .collect()
        };
        if self.debug {
            let count = |m: &[bool]| m.iter().filter(|v| **v).count();
            eprintln!(
                "families: thin objects {} px, of the layer {} px, seen through {} px (read bare \
                 by the pairs {} px)",
                count(thin_above),
                count(&of_layer),
                count(&seen_through),
                count(&bare)
            );
        }
        let mut holes: Vec<bool> = (0..n)
            .map(|i| {
                (inp.inside[i] && full.px[i].class == inner::Class::None)
                    || (thin_above[i] && !rim[i] && !of_layer[i] && !seen_through[i])
                    || fronts[i]
            })
            .collect();
        let region: &[bool] = &region
            .iter()
            .zip(&of_layer)
            .map(|(r, o)| *r || *o)
            .collect::<Vec<bool>>();
        // The layer carried under the parts above: the holes and the grain objects the region
        // touches ([`super::grain`]). A counted material seen bare is no part
        let grain = super::grain::grain_objects(shapes, region, width, height);
        let opaque: Vec<bool> = {
            let near = dilate(region, width, height, Conn::Eight, length_px(width, 1.5));
            let mut touch = crate::HashSet::<u32>::default();
            for i in 0..n {
                if near[i] && !region[i] && grain.objects[i] != u32::MAX {
                    touch.insert(grain.objects[i]);
                }
            }
            (0..n)
                .map(|i| !region[i] && touch.contains(&grain.objects[i]))
                .collect()
        };
        // with the thin objects' pixels in the rim band the holes leave out: the carry needs
        // its parts up against the region
        let parts: Vec<bool> = (0..n)
            .map(|i| holes[i] || opaque[i] || (thin_above[i] && !region[i]))
            .collect();
        crate::progress::poll().await;
        let (carried, report) = super::extent::carry_under(
            region,
            &parts,
            Some((&grain.lines, &grain.objects)),
            width,
            height,
        );
        #[cfg(feature = "diagnostics")]
        if self.debug
            && let Some(dir) = crate::diagnostics::mask_dir()
        {
            let i = self.index;
            // `carry-N.pgm`: the region 255, what it is carried under 180, the joining curves
            // 120, the parts above 60
            let carry = (0..n).map(|j| {
                if region[j] {
                    255
                } else if carried[j] {
                    180
                } else if report.barrier.get(j).copied().unwrap_or(false) {
                    120
                } else if parts[j] {
                    60
                } else {
                    0
                }
            });
            write_pgm(&dir.join(format!("carry-{i}.pgm")), width, height, carry);
            let counted: Vec<bool> = (0..inp.bmean.len())
                .map(|m| fit.materials.iter().any(|(c, ..)| *c == m))
                .collect();
            // `reading-N.pgm`: the region 255, explained by the relation 180, a counted
            // material seen bare 120, a shape read as opaque over the layer 60
            let reading = (0..n).map(|j| {
                let bare = || {
                    inp.btree
                        .nearest(self.img.px[j], MATCH_TOL)
                        .is_some_and(|(_, b)| counted[inp.bmat[b]])
                };
                if region[j] {
                    255
                } else if opaque[j] {
                    60
                } else if explained[j] {
                    180
                } else if bare() {
                    120
                } else {
                    0
                }
            });
            write_pgm(&dir.join(format!("reading-{i}.pgm")), width, height, reading);
            // the shapes, the grain's lines and objects (`<name>-N.u32`: width, height, a
            // label per pixel, LE)
            for (name, map) in [
                ("shapes", shapes),
                ("lines", &grain.lines[..]),
                ("objects", &grain.objects[..]),
            ] {
                let mut raw: Vec<u8> = Vec::with_capacity(4 * (n + 2));
                for v in [width as u32, height as u32].iter().chain(map) {
                    raw.extend_from_slice(&v.to_le_bytes());
                }
                let _ = std::fs::write(dir.join(format!("{name}-{i}.u32")), raw);
            }
            let inside = holes.iter().map(|&b| if b { 255 } else { 0 });
            write_pgm(&dir.join(format!("holesin-{i}.pgm")), width, height, inside);
            let _ = std::fs::write(
                dir.join(format!("relation-{i}.txt")),
                format!("{} {} {} {}\n", fit.k, fit.m.x, fit.m.y, fit.m.z),
            );
        }
        if report.added_px > 0 || of_layer.iter().any(|v| *v) {
            // an opaque shape the layer is carried under lies above it: a hole
            let mut under = vec![false; n_objects];
            for i in 0..n {
                if opaque[i] && carried[i] {
                    under[shapes[i] as usize] = true;
                }
            }
            for i in 0..n {
                if opaque[i] && under[shapes[i] as usize] {
                    holes[i] = true;
                }
            }
            (candidate.min, candidate.max) = super::mask_bounds(&carried, width);
            candidate.mask = carried;
        }
        if self.debug {
            eprintln!(
                "families: family {} carried under the parts above: {} opaque px, {} ends, {} \
                 pairs, {} rays, {} pieces, {} px added",
                self.index,
                opaque.iter().filter(|v| **v).count(),
                report.ends,
                report.pairs,
                report.rays,
                report.pieces,
                report.added_px
            );
        }
        if holes.iter().any(|h| *h) {
            candidate.holes = Some(std::sync::Arc::new(holes));
        }
    }
}

/// Writes an 8-bit grey PGM of the frame.
#[cfg(feature = "diagnostics")]
fn write_pgm(path: &std::path::Path, width: usize, height: usize, px: impl Iterator<Item = u8>) {
    let mut pgm = format!("P5\n{width} {height}\n255\n").into_bytes();
    pgm.extend(px);
    let _ = std::fs::write(path, pgm);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fill, Shape};
    use glam::Vec2;

    /// A bar drawn across the background splits it into two units, the bar a third.
    #[test]
    fn shapes_split_into_visible_pieces() {
        let (w, h) = (20, 10);
        let mut scene = Scene::with_background(w, h, Fill::solid(Vec3::ONE));
        let p = [
            Vec2::new(8., -1.),
            Vec2::new(12., -1.),
            Vec2::new(12., 11.),
            Vec2::new(8., 11.),
        ];
        let square: Vec<crate::tracer::Cubic> = (0..4)
            .map(|i| {
                let (a, b) = (p[i], p[(i + 1) % 4]);
                [a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.), b]
            })
            .collect();
        scene
            .shapes
            .push(Shape::new(vec![square], Fill::solid(Vec3::ZERO)));
        let labels = shape_labels(&scene, w, h);
        let (left, bar, right) = (labels[0], labels[10], labels[19]);
        assert!(left != bar && bar != right && left != right);
        assert_eq!(labels.iter().max(), Some(&3));
        assert!((0..h).all(|y| labels[y * w + 7] == left && labels[y * w + 12] == right));
    }
}
