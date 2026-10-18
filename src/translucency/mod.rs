//! Translucent shapes: a shape of color `c` at opacity `a` over `b` shows `o = k b + m` with
//! `k = 1 - a`, `m = a c`, one relation for the whole shape, so it peels to `(o - m) / k`.
//! Shading fixes the color: a shadow `o = s b` is black at `1 - s`, a highlight white.

pub mod above;
pub mod beneath;
pub mod boundary;
pub mod completion;
pub mod cost;
pub mod discovery;

pub mod evidence;
pub mod extent;
pub mod families;
pub mod grain;
pub mod homothety;
pub mod inner;
pub mod layer;
pub mod leaves;
pub mod lines;
pub mod materials;
pub mod mixed;
pub mod proposal;
pub mod rank;
pub mod raster;
pub mod region;
pub mod region_graph;
pub mod segment;
pub mod stage;
pub mod veil;
pub mod vocabulary;

use crate::tracer::Cubic;
use glam::{IVec2, Vec3};
use raster::{Conn, dilate};

use crate::{Context, Fill};
use discovery::Shade;

/// A translucent shape peeled off the image: its outline, color and opacity.
#[derive(Clone, Debug)]
pub struct Overlay {
    pub loops: Vec<Vec<Cubic>>,
    pub fill: Fill,
}

/// Most a channel may deviate from the relation for a region to belong to the shape: the
/// quantization of 8 bits plus the anti-aliasing a region's mean carries along its rim.
pub const NOISE: f32 = 2. / 255.;
/// Opacities this close to nothing or to all are not translucency: a shape that barely
/// shows or an opaque one.
pub const OPACITY_MARGIN: f32 = 0.02;
/// Least the backgrounds have to show through a shape, as the difference its two inside
/// colors keep of their backgrounds' difference: each is known to the noise, the difference
/// to twice it.
pub const SHOW_THROUGH: f32 = 4. * NOISE;
/// A plane slope below this per pixel is flat: half a level, the equation graph's.
pub const SLOPE_MIN: f32 = 0.5 / 255.;

/// Whether a color lies in gamut to the noise.
pub fn in_gamut(c: Vec3) -> bool {
    c.min_element() >= -NOISE && c.max_element() <= 1. + NOISE
}

/// The `q`th percentile of the values with linear interpolation between ranks, as numpy's;
/// sorts them. Nothing for no values.
pub fn percentile(values: &mut [f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.;
    }
    values.sort_unstable_by(f32::total_cmp);
    let at = q / 100. * (values.len() - 1) as f32;
    let (lo, hi) = (at.floor() as usize, at.ceil() as usize);
    values[lo] + (values[hi] - values[lo]) * (at - lo as f32)
}

/// The root of `a` in a union-find forest, halving the path on the way.
pub fn find_root(parent: &mut [usize], mut a: usize) -> usize {
    while parent[a] != a {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    a
}

/// The box `min..max` of a frame mask's set pixels, `(MAX, MIN)` where none is set.
pub fn mask_bounds(mask: &[bool], width: usize) -> (IVec2, IVec2) {
    let (mut min, mut max) = (IVec2::MAX, IVec2::MIN);
    for (i, _) in mask.iter().enumerate().filter(|(_, v)| **v) {
        let p = IVec2::new((i % width) as i32, (i / width) as i32);
        min = min.min(p);
        max = max.max(p + 1);
    }
    (min, max)
}

/// The overlay's fill refitted by least squares of `target = k below + m` over the covered
/// pixels `fill` already explains (others are something over the layer); kept where `below`
/// does not vary; with `shade`, only the opacity is fitted.
pub fn refit_fill(
    target: &[Vec3],
    below: &[Vec3],
    coverage: &[f32],
    fill: Fill,
    shade: Option<Shade>,
) -> Fill {
    let (k0, m0) = (1. - fill.opacity, fill.color * fill.opacity);
    let covered: Vec<usize> = (0..coverage.len())
        .filter(|&i| {
            coverage[i] >= 0.99 && (target[i] - (k0 * below[i] + m0)).abs().max_element() <= NOISE
        })
        .collect();
    if covered.len() < 16 {
        return fill;
    }
    let n = covered.len() as f32;
    if let Some(kind) = shade {
        // Through the origin in the layer's own space
        let (mut num, mut den) = (0f32, 0f32);
        for &i in &covered {
            let (t, b) = (kind.space(target[i]), kind.space(below[i]));
            num += t.dot(b);
            den += b.length_squared();
        }
        if den < n * NOISE * NOISE {
            return fill;
        }
        let k = (num / den).clamp(OPACITY_MARGIN, 1. - OPACITY_MARGIN);
        return Fill {
            color: kind.color(),
            gradient: None,
            opacity: 1. - k,
        };
    }
    let mean_t = covered.iter().map(|&i| target[i]).sum::<Vec3>() / n;
    let mean_b = covered.iter().map(|&i| below[i]).sum::<Vec3>() / n;
    let (mut num, mut den) = (0f32, 0f32);
    for &i in &covered {
        let (dt, db) = (target[i] - mean_t, below[i] - mean_b);
        num += dt.dot(db);
        den += db.length_squared();
    }
    if den < n * NOISE * NOISE {
        return fill;
    }
    let k = (num / den).clamp(OPACITY_MARGIN, 1. - OPACITY_MARGIN);
    let m = mean_t - k * mean_b;
    Fill {
        color: (m / (1. - k)).clamp(Vec3::ZERO, Vec3::ONE),
        gradient: None,
        opacity: 1. - k,
    }
}

/// A translucent shape found on an image: its relation `o = k b + m`, its pixels, their box,
/// and whether it is a shading layer.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub k: f32,
    pub m: Vec3,
    /// Per pixel of the frame, whether the shape covers it.
    pub mask: Vec<bool>,
    pub min: IVec2,
    pub max: IVec2,
    pub shade: Option<Shade>,
    /// What the pin's full fit read above the layer within its region, where a family
    /// has been pinned ([`inner::fit_full`]).
    pub above: Option<std::sync::Arc<Above>>,
    /// Per pixel, what lies over the beneath outside the relation: what the pin's full fit
    /// leaves unexplained in the region, and the thin objects [`above`] puts over their sides.
    /// The beneath is completed across them ([`completion`]) and they are traced above it.
    pub holes: Option<std::sync::Arc<Vec<bool>>>,
}

/// What lies above a layer in its region, per pixel, as the pin's full fit reads it: a white
/// (highlights) or a tint (dark folds). Peeled with the layer so the beneath does not keep
/// them as lenses, and drawn above it by the vocabulary.
#[derive(Clone, Debug, Default)]
pub struct Above {
    /// the white's opacity, 0 where there is none
    pub white: Vec<f32>,
    /// the tint over the pixel, an index into `tints`
    pub tint: Vec<Option<u8>>,
    /// the tints as (k, m): the pixel is `k` times what lies beneath plus `m`
    pub tints: Vec<(f32, Vec3)>,
}

impl Above {
    /// The pixel with what lies above the layer taken off: the white first (it is on
    /// top), then the tint.
    pub fn peel(&self, i: usize, p: Vec3) -> Vec3 {
        let mut p = p;
        let t = self.white.get(i).copied().unwrap_or(0.);
        if t > 0. && t < 1. {
            p = ((p - Vec3::splat(t)) / (1. - t)).clamp(Vec3::ZERO, Vec3::ONE);
        }
        if let Some(ti) = self.tint.get(i).copied().flatten() {
            let (k, m) = self.tints[ti as usize];
            if k > 0. {
                p = ((p - m) / k).clamp(Vec3::ZERO, Vec3::ONE);
            }
        }
        p
    }
}

impl Candidate {
    /// A shape of relation `o = k b + m` over `mask`, its pixels within `bounds`.
    pub fn new(
        k: f32,
        m: Vec3,
        mask: Vec<bool>,
        (min, max): (IVec2, IVec2),
        shade: Option<Shade>,
    ) -> Candidate {
        Candidate {
            k,
            m,
            mask,
            min,
            max,
            shade,
            above: None,
            holes: None,
        }
    }

    /// Adds a stage region's pixels to the shape.
    fn cover(&mut self, r: &stage::Region, width: usize) {
        for p in region_pixels(r) {
            self.mask[p.y as usize * width + p.x as usize] = true;
            self.min = self.min.min(p);
            self.max = self.max.max(p + 1);
        }
    }

    /// What the shape is drawn with: its color at opacity `1 - k`.
    pub fn fill(&self) -> Fill {
        let color = match self.shade {
            Some(kind) => kind.color(),
            None => (self.m / (1. - self.k)).clamp(Vec3::ZERO, Vec3::ONE),
        };
        Fill {
            color,
            gradient: None,
            opacity: 1. - self.k,
        }
    }

    /// The image with the shape peeled off, as 8-bit RGB: where `overlay` covers a pixel by
    /// `a`, `b = (o - a m) / (1 - a (1 - k))`, so the anti-aliased rim peels too and leaves no
    /// line a pixel wide beneath.
    pub fn peeled(&self, ctx: &Context, overlay: &Overlay) -> Vec<u8> {
        let (width, height) = (ctx.width, ctx.height);
        let shape = crate::Shape::new(overlay.loops.clone(), overlay.fill);
        let window = crate::Window::around(shape.min, shape.max, 1., width, height);
        let mut coverage = vec![0f32; width * height];
        for (i, a) in window.pixels(width).zip(shape.coverage(window)) {
            coverage[i] = a;
        }
        // Where a pixel peels out of gamut the outline covers it by less, the most that keeps
        // it in gamut. Per channel the peel moves one way with the coverage (the sign of
        // `(1 - k) o - m`), so the bound is where it leaves the gamut.
        let q = 1. - self.k;
        let mut out = Vec::with_capacity(ctx.target.len() * 3);
        for (i, &p) in ctx.target.iter().enumerate() {
            let mut a = coverage[i];
            // what lies above the layer comes off first, where the shape covers the pixel
            // whole (the rim is the layer's own edge)
            let p = match &self.above {
                Some(above) if a >= 1. => above.peel(i, p),
                _ => p,
            };
            for c in 0..3 {
                let at = |a: f32| (p[c] - a * self.m[c]) / (1. - a * q);
                for bound in [-NOISE, 1. + NOISE] {
                    let beyond = |v: f32| if bound < 0. { v < bound } else { v > bound };
                    if a > 0. && beyond(at(a)) {
                        let den = self.m[c] - bound * q;
                        a = if den.abs() > 1e-6 {
                            ((p[c] - bound) / den).clamp(0., a)
                        } else {
                            0.
                        };
                    }
                }
            }
            let b = if a > 0. {
                ((p - a * self.m) / (1. - a * q)).clamp(Vec3::ZERO, Vec3::ONE)
            } else {
                p
            };
            for v in [b.x, b.y, b.z] {
                out.push((v * 255.).round() as u8);
            }
        }
        out
    }

    /// The shape's outline traced from its pixels, on a frame of the given size.
    pub fn overlay(&self, width: usize, height: usize) -> Option<Overlay> {
        let loops = crate::outline::outline_of(
            self.min,
            self.max,
            width,
            height,
            &crate::outline::FIT,
            |i| self.mask[i],
        );
        (!loops.is_empty()).then(|| Overlay {
            loops,
            fill: self.fill(),
        })
    }

    /// The shape within a window of its frame, as a candidate on the window's frame.
    pub fn crop(&self, window: crate::Window, width: usize) -> Candidate {
        let origin = IVec2::new(window.x as i32, window.y as i32);
        let mask: Vec<bool> = window.pixels(width).map(|i| self.mask[i]).collect();
        let size = IVec2::new(window.width as i32, window.height as i32);
        // what lies above is read on the whole frame; the window's decision peels the layer alone
        let bounds = (
            (self.min - origin).clamp(IVec2::ZERO, size),
            (self.max - origin).clamp(IVec2::ZERO, size),
        );
        Candidate::new(self.k, self.m, mask, bounds, self.shade)
    }
}

/// How many hypotheses of the list the region stage grows.
const STAGE_TOP: usize = 20;
/// How far a shading layer's scales may differ between channels for it to be one opacity of
/// black or white: the shading pass's Hough step. Layers past it would need a per-channel blend.
const UNIFORM_SHADE: f32 = 0.025;

/// The translucent shapes over `ctx`'s image read on its leaves `lv`: each relation's passing
/// regions, pieces merged ([`merge_phase`]), overlaps to the shape on top, ranked uncovered
/// first, then by boundary share times k and continuity, kept where [`evidence::witnessed_at`].
pub async fn candidates(ctx: &Context, lv: &leaves::Leaves) -> Vec<Candidate> {
    let debug = crate::diagnostics::debug();
    let (width, height) = (ctx.width, ctx.height);
    let target = &ctx.target[..];
    let plane = discovery::plane_pass(lv);
    crate::progress::poll().await;
    let graph = discovery::equation_graph(lv);
    crate::progress::poll().await;
    let hyps = discovery::merge_lists(&[plane.clone(), graph.clone()]);
    let layers = discovery::shading_pass(lv);
    if debug {
        let show = |h: &discovery::Hypothesis| {
            format!(
                "k {:.3} c {:?} {:.2}",
                h.k,
                (h.c * 255.).round().to_array(),
                h.score
            )
        };
        eprintln!(
            "translucency: {} leaves ({} thin), {} pairs, {} points",
            lv.n,
            lv.thin.iter().filter(|&&t| t).count(),
            lv.pairs.len(),
            lv.points.len()
        );
        eprintln!(
            "translucency: plane pass {:?}",
            plane.iter().take(5).map(show).collect::<Vec<_>>()
        );
        eprintln!(
            "translucency: equation graph {:?}",
            graph.iter().take(5).map(show).collect::<Vec<_>>()
        );
        eprintln!(
            "translucency: shading {:?}",
            layers
                .iter()
                .take(5)
                .map(|l| format!("{:?} s {:.3?} {:.2}", l.kind, l.s.to_array(), l.score))
                .collect::<Vec<_>>()
        );
    }
    // The relations to grow: the list's top, then the uniform shading layers as black or
    // white relations
    let mut relations: Vec<(f32, Vec3, Option<Shade>)> = hyps
        .iter()
        .take(STAGE_TOP)
        .map(|h| (h.k, (1. - h.k) * h.c, None))
        .collect();
    for layer in layers.iter().take(STAGE_TOP) {
        let k = layer.s.element_sum() / 3.;
        if (layer.s - k).abs().max_element() <= UNIFORM_SHADE {
            relations.push((k, (1. - k) * layer.kind.color(), Some(layer.kind)));
        }
    }
    let mut found: Vec<(f32, Candidate)> = Vec::new();
    // Every region that shows its backgrounds through and has an interior, whatever its
    // share, for the overlaps below
    let mut overlaps: Vec<stage::Region> = Vec::new();
    for (k, m, shade) in relations {
        crate::progress::poll().await;
        let regions = stage::regions_of(lv, target, k, m, shade);
        overlaps.extend(regions.iter().filter(|r| r.show && r.interior).cloned());
        let good: Vec<&stage::Region> = regions.iter().filter(|r| r.kept()).collect();
        if debug {
            eprintln!(
                "translucency: k {k:.3} c {:?}{}: {} regions, kept {:?}",
                (m / (1. - k) * 255.).round().to_array(),
                shade.map_or(String::new(), |s| format!(" {s:?}")),
                regions.len(),
                regions
                    .iter()
                    .map(|r| format!(
                        "{} px k {:.3} {}/{} spread {:.0}{}{}{}",
                        r.area,
                        r.k,
                        r.fitting,
                        r.pairs,
                        r.spread * 255.,
                        if r.share { "" } else { " -share" },
                        if r.show { "" } else { " -show" },
                        if r.interior { "" } else { " -interior" }
                    ))
                    .collect::<Vec<_>>()
            );
        }
        if good.is_empty() {
            continue;
        }
        // The relation of the region with the most own edge
        let best = good
            .iter()
            .fold(good[0], |a, &b| if b.fitting > a.fitting { b } else { a });
        let (k, mut m) = (best.k, best.m);
        if !(OPACITY_MARGIN..=1. - OPACITY_MARGIN).contains(&k) {
            continue;
        }
        // A translucent black or white is a shading layer whichever pass found it
        let shade = shade.or_else(|| Shade::of(m / (1. - k), k));
        if let Some(kind) = shade {
            m = (1. - k) * kind.color();
        }
        let empty = vec![false; width * height];
        let mut candidate = Candidate::new(k, m, empty, (IVec2::MAX, IVec2::MIN), shade);
        for r in &good {
            candidate.cover(r, width);
        }
        let (fitting, pairs) = good
            .iter()
            .fold((0, 0), |(f, p), r| (f + r.fitting, p + r.pairs));
        let share = fitting as f32 / pairs.max(1) as f32;
        // The same shape from two relations is one; a shading layer's reading of it replaces
        // a colored one, the same picture for a segment less
        match found.iter().position(|(_, c)| c.mask == candidate.mask) {
            Some(at) if found[at].1.shade.is_none() && shade.is_some() => {
                found[at] = (share, candidate);
            }
            Some(_) => {}
            None => found.push((share, candidate)),
        }
    }
    // Pieces of one shape that no relation grew whole
    for (share, candidate) in merge_phase(lv, target, &overlaps, debug) {
        if !found.iter().any(|(_, c)| c.mask == candidate.mask) {
            found.push((share, candidate));
        }
    }
    // An overlap of two shapes composes to `o = k1 (k2 b + m2) + m1`: a region of that
    // relation touching the upper candidate joins it. Each relation is fitted to the noise,
    // so the composition is held to twice it.
    let composes = |r: &stage::Region, top: &Candidate, under: &Candidate| {
        (r.k - top.k * under.k).abs() + (r.m - (top.k * under.m + top.m)).abs().max_element()
            <= 4. * NOISE
    };
    // A candidate with another over it, by some overlap, goes after that one
    let covered: Vec<bool> = (0..found.len())
        .map(|u| {
            overlaps.iter().any(|r| {
                (0..found.len()).any(|t| {
                    t != u
                        && composes(r, &found[t].1, &found[u].1)
                        && region_pixels(r).any(|p| touches(&found[t].1.mask, p, width, height))
                })
            })
        })
        .collect();
    let joined: Vec<Vec<usize>> = (0..found.len())
        .map(|t| {
            let top = &found[t].1;
            (0..overlaps.len())
                .filter(|&o| {
                    let r = &overlaps[o];
                    (0..found.len()).any(|u| u != t && composes(r, top, &found[u].1))
                        && region_pixels(r).any(|p| touches(&top.mask, p, width, height))
                })
                .collect()
        })
        .collect();
    for (t, list) in joined.into_iter().enumerate() {
        for o in list {
            let c = &mut found[t].1;
            c.cover(&overlaps[o], width);
            if debug {
                eprintln!(
                    "translucency: the overlap of {} px at k {:.3} joins the candidate at k {:.3}",
                    overlaps[o].area, overlaps[o].k, c.k
                );
            }
        }
    }
    // Ranked, the uncovered first, by the share of their boundary that fits times k, the
    // show-through, times the share of their interior that peels into colors seen around them
    // (see [`continuity`]): a score, never a gate
    let found: Vec<(f32, Candidate)> = {
        let mut ranked: Vec<(bool, f32, (f32, Candidate))> = covered
            .into_iter()
            .zip(found)
            .map(|(cov, (share, c))| {
                let score = share * c.k * continuity(target, &c, width, height);
                if debug {
                    eprintln!(
                        "translucency: rank k {:.3}: share {share:.2}, score {score:.3}{}",
                        c.k,
                        if cov { ", covered" } else { "" }
                    );
                }
                (cov, score, (share, c))
            })
            .collect();
        ranked.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
        ranked.into_iter().map(|(_, _, f)| f).collect()
    };
    if debug {
        for (share, c) in &found {
            eprintln!(
                "translucency: candidate k {:.3} c {:?}{} share {share:.2}, {} px in {:?}..{:?}",
                c.k,
                (c.fill().color * 255.).round().to_array(),
                c.shade.map_or(String::new(), |s| format!(" {s:?}")),
                c.mask.iter().filter(|&&m| m).count(),
                c.min,
                c.max
            );
        }
    }
    found
        .into_iter()
        .map(|(_, c)| c)
        .filter(|c| evidence::witnessed_at(target, width, c.k, c.m, &c.mask))
        .collect()
}

/// The pixels of a region.
fn region_pixels(r: &stage::Region) -> impl Iterator<Item = IVec2> + '_ {
    let bw = (r.bounds.max.x - r.bounds.min.x) as usize;
    r.bits
        .iter()
        .enumerate()
        .filter(|(_, b)| **b)
        .map(move |(j, _)| r.bounds.min + IVec2::new((j % bw) as i32, (j / bw) as i32))
}

/// Whether a pixel is in a mask or next to it.
fn touches(mask: &[bool], p: IVec2, width: usize, height: usize) -> bool {
    [IVec2::ZERO, IVec2::X, -IVec2::X, IVec2::Y, -IVec2::Y]
        .into_iter()
        .map(|d| p + d)
        .any(|q| {
            q.x >= 0
                && q.y >= 0
                && (q.x as usize) < width
                && (q.y as usize) < height
                && mask[q.y as usize * width + q.x as usize]
        })
}

/// A group of pieces of one shape in the merge phase: its relation, its pieces, their
/// boundary pairs.
struct Group {
    k: f32,
    m: Vec3,
    members: Vec<usize>,
    own: Vec<(Vec3, Vec3)>,
}

impl Group {
    /// Its boundary pairs within the noise of a relation.
    fn inliers(&self, k: f32, m: Vec3) -> Vec<(Vec3, Vec3)> {
        self.own
            .iter()
            .filter(|(b, o)| (*o - (k * *b + m)).abs().max_element() <= 3. * NOISE)
            .copied()
            .collect()
    }
}

/// Least share of a piece's inliers a joint relation has to keep: one relation explains each
/// piece's evidence, the tenth an allowance for the pairs that land on an opaque line beside
/// the shape.
const JOINT_SHARE: f32 = 0.9;

/// One relation by least squares over the groups' own inliers, if it keeps nine in ten of
/// each group's inliers and each group's relation explains the others' too: a piece over one
/// background has a relation free along a line, along which the joint fit would drift.
fn joint_fit(groups: &[&Group]) -> Option<(f32, Vec3)> {
    let sets: Vec<Vec<(Vec3, Vec3)>> = groups.iter().map(|g| g.inliers(g.k, g.m)).collect();
    let explains = |k: f32, m: Vec3, set: &[(Vec3, Vec3)]| {
        set.is_empty()
            || set
                .iter()
                .filter(|(b, o)| (*o - (k * *b + m)).abs().max_element() <= 3. * NOISE)
                .count() as f32
                >= JOINT_SHARE * set.len() as f32
    };
    if groups
        .iter()
        .any(|g| sets.iter().any(|set| !explains(g.k, g.m, set)))
    {
        return None;
    }
    let all: Vec<(Vec3, Vec3)> = sets.iter().flatten().copied().collect();
    if all.len() < 8 {
        return None;
    }
    let n = all.len() as f32;
    let mb = all.iter().map(|p| p.0).sum::<Vec3>() / n;
    let mo = all.iter().map(|p| p.1).sum::<Vec3>() / n;
    let den: f32 = all.iter().map(|p| (p.0 - mb).length_squared()).sum();
    if den <= 1e-9 {
        return None;
    }
    let k = all.iter().map(|p| (p.0 - mb).dot(p.1 - mo)).sum::<f32>() / den;
    let m = mo - k * mb;
    if !(0.02..=0.98).contains(&k) || sets.iter().any(|set| !explains(k, m, set)) {
        return None;
    }
    Some((k, m))
}

/// Pieces (stage regions with show-through and interior) merge, nearest first, when one
/// relation explains them ([`joint_fit`]) and the gap is no wider than the wider piece; a
/// group is a candidate where its union passes the share rule.
fn merge_phase(
    lv: &leaves::Leaves,
    target: &[Vec3],
    regions: &[stage::Region],
    debug: bool,
) -> Vec<(f32, Candidate)> {
    let (width, height) = (lv.width, lv.height);
    // One piece per distinct region
    let mut pieces: Vec<&stage::Region> = Vec::new();
    for r in regions {
        if !pieces
            .iter()
            .any(|p| p.bounds == r.bounds && p.bits == r.bits)
        {
            pieces.push(r);
        }
    }
    let n = pieces.len();
    if n < 2 {
        return Vec::new();
    }
    let boxes: Vec<(IVec2, IVec2)> = pieces
        .iter()
        .map(|p| {
            region_pixels(p).fold((IVec2::MAX, IVec2::MIN), |(lo, hi), q| {
                (lo.min(q), hi.max(q + 1))
            })
        })
        .collect();
    let mut diameter: Vec<f32> = pieces.iter().map(|p| 2. * p.depth.sqrt()).collect();
    // The gap between two pieces: the nearest of their pixels, computed in the box of both
    // where their boxes lie near enough for it to matter
    let mut gap = vec![vec![f32::INFINITY; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let ((a0, a1), (b0, b1)) = (boxes[i], boxes[j]);
            let apart = (b0 - a1).max(a0 - b1).max(IVec2::ZERO).as_vec2().length();
            if apart > diameter[i].max(diameter[j]) {
                continue;
            }
            let (lo, hi) = (a0.min(b0), a1.max(b1));
            let bw = (hi.x - lo.x) as usize;
            let bh = (hi.y - lo.y) as usize;
            let mut mine = vec![false; bw * bh];
            for q in region_pixels(pieces[i]) {
                let d = q - lo;
                mine[d.y as usize * bw + d.x as usize] = true;
            }
            let d = stage::distance_squared(bw, bh, |k| mine[k]);
            let g = region_pixels(pieces[j])
                .map(|q| {
                    let e = q - lo;
                    d[e.y as usize * bw + e.x as usize]
                })
                .fold(f32::INFINITY, f32::min)
                .sqrt();
            gap[i][j] = g;
            gap[j][i] = g;
        }
    }
    let mut groups: Vec<Option<Group>> = pieces
        .iter()
        .enumerate()
        .map(|(i, p)| {
            Some(Group {
                k: p.k,
                m: p.m,
                members: vec![i],
                own: p.own.clone(),
            })
        })
        .collect();
    let mut failed = crate::HashSet::<(usize, usize)>::default();
    loop {
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                if groups[i].is_some()
                    && groups[j].is_some()
                    && gap[i][j] <= diameter[i].max(diameter[j])
                    && !failed.contains(&(i, j))
                {
                    pairs.push((gap[i][j], i, j));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged = false;
        for (_, i, j) in pairs {
            let fit = {
                let (a, b) = (groups[i].as_ref().unwrap(), groups[j].as_ref().unwrap());
                joint_fit(&[a, b])
            };
            let Some((k, m)) = fit else {
                failed.insert((i, j));
                continue;
            };
            let b = groups[j].take().unwrap();
            let a = groups[i].as_mut().unwrap();
            a.k = k;
            a.m = m;
            a.members.extend(b.members);
            a.own.extend(b.own);
            let row: Vec<f32> = (0..n).map(|t| gap[i][t].min(gap[j][t])).collect();
            for (t, g) in row.into_iter().enumerate() {
                gap[i][t] = g;
                gap[t][i] = g;
            }
            gap[i][i] = f32::INFINITY;
            diameter[i] = diameter[i].max(diameter[j]);
            failed.retain(|&(x, y)| x != i && y != i && x != j && y != j);
            merged = true;
            break;
        }
        if !merged {
            break;
        }
    }
    let mut out = Vec::new();
    for g in groups.into_iter().flatten() {
        if g.members.len() < 2 {
            continue;
        }
        let empty = vec![false; width * height];
        let mut candidate = Candidate::new(g.k, g.m, empty, (IVec2::MAX, IVec2::MIN), None);
        for &i in &g.members {
            candidate.cover(pieces[i], width);
        }
        let Some(r) = stage::judge(lv, target, &candidate.mask, g.k, g.m, None, true)
            .into_iter()
            .next()
        else {
            continue;
        };
        if debug {
            eprintln!(
                "translucency: merged {} pieces at k {:.3} c {:?}: {}/{} of its boundary fit",
                g.members.len(),
                r.k,
                (r.m / (1. - r.k) * 255.).round().to_array(),
                r.fitting,
                r.pairs
            );
        }
        if !r.share || !(OPACITY_MARGIN..=1. - OPACITY_MARGIN).contains(&r.k) {
            continue;
        }
        (candidate.k, candidate.m) = (r.k, r.m);
        out.push((r.fitting as f32 / r.pairs as f32, candidate));
    }
    out
}

/// The ring around a shape its colors are compared with, from 2 px past its edge, past the
/// anti-aliasing, to this far.
const RING: i32 = 12;
/// Colors are compared as palettes quantized to this many levels.
const PALETTE_STEP: f32 = 4.;
/// Fewest pixels of the core and of the ring for the comparison to say anything.
const PALETTE_MIN: usize = 30;

/// Material continuity: the share of the interior (2 px in) whose color is a ring color seen
/// through the relation, weighted by how far it moved, so k near 1 explains nothing. Zero
/// where either side is too small to tell.
fn continuity(target: &[Vec3], c: &Candidate, width: usize, height: usize) -> f32 {
    let pad = RING + 1;
    let lo = (c.min - pad).max(IVec2::ZERO);
    let hi = (c.max + pad).min(IVec2::new(width as i32, height as i32));
    let (bw, bh) = ((hi.x - lo.x) as usize, (hi.y - lo.y) as usize);
    let frame = |j: usize| (lo.y as usize + j / bw) * width + lo.x as usize + j % bw;
    let inside: Vec<bool> = (0..bw * bh).map(|j| c.mask[frame(j)]).collect();
    let outside: Vec<bool> = inside.iter().map(|v| !v).collect();
    // within two pixels of the outside and of the inside (city-block), and within the ring
    let near_outside = dilate(&outside, bw, bh, Conn::Four, 2);
    let near_inside = dilate(&inside, bw, bh, Conn::Four, 2);
    let in_ring = dilate(&inside, bw, bh, Conn::Four, RING as usize);
    let mut core: Vec<usize> = (0..bw * bh)
        .filter(|&j| inside[j] && !near_outside[j])
        .collect();
    if core.len() < PALETTE_MIN {
        core = (0..bw * bh).filter(|&j| inside[j]).collect();
    }
    let ring: Vec<usize> = (0..bw * bh)
        .filter(|&j| !inside[j] && !near_inside[j] && in_ring[j])
        .collect();
    if core.len() < PALETTE_MIN || ring.len() < PALETTE_MIN {
        return 0.;
    }
    let palette = |pixels: &[usize]| -> Vec<(Vec3, f32)> {
        let mut counts = std::collections::BTreeMap::<[i32; 3], f32>::new();
        for &j in pixels {
            let q = (target[frame(j)] * 255. / PALETTE_STEP).round().as_ivec3();
            *counts.entry(q.to_array()).or_default() += 1.;
        }
        counts
            .into_iter()
            .map(|(q, n)| {
                (
                    glam::IVec3::from_array(q).as_vec3() * PALETTE_STEP / 255.,
                    n,
                )
            })
            .collect()
    };
    let (own, around) = (palette(&core), palette(&ring));
    let seen: Vec<Vec3> = around.iter().map(|(b, _)| c.k * *b + c.m).collect();
    let (mut explained, mut total) = (0f32, 0f32);
    for (o, w) in &own {
        total += w;
        // The nearest color around through the relation, within the tolerance
        let nearest = seen
            .iter()
            .enumerate()
            .map(|(j, s)| ((*o - *s).abs().max_element(), j))
            .filter(|(d, _)| *d <= 3. * NOISE)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((_, j)) = nearest {
            let moved = (*o - around[j].0).abs().max_element();
            explained += w * ((moved - 2. * NOISE) / SHOW_THROUGH).clamp(0., 1.);
        }
    }
    explained / total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two bands with a disc at opacity one half over both: one candidate, the disc, at its
    /// relation.
    #[test]
    fn finds_the_disc_over_two_bands() {
        let (width, height) = (96usize, 96usize);
        let (top, bottom) = (Vec3::new(0.9, 0.85, 0.7), Vec3::new(0.2, 0.3, 0.5));
        let (color, opacity) = (Vec3::new(0.8, 0.2, 0.3), 0.5);
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let b = if y < height / 2 { top } else { bottom };
                // The disc's coverage of the pixel, by 4 x 4 samples
                let mut covered = 0.;
                for s in 0..16 {
                    let (u, v) = (
                        x as f32 + (s % 4) as f32 / 4. + 0.125,
                        y as f32 + (s / 4) as f32 / 4. + 0.125,
                    );
                    if (u - 48.).powi(2) + (v - 48.).powi(2) <= 30f32.powi(2) {
                        covered += 1. / 16.;
                    }
                }
                let o = b + covered * opacity * (color - b);
                for c in o.to_array() {
                    rgb.push((c * 255.).round() as u8);
                }
            }
        }
        let mut timings = crate::timing::Timings::default();
        let (ctx, hierarchy) = crate::progress::finish(crate::hierarchy::analyze(
            &rgb,
            width,
            height,
            1.,
            &mut timings,
        ));
        let lv = leaves::Leaves::new(&ctx.target, hierarchy.labels(), width, height);
        let found = crate::progress::finish(candidates(&ctx, &lv));
        let first = found.first().expect("a candidate");
        assert!((first.k - 0.5).abs() < 0.02, "k {}", first.k);
        assert!(
            (first.fill().color - color).abs().max_element() < 4. / 255.,
            "color {:?}",
            first.fill().color * 255.
        );
        let area = first.mask.iter().filter(|&&m| m).count() as f32;
        let disc = std::f32::consts::PI * 30f32.powi(2);
        assert!((area - disc).abs() < 0.05 * disc, "area {area} for {disc}");
    }
}
