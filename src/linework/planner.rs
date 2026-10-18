//! Where a candidate goes in the stack, and its stand-in shapes left out or refitted.

use super::*;

// ----------------------------------------------------------------------------- the planner

/// What the planner's linework pass found and kept ([`into_scene`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct Planned {
    pub missing_lines: usize,
    pub missing_steps: usize,
    /// candidates fitted, and those that paid on their first fit
    pub proposed: usize,
    pub paying: usize,
    /// strokes (one shape per subpath) and cuts kept
    pub strokes: usize,
    pub cuts: usize,
    /// of those, kept below a shape of the scene rather than on top
    pub below: usize,
    /// cut candidates whose step fades out, closed along the cheapest path
    pub closed: usize,
    /// shapes that stood in for a kept line, left out or refitted with it drawn
    pub dropped: usize,
    pub refitted: usize,
}

/// Most shapes of the stack a candidate is tried below, the topmost that overlap it first:
/// the stroke recognition's count (`stroke::MAX_POSITIONS`).
pub(super) const MAX_POSITIONS: usize = 2;

/// A window of the scene as a candidate placed at `position` of the stack sees it: what is
/// drawn below that place, and the shapes from it up as colour and transmittance (the
/// composite is `colour + transmittance * below`), each shape at its fill's opacity.
pub(super) struct Site {
    pub(super) window: crate::Window,
    pub(super) below: Vec<DVec3>,
    pub(super) colour: Vec<DVec3>,
    pub(super) through: Vec<DVec3>,
}

/// The scene as a candidate is weighed against it: as it is, with one shape left out, or with
/// one shape at another fill (a shape that stood in for the line, see [`stand_ins`]).
// copied, a few per candidate: the fill's size does not matter
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug)]
pub(super) enum Variant {
    Same,
    Drop(usize),
    Paint(usize, crate::Fill),
}

impl Variant {
    /// The fill shape `k` is drawn with, none where it is left out.
    pub(super) fn fill<'a>(&'a self, scene: &'a crate::Scene, k: usize) -> Option<&'a crate::Fill> {
        match self {
            Variant::Drop(d) if *d == k => None,
            Variant::Paint(p, f) if *p == k => Some(f),
            _ => Some(&scene.shapes[k].fill),
        }
    }
}

/// The image index of pixel `j` of `window`, in an image `width` wide.
pub(super) fn image_index(window: crate::Window, j: usize, width: usize) -> usize {
    (window.y + j / window.width) * width + window.x + j % window.width
}

/// Composites `shape` (the scene's shape `k`, for its clip, if it is one) at `fill` over a
/// window's colour and transmittance.
pub(super) fn composite_into(
    scene: &crate::Scene,
    window: crate::Window,
    shape: &crate::Shape,
    k: Option<usize>,
    fill: &crate::Fill,
    colour: &mut [DVec3],
    through: &mut [DVec3],
) {
    if !window.intersects(shape.min, shape.max) {
        return;
    }
    let clip = k.and_then(|k| scene.clip_mask(window, k));
    // Over the rows and columns the shape's raster shares with the window: it covers nothing
    // beyond them
    let raster = shape.raster();
    let x0 = raster.x0.max(window.x);
    let x1 = (raster.x0 + raster.w).min(window.x + window.width);
    let y0 = raster.y0.max(window.y);
    let y1 = (raster.y0 + raster.h).min(window.y + window.height);
    for y in y0..y1 {
        for x in x0..x1 {
            let c = raster.values[(y - raster.y0) * raster.w + x - raster.x0];
            let j = (y - window.y) * window.width + x - window.x;
            let covered = c * clip.as_ref().map_or(1.0, |m| m[j]);
            if covered <= 0.0 {
                continue;
            }
            let (f, alpha) = fill.paint_pixel(x, y);
            let a = (covered * alpha) as f64;
            if a <= 0.0 {
                continue;
            }
            let f = f.as_dvec3();
            let c0 = colour[j];
            colour[j] = c0 + a * (f - c0);
            through[j] *= 1.0 - a;
        }
    }
}

/// The scene under `variant` over a window, on white, with `insert` put into the stack at its
/// position (before the scene's shape there).
pub(super) fn composite(
    scene: &crate::Scene,
    window: crate::Window,
    variant: &Variant,
    insert: Option<(usize, &[crate::Shape])>,
) -> Vec<DVec3> {
    let n = window.width * window.height;
    let (mut colour, mut through) = (vec![DVec3::ONE; n], vec![DVec3::ZERO; n]);
    let put = |colour: &mut Vec<DVec3>, through: &mut Vec<DVec3>| {
        if let Some((_, shapes)) = insert {
            for s in shapes {
                composite_into(scene, window, s, None, &s.fill, colour, through);
            }
        }
    };
    for k in 0..scene.shapes.len() {
        if insert.is_some_and(|(p, _)| p == k) {
            put(&mut colour, &mut through);
        }
        composite_shape(scene, window, variant, k, &mut colour, &mut through);
    }
    if insert.is_some_and(|(p, _)| p >= scene.shapes.len()) {
        put(&mut colour, &mut through);
    }
    colour
}

/// Composites the scene's shape `k` as `variant` draws it, see [`composite_into`].
fn composite_shape(
    scene: &crate::Scene,
    window: crate::Window,
    variant: &Variant,
    k: usize,
    colour: &mut [DVec3],
    through: &mut [DVec3],
) {
    if let Some(f) = variant.fill(scene, k) {
        composite_into(scene, window, &scene.shapes[k], Some(k), f, colour, through);
    }
}

impl Site {
    pub(super) fn new(scene: &crate::Scene, window: crate::Window, position: usize) -> Self {
        Self::under(scene, window, position, &Variant::Same)
    }

    pub(super) fn under(
        scene: &crate::Scene,
        window: crate::Window,
        position: usize,
        variant: &Variant,
    ) -> Self {
        let n = window.width * window.height;
        let below = if let Variant::Same = variant {
            scene
                .render(window, 0..position, [1.0; 3])
                .iter()
                .map(|v| v.as_dvec3())
                .collect()
        } else {
            let (mut c, mut t) = (vec![DVec3::ONE; n], vec![DVec3::ZERO; n]);
            for k in 0..position {
                composite_shape(scene, window, variant, k, &mut c, &mut t);
            }
            c
        };
        let (mut colour, mut through) = (vec![DVec3::ZERO; n], vec![DVec3::ONE; n]);
        for k in position..scene.shapes.len() {
            composite_shape(scene, window, variant, k, &mut colour, &mut through);
        }
        Site {
            window,
            below,
            colour,
            through,
        }
    }

    pub(super) fn at(&self, j: usize, width: usize) -> usize {
        image_index(self.window, j, width)
    }

    /// The composite at window pixel `j` with `paint` drawn over what is below at alpha `a`.
    pub(super) fn with(&self, j: usize, a: f64, paint: DVec3) -> DVec3 {
        let d = self.below[j];
        self.colour[j] + self.through[j] * (d + a * (paint - d))
    }

    /// What drawing `paint` at alpha per window pixel saves under the objective: the squared
    /// error, and the edge term's extra weight at the target's edges on that saving.
    pub(super) fn saving(
        &self,
        img: &Image,
        field: &EdgeField,
        lambda: f64,
        alpha: &[f64],
        paint: &dyn Fn(usize) -> DVec3,
    ) -> (f64, f64) {
        let (mut plain, mut edge) = (0.0, 0.0);
        for (j, &a) in alpha.iter().enumerate() {
            if a <= 0.0 {
                continue;
            }
            let p = self.at(j, img.width);
            let o = img.px[p];
            let saved = field.residual(p, self.with(j, 0.0, DVec3::ZERO) - o)
                - field.residual(p, self.with(j, a, paint(j)) - o);
            plain += saved;
            edge += (field.weight(p, lambda) as f64 - 1.0) * saved;
        }
        (plain, edge)
    }

    /// A stroke's colour and opacity over this site, as [`fit_paint`] fits them but through
    /// the shapes above: per pixel and channel `o - L - T d = T a cov (C - d)`, with the same
    /// pull toward an opaque line and the same gamut refusal.
    pub(super) fn fit_paint(&self, img: &Image, cov: &[f64]) -> Option<(DVec3, f64)> {
        let mut ata = DMat4::ZERO;
        let mut atb = DVec4::ZERO;
        let mut n = 0;
        for (j, &w) in cov.iter().enumerate() {
            if w <= 0.05 {
                continue;
            }
            n += 1;
            let p = self.at(j, img.width);
            let (o, d, l, t) = (img.px[p], self.below[j], self.colour[j], self.through[j]);
            for c in 0..3 {
                let mut row = DVec4::ZERO;
                row[c] = t[c] * w;
                row[3] = -t[c] * w * d[c];
                let y = o[c] - l[c] - t[c] * d[c];
                atb += row * y;
                ata += DMat4::from_cols(row * row[0], row * row[1], row * row[2], row * row[3]);
            }
        }
        if n < 3 {
            return None;
        }
        let pull = 1e-4 * n as f64;
        ata.w_axis.w += pull;
        atb.w += pull;
        if ata.determinant().abs() < 1e-12 {
            return None;
        }
        let sol = ata.inverse() * atb;
        let a = sol[3].clamp(0.0, 1.0);
        if !a.is_finite() || a < 1e-3 {
            return None;
        }
        let (mut num, mut den) = (DVec3::ZERO, DVec3::ZERO);
        for (j, &w) in cov.iter().enumerate() {
            if w <= 0.05 {
                continue;
            }
            let p = self.at(j, img.width);
            let (o, d, l, t) = (img.px[p], self.below[j], self.colour[j], self.through[j]);
            let k = t * (a * w);
            num += (o - l - t * d + k * d) * k;
            den += k * k;
        }
        if den.min_element() <= 1e-12 {
            return None;
        }
        let raw = num / den;
        let c = raw.clamp(DVec3::ZERO, DVec3::ONE);
        if a * (raw - c).abs().max_element() > NOISE {
            return None;
        }
        Some((c, a))
    }
}

/// The places in the stack a candidate over `window` is tried at: on top, and below each of
/// the topmost [`MAX_POSITIONS`] shapes reaching into it (so a line can go below a shape that
/// stood in for it and a translucent layer over it).
pub(super) fn positions(scene: &crate::Scene, window: crate::Window) -> Vec<usize> {
    let top = scene.shapes.len();
    let mut out = vec![top];
    for k in (1..top).rev() {
        if out.len() > MAX_POSITIONS {
            break;
        }
        let s = &scene.shapes[k];
        if window.intersects(s.min, s.max) {
            out.push(k);
        }
    }
    out
}

/// The window over a set of shapes, grown by the anti-aliasing reach.
pub(super) fn window_of(shapes: &[crate::Shape], width: usize, height: usize) -> crate::Window {
    let (mut lo, mut hi) = (glam::Vec2::INFINITY, glam::Vec2::NEG_INFINITY);
    for s in shapes {
        lo = lo.min(s.min);
        hi = hi.max(s.max);
    }
    crate::Window::around(lo, hi, (REACH + 1) as f32, width, height)
}

/// The shapes' joint coverage of a window, each drawn over the others.
pub(super) fn joint_coverage(shapes: &[crate::Shape], window: crate::Window) -> Vec<f64> {
    let mut left = vec![1.0f64; window.width * window.height];
    for s in shapes {
        for (l, c) in left.iter_mut().zip(s.coverage(window)) {
            *l *= 1.0 - c as f64;
        }
    }
    left.into_iter().map(|l| 1.0 - l).collect()
}

/// The target's colour on a stroke's two sides: the mean of the samples past its
/// anti-aliased rim either side of its centreline.
pub(super) fn side_colour(img: &Image, paths: &[Vec<DVec2>], width: f64) -> DVec3 {
    let off = width / 2.0 + REACH as f64 + 0.5;
    let (mut sum, mut n) = (DVec3::ZERO, 0.0f64);
    for p in paths {
        for (q, nrm) in p.iter().zip(normals(p)) {
            for s in [1.0, -1.0] {
                let r = *q + s * off * nrm - DVec2::splat(0.5);
                sum += img.sample(r.x, r.y);
                n += 1.0;
            }
        }
    }
    sum / n.max(1.0)
}

/// A kept candidate's shapes go into the stack at `position`, in order.
pub(super) fn insert_at(scene: &mut crate::Scene, position: usize, shapes: Vec<crate::Shape>) {
    for (k, s) in shapes.into_iter().enumerate() {
        scene.insert(position + k, s);
    }
}

pub(super) fn to_cubic(b: &Bezier) -> crate::tracer::Cubic {
    b.map(DVec2::as_vec2)
}

/// The shapes a stroke candidate can be drawn as, with their centreline cubics and price
/// beyond a mark's: the fitted paths, a stroked path each; and for a dash row one dashed
/// stroke through its gaps (a row of stitches is one simple edge), a segment for the pattern.
pub(super) fn forms(
    ctx: &crate::Context,
    f: &Fit,
    paint: &crate::Fill,
) -> Vec<(Vec<crate::Shape>, Vec<Bezier>, f64)> {
    use crate::stroke::{Cap, Dashes, Join, Stroke};
    let stroke = |centerline: Vec<crate::tracer::Cubic>, dashes: Option<Dashes>| {
        let mut s = Stroke {
            centerline,
            closed: false,
            width: f.width as f32,
            cap: Cap::Round,
            join: Join::Round,
            corners: Vec::new(),
            dashes,
        }
        .shape(*paint);
        s.smooth = true;
        s
    };
    let mut out = Vec::new();
    let paths: Vec<&Vec<Bezier>> = f.paths.iter().filter(|p| !p.is_empty()).collect();
    if !paths.is_empty() {
        let n = paths.len() as f64;
        out.push((
            paths
                .iter()
                .map(|p| stroke(p.iter().map(to_cubic).collect(), None))
                .collect(),
            paths.iter().copied().flatten().copied().collect(),
            (n - 1.0) * ctx.segment_cost + n * ctx.path_cost,
        ));
    }
    if let Some((through, spans)) = &f.row
        && spans.len() > 1
    {
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let dash = median(spans.iter().map(|s| s.1 - s.0).collect());
        // the pattern's period and phase by least squares on the dashes' starts, each at the
        // whole number of periods the median spacing puts it at
        let rough = median(spans.windows(2).map(|s| s[1].0 - s[0].0).collect());
        let (mut sk, mut ss, mut skk, mut sks, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for s in spans {
            let k = ((s.0 - spans[0].0) / rough.max(1e-9)).round();
            sk += k;
            ss += s.0;
            skk += k * k;
            sks += k * s.0;
            n += 1.0;
        }
        let den = n * skk - sk * sk;
        let period = if den > 1e-9 {
            (n * sks - sk * ss) / den
        } else {
            rough
        };
        let phase = (ss - period * sk) / n;
        let gap = period - dash;
        let centre = fit_cubics(through, FIT_TOL);
        // a gap the eye sees is wider than the round caps eat of it, by the anti-aliasing
        // reach: members meeting at a junction are one line, not dashes
        if dash > 0.0 && gap > f.width + REACH as f64 && !centre.is_empty() {
            let offset = (period - phase.rem_euclid(period)).rem_euclid(period);
            out.push((
                vec![stroke(
                    centre.iter().map(to_cubic).collect(),
                    Some(Dashes {
                        array: vec![dash as f32, gap as f32],
                        offset: offset as f32,
                    }),
                )],
                centre,
                ctx.segment_cost + ctx.path_cost,
            ));
        }
    }
    out
}

/// A candidate at its best form and place in the stack: what it saves under the objective
/// less its extra price (`gain`, what the keep rule sees), that less a mark's price too
/// (`net`, what candidates are compared by), and the shapes to insert at `position`.
pub(super) struct Choice {
    pub(super) net: f64,
    pub(super) gain: f64,
    /// its price beyond a mark's, taken off `gain` already
    pub(super) extra: f64,
    pub(super) width: f64,
    pub(super) kind: MarkKind,
    pub(super) shapes: Vec<crate::Shape>,
    pub(super) cubics: Vec<Bezier>,
    pub(super) position: usize,
}

impl Choice {
    /// Whether the planner's one keep rule for marks (`plan::keep_mark`) keeps it.
    pub(super) fn kept(&self, keep: Keep, segment_cost: f64) -> bool {
        keep(
            &Mark {
                cubics: &self.cubics,
                gain: self.gain,
            },
            segment_cost,
        )
    }
}

/// A stroke candidate in each of its [`forms`] at its places in the stack, its paint refitted
/// at each: the best.
pub(super) fn try_stroke(
    ctx: &crate::Context,
    scene: &crate::Scene,
    img: &Image,
    field: &EdgeField,
    lambda: f64,
    f: &Fit,
    variant: &Variant,
) -> Option<Choice> {
    let (w, h) = (img.width, img.height);
    let paint = crate::Fill {
        color: f.colour.as_vec3(),
        gradient: None,
        opacity: f.opacity as f32,
    };
    let sides = side_colour(img, &f.flat, f.width);
    let mut best: Option<Choice> = None;
    let all = forms(ctx, f, &paint);
    let mut consider = |shapes: &[crate::Shape],
                        cubics: &[Bezier],
                        extra: f64,
                        width: f64,
                        position: usize,
                        gain: f64,
                        paint: (DVec3, f64)| {
        let net = gain - (cubics.len() as f64 + 1.0) * ctx.segment_cost;
        if best.as_ref().is_none_or(|b| net > b.net) {
            best = Some(Choice {
                net,
                gain,
                extra,
                width,
                kind: MarkKind::Line,
                shapes: shapes
                    .iter()
                    .cloned()
                    .map(|mut s| {
                        s.fill.color = paint.0.as_vec3();
                        s.fill.opacity = paint.1 as f32;
                        s
                    })
                    .collect(),
                cubics: cubics.to_vec(),
                position,
            });
        }
    };
    for (shapes, cubics, extra) in &all {
        let window = window_of(shapes, w, h);
        if window.is_empty() {
            continue;
        }
        let cov = joint_coverage(shapes, window);
        for position in positions(scene, window) {
            let site = Site::under(scene, window, position, variant);
            let Some((c, a)) = site.fit_paint(img, &cov) else {
                continue;
            };
            // a line stands out from its sides above the detector's weak level: a fainter
            // paint is a correction of the composite's tone (faint strokes on shading ridges)
            if a * (c - sides).abs().max_element() < 3.0 * NOISE {
                continue;
            }
            let alpha: Vec<f64> = cov.iter().map(|v| a * v).collect();
            let (plain, edge) = site.saving(img, field, lambda, &alpha, &|_| c);
            let gain = plain + f.smooth * edge - extra;
            consider(shapes, cubics, *extra, f.width, position, gain, (c, a));
        }
    }
    best
}

/// The shapes that may have stood in for a line over `window` (as a dark wedge drawing the
/// line's darkness as a tone): the topmost [`MAX_POSITIONS`] and one more, background aside,
/// whose box lies at least half within the window.
pub(super) fn stand_ins(scene: &crate::Scene, window: crate::Window) -> Vec<usize> {
    let (wx0, wy0) = (window.x as f32, window.y as f32);
    let (wx1, wy1) = (wx0 + window.width as f32, wy0 + window.height as f32);
    let mut out = Vec::new();
    for k in (1..scene.shapes.len()).rev() {
        if out.len() > MAX_POSITIONS {
            break;
        }
        let s = &scene.shapes[k];
        let area = ((s.max.x - s.min.x) * (s.max.y - s.min.y)).max(1e-6);
        let ox = (s.max.x.min(wx1) - s.min.x.max(wx0)).max(0.0);
        let oy = (s.max.y.min(wy1) - s.min.y.max(wy0)).max(0.0);
        if 2.0 * ox * oy >= area {
            out.push(k);
        }
    }
    out
}

/// What a joint move saves over `window`: the scene as it is against the scene under
/// `variant` with `insert` drawn, the squared error and the edge term's extra weight on it.
pub(super) fn joint_saving(
    scene: &crate::Scene,
    img: &Image,
    field: &EdgeField,
    lambda: f64,
    window: crate::Window,
    variant: &Variant,
    insert: (usize, &[crate::Shape]),
) -> (f64, f64) {
    let before = composite(scene, window, &Variant::Same, None);
    // shapes put over the whole scene as it is are composited over `before` alone
    let after = if matches!(variant, Variant::Same) && insert.0 >= scene.shapes.len() {
        let mut after = before.clone();
        let mut through = vec![DVec3::ZERO; after.len()];
        for s in insert.1 {
            composite_into(scene, window, s, None, &s.fill, &mut after, &mut through);
        }
        after
    } else {
        composite(scene, window, variant, Some(insert))
    };
    let (mut plain, mut edge) = (0.0, 0.0);
    for (j, (b, a)) in before.iter().zip(&after).enumerate() {
        let p = image_index(window, j, img.width);
        let o = img.px[p];
        let saved = field.residual(p, *b - o) - field.residual(p, *a - o);
        plain += saved;
        edge += (field.weight(p, lambda) as f64 - 1.0) * saved;
    }
    (plain, edge)
}

/// The window over a set of shapes and the scene's shape `k`.
pub(super) fn union_window(
    scene: &crate::Scene,
    shapes: &[crate::Shape],
    k: usize,
) -> crate::Window {
    let s = &scene.shapes[k];
    let (mut lo, mut hi) = (s.min, s.max);
    for t in shapes {
        lo = lo.min(t.min);
        hi = hi.max(t.max);
    }
    crate::Window::around(lo, hi, (REACH + 1) as f32, scene.width, scene.height)
}

/// The line `f` with the stand-in shape `k` left out or refitted (flat or a gradient, by
/// least squares with `alone`, the line's best choice as the scene is, drawn), whichever the
/// objective prefers, with what leaving the shape out saves of its price.
#[allow(clippy::too_many_arguments)]
pub(super) fn joint_move(
    ctx: &crate::Context,
    scene: &crate::Scene,
    img: &Image,
    field: &EdgeField,
    lambda: f64,
    f: &Fit,
    alone: Option<&Choice>,
    k: usize,
) -> Option<(Choice, Variant)> {
    let shape = &scene.shapes[k];
    let mut variants = vec![Variant::Drop(k)];
    if let Some(line) = alone
        && shape.fill.gradient.is_none()
    {
        let window = union_window(scene, &line.shapes, k);
        let at = |c: f32| {
            let fill = crate::Fill {
                color: glam::Vec3::splat(c),
                ..shape.fill
            };
            composite(
                scene,
                window,
                &Variant::Paint(k, fill),
                Some((line.position, &line.shapes)),
            )
        };
        let (zero, one) = (at(0.0), at(1.0));
        let (mut num, mut den) = (DVec3::ZERO, DVec3::ZERO);
        for (j, (a, b)) in zero.iter().zip(&one).enumerate() {
            let p = image_index(window, j, img.width);
            let wt = field.weight(p, lambda) as f64;
            let slope = *b - *a;
            num += wt * slope * (img.px[p] - *a);
            den += wt * slope * slope;
        }
        if den.min_element() > 1e-9 {
            let colour = (num / den).clamp(DVec3::ZERO, DVec3::ONE);
            variants.push(Variant::Paint(
                k,
                crate::Fill {
                    color: colour.as_vec3(),
                    ..shape.fill
                },
            ));
        }
        // and as a linear gradient along the target's strongest change under the shape
        // ([`area_fills`]), its ends by least squares: a shading one colour cannot follow
        let cov = shape.coverage(window);
        let under: Vec<usize> = cov
            .iter()
            .enumerate()
            .filter(|(_, c)| **c >= 0.5)
            .map(|(j, _)| image_index(window, j, img.width))
            .collect();
        if let (_, Some(CutFill::Linear { from, to, .. })) = area_fills(img, &under) {
            let d = to - from;
            let t = |j: usize| {
                let p = DVec2::new(
                    (window.x + j % window.width) as f64 + 0.5,
                    (window.y + j / window.width) as f64 + 0.5,
                );
                ((p - from).dot(d) / d.length_squared().max(1e-12)).clamp(0.0, 1.0)
            };
            // per channel the 2 x 2 normal equations of the two ends
            let (mut s00, mut s01, mut s11) = (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO);
            let (mut b0, mut b1) = (DVec3::ZERO, DVec3::ZERO);
            for (j, (a, b)) in zero.iter().zip(&one).enumerate() {
                let p = image_index(window, j, img.width);
                let wt = field.weight(p, lambda) as f64;
                let slope = *b - *a;
                let (u, v) = (slope * (1.0 - t(j)), slope * t(j));
                let r = img.px[p] - *a;
                s00 += wt * u * u;
                s01 += wt * u * v;
                s11 += wt * v * v;
                b0 += wt * u * r;
                b1 += wt * v * r;
            }
            let det = s00 * s11 - s01 * s01;
            if det.min_element() > 1e-12 {
                let c0 = ((s11 * b0 - s01 * b1) / det).clamp(DVec3::ZERO, DVec3::ONE);
                let c1 = ((s00 * b1 - s01 * b0) / det).clamp(DVec3::ZERO, DVec3::ONE);
                variants.push(Variant::Paint(
                    k,
                    crate::Fill {
                        color: c0.as_vec3(),
                        gradient: Some(crate::render::Gradient::Linear {
                            start: from.as_vec2(),
                            end: to.as_vec2(),
                            to: crate::render::opaque(c1.as_vec3().to_array()),
                            stops: crate::render::scene::Stops::NONE,
                        }),
                        ..shape.fill
                    },
                ));
            }
        }
    }
    let mut best: Option<(Choice, Variant)> = None;
    for v in variants {
        let Some(mut ch) = try_stroke(ctx, scene, img, field, lambda, f, &v) else {
            continue;
        };
        let window = union_window(scene, &ch.shapes, k);
        let (plain, edge) = joint_saving(
            scene,
            img,
            field,
            lambda,
            window,
            &v,
            (ch.position, &ch.shapes),
        );
        let saved = match v {
            Variant::Drop(_) => shape.cost_segments() * ctx.segment_cost + ctx.path_cost,
            // a refitted fill pays the difference of its paint, as the planner's recolouring
            // prices a fill (a gradient its ends, `fill::GRADIENT_SEGMENTS`)
            Variant::Paint(_, f) => {
                (crate::fill::cost_segments(&shape.fill) - crate::fill::cost_segments(&f))
                    * ctx.segment_cost
            }
            Variant::Same => 0.0,
        };
        ch.gain = plain + f.smooth * edge - ch.extra + saved;
        ch.net = ch.gain - (ch.cubics.len() as f64 + 1.0) * ctx.segment_cost;
        if best.as_ref().is_none_or(|b| ch.net > b.0.net) {
            best = Some((ch, v));
        }
    }
    best
}
