//! A translucent layer's own marks as vector elements (its outline, blurred white lenses,
//! hems along its edge, lines inside), each fitted against the composite drawn so far so it
//! pays for what the output lacks. Coordinates are SVG units: pixel centres at x + 0.5.

use glam::{DVec2, DVec3};

use super::lines::{Bezier, FIT_TOL, bez, fit_cubics, ridge_radii, smooth_open, trace_chains};

use super::raster::{
    Conn, Image, NOISE, SHOW_THROUGH, dilate, gaussian_filter, label, length_px, map1, skeletonize,
    unit_of,
};

mod geometry;
mod hems;
mod paint;
mod sheen;

pub use geometry::*;
pub use hems::*;
use paint::{draw_stroke, stroke_gain, stroke_runs};
pub use sheen::*;

/// What to emit, and the planner's price of a cubic segment in squared RGB error
/// (`Context::segment_cost`), which decides where a line splits.
#[derive(Clone, Debug)]
pub struct Options {
    pub segment_cost: f64,
    /// The rule that keeps a mark (see [`Keep`]).
    pub keep: Keep,
    /// The edge term (see [`crate::edges`]): a mark's error weighted at the target's edges,
    /// so a mark that saves little error but draws an edge still pays.
    pub edge: Option<std::sync::Arc<(crate::edges::EdgeTerm, crate::edges::EdgeField)>>,
    /// How far each pixel may deviate before it counts as error (the plan's
    /// [`crate::Context::tolerance`]): a mark's gain is the residual beyond it that it
    /// removes. None for the plain squared error.
    pub tolerance: Option<std::sync::Arc<Vec<f32>>>,
    /// The layer's shape as it is drawn, its coverage per pixel, which the lenses are clipped
    /// to; None clips them to the region.
    pub clip: Option<std::sync::Arc<Vec<f32>>>,
}

/// A mark as the rule that keeps it sees it: its cubics (a line's centreline, a lens's
/// outline) and the squared error it removes from the composite.
pub struct Mark<'a> {
    pub cubics: &'a [Bezier],
    pub gain: f64,
}

/// What the linework draws: a line, or a cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkKind {
    Line,
    /// a fill the linework module cuts from an output shape along a step it misses (see
    /// [`crate::linework`]), kept by the same rule as the strokes
    Cut,
}

/// Whether a mark is kept, given the planner's price of a segment: one function for every
/// mark, so that an objective weighing a line by more than its squared error (an edge term)
/// replaces it in one place.
pub type Keep = fn(&Mark, f64) -> bool;

impl Options {
    pub fn new(segment_cost: f64, keep: Keep) -> Self {
        Options {
            segment_cost,
            keep,
            edge: None,
            tolerance: None,
            clip: None,
        }
    }

    /// The tolerance per pixel as a slice, if any.
    fn tol(&self) -> Option<&[f32]> {
        self.tolerance.as_deref().map(Vec::as_slice)
    }

    /// Whether `mark` is kept by the rule given.
    pub fn keeps(&self, mark: &Mark) -> bool {
        (self.keep)(mark, self.segment_cost)
    }

    /// What a mark drawn at `alpha` over the window `win` in `colour` does for the output's
    /// edges under the edge term, in error units (none without it): its saving again at the
    /// edge weights less one.
    fn edge_credit(
        &self,
        img: &Image,
        comp: &[DVec3],
        win: Win,
        alpha: &[f64],
        colour: DVec3,
    ) -> f64 {
        let Some(edge) = &self.edge else {
            return 0.0;
        };
        let (term, field) = (&edge.0, &edge.1);
        let mut g = 0.0;
        for (i, &a) in alpha.iter().enumerate() {
            if a <= 0.0 {
                continue;
            }
            let p = win.at(i, img.width);
            let d = comp[p];
            let n = d + a * (colour - d);
            let saved = field.residual(p, d - img.px[p]) - field.residual(p, n - img.px[p]);
            g += (field.weight(p, term.lambda) as f64 - 1.0) * saved;
        }
        g
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrokeKind {
    /// along the region's edge
    Hem,
    /// inside the region
    Line,
}

/// A stroke: cubics, a width, a colour at an opacity, round caps and joins.
#[derive(Clone, Debug)]
pub struct Stroke {
    pub cubics: Vec<Bezier>,
    pub width: f64,
    pub colour: DVec3,
    pub opacity: f64,
    pub kind: StrokeKind,
}

/// A lens: a white filled shape tapering along its centreline to points at both ends,
/// `width` at its middle, blurred by a Gaussian of `sigma` and clipped to the layer.
#[derive(Clone, Debug)]
pub struct Lens {
    pub outline: Vec<Bezier>,
    pub sigma: f64,
    pub opacity: f64,
    pub gain: f64,
}

/// The layer's marks as vector elements.
pub struct Vocabulary {
    pub lenses: Vec<Lens>,
    /// the tints above the layer as shapes, each at its own colour and opacity
    pub tints: Vec<TintShape>,
    pub strokes: Vec<Stroke>,
}

/// A tint of the pin's full fit as a shape: the outline of the pixels it explains at colour
/// `c` and opacity `1 - k` above the layer (dark folds). The peel took the tint off the
/// beneath ([`super::Above::peel`]), so the shape draws it back.
#[derive(Clone, Debug)]
pub struct TintShape {
    pub loops: Vec<Vec<Bezier>>,
    pub k: f64,
    pub c: DVec3,
}

// ----------------------------------------------------------------------------- the layer

/// The region's outline as cubics: its crack loops smoothed over half a pixel, fitted to
/// half the lines' tolerance.
pub fn outline(region: &[bool], width: usize, height: usize) -> Vec<Vec<Bezier>> {
    let u = unit_of(width);
    crack_loops(region, width, height)
        .into_iter()
        .map(|lp| fit_loop(&smooth_loop(&lp, 0.5), FIT_TOL * u * 0.5))
        .collect()
}

/// The layer's vocabulary over `base` (the traced beneath with the region at k, c drawn):
/// the tints, the lenses, the hems, then the lines inside, each fitted on what is drawn
/// before it.
pub async fn emit(
    img: &Image,
    base: &[DVec3],
    region: &[bool],
    c: DVec3,
    above: Option<&super::Above>,
    opt: &Options,
) -> Vocabulary {
    let (w, h) = (img.width, img.height);
    let mut comp = base.to_vec();
    // The tints first: shapes over the layer, drawn into the composite the lenses and
    // strokes are then fitted against
    let mut tint_shapes = Vec::new();
    if let Some(above) = above {
        for (ti, &(tk, tm)) in above.tints.iter().enumerate() {
            let mask: Vec<bool> = (0..img.len())
                .map(|i| region[i] && above.tint.get(i).copied().flatten() == Some(ti as u8))
                .collect();
            if !mask.iter().any(|v| *v) || tk <= 0.0 {
                continue;
            }
            let (tk, tm) = (f64::from(tk), tm.as_dvec3());
            for (i, m) in mask.iter().enumerate() {
                if *m {
                    comp[i] = tk * comp[i] + tm;
                }
            }
            tint_shapes.push(TintShape {
                loops: outline(&mask, w, h),
                k: tk,
                c: tm / (1.0 - tk),
            });
        }
    }
    let clip = |p: usize| match &opt.clip {
        Some(c) => f64::from(c[p]),
        None => f64::from(u8::from(region[p])),
    };
    let sts = lenses(img, &mut comp, region, &clip, above, opt).await;
    let (strength, dir) = colour_lines(img);
    let band = dilate(region, w, h, Conn::Eight, length_px(w, 3.0));
    let thin: Vec<bool> = thin_lines(&strength, &dir, w, h, 3.0 * NOISE, 6.0 * NOISE)
        .into_iter()
        .zip(&band)
        .map(|(a, b)| a && *b)
        .collect();
    let mut taken = vec![false; img.len()];
    let mut strokes = Vec::new();
    let mut take = async |pts: &[DVec2],
                          kind: StrokeKind,
                          comp: &mut Vec<DVec3>,
                          taken: &mut Vec<bool>| {
        for f in stroke_runs(img, comp, pts, opt.segment_cost, opt.tol()).await {
            crate::progress::poll().await;
            // the pieces were fitted side by side: where an earlier stroke drew over this
            // one's place, it may no longer pay
            let mark = Mark {
                cubics: &f.cubics,
                gain: stroke_gain(img, comp, &f, opt),
            };
            let of_hue = of_layer_hue(f.colour, c);
            if crate::diagnostics::debug() {
                eprintln!(
                    "marks: {kind:?} candidate at {:?} (layer c {:?}) width {:.1} opacity {:.2} \
                     gain {:.1}: {}{}",
                    (f.colour * 255.0).round().to_array(),
                    (c * 255.0).round().to_array(),
                    f.width,
                    f.opacity,
                    mark.gain,
                    if opt.keeps(&mark) {
                        "pays"
                    } else {
                        "does not pay"
                    },
                    if of_hue { "" } else { ", not the layer's hue" }
                );
            }
            if !opt.keeps(&mark) || !of_hue {
                continue;
            }
            for p in draw_stroke(img, comp, &f) {
                taken[p] = true;
            }
            strokes.push(Stroke {
                cubics: f.cubics,
                width: f.width,
                colour: f.colour,
                opacity: f.opacity,
                kind,
            });
        }
    };
    let mut hems = hem_lines(region, &strength, w, h);
    hems.sort_by_key(|p| std::cmp::Reverse(p.len()));
    for p in &hems {
        take(p, StrokeKind::Hem, &mut comp, &mut taken).await;
    }
    let near = dilate(&taken, w, h, Conn::Eight, length_px(w, 1.0));
    let rest: Vec<bool> = thin.iter().zip(&near).map(|(a, b)| *a && !*b).collect();
    let mut chains = line_chains(&rest, w, h, 5.0);
    chains.sort_by_key(|p| std::cmp::Reverse(p.len()));
    for p in &chains {
        take(p, StrokeKind::Line, &mut comp, &mut taken).await;
    }
    Vocabulary {
        lenses: sts,
        tints: tint_shapes,
        strokes,
    }
}

/// Whether a stroke's colour is the layer's own (its body colour mixed with white or black):
/// its chroma along the layer's within the show-through, or none. So the layer's deeper
/// folds are its own, and strands of another hue crossing it are not.
pub fn of_layer_hue(colour: DVec3, c: DVec3) -> bool {
    let chroma = |v: DVec3| v - DVec3::splat((v.x + v.y + v.z) / 3.0);
    let (a, b) = (chroma(colour), chroma(c));
    if a.abs().max_element() <= SHOW_THROUGH || b.length_squared() == 0.0 {
        return true;
    }
    let along = b * (a.dot(b) / b.length_squared());
    along.dot(b) >= 0.0 && (a - along).abs().max_element() <= SHOW_THROUGH
}

/// The layer's marks as model objects without its shape (the lenses, the tints, then the
/// strokes); the lenses are clipped to the layer's shape as drawn (see [`crate::Scene::marks`]).
pub fn objects(v: &Vocabulary) -> Vec<crate::model::Object> {
    use crate::model::{Geometry, Object, Paint, StrokeGeometry};
    use crate::render::FillRule;
    use crate::stroke::{Cap, Join};
    let lenses = v.lenses.iter().map(|t| Object {
        geometry: Geometry::Fill {
            loops: vec![t.outline.clone()],
            rule: FillRule::NonZero,
        },
        paint: Paint::Solid(DVec3::ONE),
        opacity: t.opacity,
        blur: t.sigma,
        clip: Some(0),
    });
    let tints = v.tints.iter().map(|t| Object {
        geometry: Geometry::Fill {
            loops: t.loops.clone(),
            rule: FillRule::EvenOdd,
        },
        paint: Paint::Solid(t.c),
        opacity: 1.0 - t.k,
        blur: 0.0,
        clip: None,
    });
    let strokes = v.strokes.iter().map(|t| Object {
        geometry: Geometry::Stroke(StrokeGeometry {
            centerline: t.cubics.clone(),
            closed: false,
            width: t.width,
            cap: Cap::Round,
            join: Join::Round,
            corners: vec![true; t.cubics.len()],
            dashes: None,
        }),
        paint: Paint::Solid(t.colour),
        opacity: t.opacity,
        blur: 0.0,
        clip: None,
    });
    lenses.chain(tints).chain(strokes).collect()
}

#[cfg(test)]
mod tests {
    use super::paint::fit_paint;
    use super::*;

    /// The coverage measured over the segments' neighbourhoods is the coverage of every
    /// segment measured at every pixel, bit for bit.
    #[test]
    fn coverage_is_the_full_scan() {
        fn full(p: &[DVec2], width: f64, win: Win) -> Vec<f64> {
            (0..win.len())
                .map(|i| {
                    let q = DVec2::new(
                        (win.x0 + i % win.w()) as f64 + 0.5,
                        (win.y0 + i / win.w()) as f64 + 0.5,
                    );
                    let mut d = f64::INFINITY;
                    if p.len() == 1 {
                        d = (q - p[0]).length();
                    }
                    for s in p.windows(2) {
                        let ab = s[1] - s[0];
                        let t =
                            ((q - s[0]).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
                        d = d.min((q - (s[0] + t * ab)).length());
                    }
                    (width / 2.0 - d + 0.5).clamp(0.0, 1.0)
                })
                .collect()
        }
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for n in [0, 1, 2, 3, 9, 40] {
            for width in [0.5, 1.0, 2.37, 6.0] {
                let p: Vec<DVec2> = (0..n)
                    .map(|_| DVec2::new(next() * 50.0 - 5.0, next() * 40.0 - 5.0))
                    .collect();
                let win = Win {
                    x0: 3,
                    y0: 2,
                    x1: 41,
                    y1: 33,
                };
                let (a, b) = (coverage(&p, width, win), full(&p, width, win));
                assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
            }
        }
    }

    #[test]
    fn crack_loop_of_a_square() {
        let (w, h) = (6, 5);
        let mut m = vec![false; w * h];
        for y in 1..4 {
            for x in 2..5 {
                m[y * w + x] = true;
            }
        }
        let loops = crack_loops(&m, w, h);
        assert_eq!(loops.len(), 1);
        // twelve cracks around a 3 x 3 square, their midpoints on its sides
        assert_eq!(loops[0].len(), 12);
        for p in &loops[0] {
            assert!(
                (p.x - 2.0).abs() < 1e-9
                    || (p.x - 5.0).abs() < 1e-9
                    || (p.y - 1.0).abs() < 1e-9
                    || (p.y - 4.0).abs() < 1e-9
            );
        }
    }

    #[test]
    fn a_pale_line_is_a_colour_line() {
        // a pale green line one pixel wide on a blue ground: off the segment between its
        // sides; a step from blue to green is on it
        let (w, h) = (9, 9);
        let blue = [190u8, 215, 240];
        let green = [200u8, 255, 200];
        let mut rgb = Vec::new();
        for _y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(if x == 4 { &green } else { &blue });
            }
        }
        let img = Image::from_rgb8(&rgb, w, h);
        let (s, _) = colour_lines(&img);
        assert!(s[4 * w + 4] > 6.0 * NOISE);
        assert!(s[4 * w + 1] < NOISE);
        let mut rgb = Vec::new();
        for _y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(if x >= 4 { &green } else { &blue });
            }
        }
        let img = Image::from_rgb8(&rgb, w, h);
        let (s, _) = colour_lines(&img);
        assert!(s.iter().all(|v| *v < 1e-9));
    }

    #[test]
    fn a_stroke_fits_its_own_render() {
        // a horizontal line of colour (0.2, 0.6, 0.2) at opacity 0.7 and width 2 over a
        // ground that changes along it (over one colour only the paint a C is told, not a):
        // the fit at the right place recovers both
        let (w, h) = (20, 11);
        let ground: Vec<DVec3> = (0..w * h)
            .map(|i| {
                DVec3::new(
                    0.3 + 0.03 * (i % w) as f64,
                    0.8,
                    0.9 - 0.02 * (i % w) as f64,
                )
            })
            .collect();
        let flat = vec![DVec2::new(3.0, 5.5), DVec2::new(17.0, 5.5)];
        let win = Win {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        };
        let cov = coverage(&flat, 2.0, win);
        let col = DVec3::new(0.2, 0.6, 0.2);
        let px: Vec<DVec3> = cov
            .iter()
            .zip(&ground)
            .map(|(c, g)| *g + 0.7 * *c * (col - *g))
            .collect();
        let img = Image {
            width: w,
            height: h,
            px,
        };
        let (c, a) = fit_paint(&img, &ground, win, &cov).unwrap();
        assert!((a - 0.7).abs() < 1e-6, "{a}");
        assert!((c - col).abs().max_element() < 1e-6);
    }
}
