//! What the parts of a trace take, for the host's progress (see [`crate::progress`]): each
//! step weighted by an estimate of its time from what is known where it is weighed, the
//! picture's pixels from the start, the runs the segmentation merges once they are found and
//! the planned shapes once the plan is decided. In milliseconds per megapixel, per thousand
//! runs and per hundred planned shapes, measured on photos, paintings and flat art with the
//! default and with every stage.

// clean of these, which the parent module still expects
#![cfg_attr(not(test), deny(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::too_many_lines,
))]

use crate::Stages;

/// A step by name with its weight (see [`crate::progress::enter`]).
pub(crate) type Step = (&'static str, f64);

/// The first looks, per megapixel.
const PREVIEW: f64 = 800.;
/// Reading the picture: the smoothing, the runs of equal colour, their merging (per
/// megapixel and per thousand runs), the blends dissolved and the region tree.
const SMOOTH: f64 = 450.;
const SEED: f64 = 50.;
const SEGMENT: (f64, f64) = (30., 10.);
const DISSOLVE: f64 = 20.;
const TREE: f64 = 150.;
/// Runs of equal colour per megapixel, before the segmentation finds them.
const RUNS_PER_MEGAPIXEL: f64 = 300_000.;
/// The picture's uncertainty, per megapixel.
const UNCERTAINTY: f64 = 600.;
/// Planning, per megapixel and per hundred planned shapes.
const PLAN: (f64, f64) = (200., 70.);
/// Drawing the plan, and each shape again, per megapixel and per hundred planned shapes.
const DRAW: (f64, f64) = (2100., 80.);
const REDRAW: (f64, f64) = (900., 30.);
/// Recolouring, settling and cleaning the shapes, per megapixel and per hundred planned
/// shapes.
const SHAPES: (f64, f64) = (500., 150.);
/// The strokes, per megapixel and per hundred planned shapes.
const STROKES: (f64, f64) = (1000., 250.);
/// The passes over the whole scene, per megapixel and per hundred planned shapes.
const INSTANCING: (f64, f64) = (50., 5.);
const ECONOMY: (f64, f64) = (500., 250.);
const POLISH: (f64, f64) = (1500., 20.);
/// The linework and the effects, per megapixel of a busy picture: a flat one, of few shapes
/// for its pixels, has few lines and runs of bands to try.
const LINEWORK: f64 = 16_000.;
const EFFECTS: f64 = 18_000.;
/// Planned shapes per megapixel at which the linework and the effects take half their time.
const BUSY: f64 = 300.;
/// The families' gate, per megapixel.
const FAMILIES: f64 = 4000.;
/// Planned shapes per megapixel, before the plan tells.
const SHAPES_PER_MEGAPIXEL: f64 = 1500.;
/// The result's lowering for the host, as a share of the trace.
pub(crate) const LOWERING: f64 = 0.02;

/// What a level's work is reckoned in: its picture's megapixels and its planned shapes in
/// hundreds.
#[derive(Clone, Copy)]
pub(crate) struct Work {
    megapixels: f64,
    hundreds: f64,
}

/// `n` as a float, saturating.
fn float(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

impl Work {
    /// A level of `pixels`, its shapes estimated until the plan is decided.
    pub(crate) fn new(pixels: usize) -> Self {
        let megapixels = float(pixels) / 1e6;
        Self {
            megapixels,
            hundreds: megapixels * SHAPES_PER_MEGAPIXEL / 100.,
        }
    }

    /// The level once it planned `shapes`.
    pub(crate) fn planned(self, shapes: usize) -> Self {
        Self {
            hundreds: float(shapes) / 100.,
            ..self
        }
    }

    /// `per` megapixel and hundred planned shapes.
    fn of(self, per: (f64, f64)) -> f64 {
        per.0 * self.megapixels + per.1 * self.hundreds
    }

    /// The steps of a level's trace with `stages`: the picture's own (`top`, with its first
    /// looks and passes) or one beneath a layer or in a window, with its families where
    /// `families`.
    pub(crate) fn level(self, stages: Stages, top: bool, families: bool) -> Vec<Step> {
        let mut steps = Vec::new();
        if top {
            steps.push(("preview", PREVIEW * self.megapixels));
        }
        steps.extend([
            ("analyze", self.reading(None)),
            ("uncertainty", UNCERTAINTY * self.megapixels),
        ]);
        steps.extend(self.drawing(stages, top, families));
        steps
    }

    /// The steps of reading the picture, the runs of equal colour found where `runs`.
    pub(crate) fn analyze(self, runs: Option<usize>) -> [Step; 5] {
        let m = self.megapixels;
        let runs = runs.map_or(m * RUNS_PER_MEGAPIXEL, float) / 1000.;
        [
            ("smooth", SMOOTH * m),
            ("seed", SEED * m),
            ("segment", SEGMENT.0 * m + SEGMENT.1 * runs),
            ("dissolve", DISSOLVE * m),
            ("tree", TREE * m),
        ]
    }

    /// What reading the picture takes in all (see [`Work::analyze`]).
    pub(crate) fn reading(self, runs: Option<usize>) -> f64 {
        self.analyze(runs).iter().map(|s| s.1).sum()
    }

    /// The steps of a level from its plan on, as [`Work::level`].
    pub(crate) fn drawing(self, stages: Stages, top: bool, families: bool) -> [Step; 7] {
        let families = if families && stages.translucency {
            FAMILIES * self.megapixels
        } else {
            0.
        };
        let passes = if top {
            self.passes(stages).iter().map(|s| s.1).sum()
        } else {
            0.
        };
        [
            ("plan", self.of(PLAN)),
            ("draw", self.of(DRAW)),
            ("redraw", self.of(REDRAW)),
            ("shapes", self.shapes(stages).iter().map(|s| s.1).sum()),
            ("families", families),
            ("passes", passes),
            ("overlay", 0.),
        ]
    }

    /// What the families stage takes where `whole` families are traced whole: its gate, a
    /// level beneath each, and a discovery's.
    pub(crate) fn families(self, stages: Stages, whole: usize) -> f64 {
        FAMILIES * self.megapixels + self.beneath(stages) * (float(whole) + 1.)
    }

    /// What a level beneath a layer takes, whole: its families aside, as its own trace.
    pub(crate) fn beneath(self, stages: Stages) -> f64 {
        self.level(stages, false, false).iter().map(|s| s.1).sum()
    }

    /// The steps of the shapes stage: the recolouring, the settling, the cleanup and the
    /// strokes.
    pub(crate) fn shapes(self, stages: Stages) -> [Step; 4] {
        let shapes = self.of(SHAPES);
        [
            ("recolor", 0.4 * shapes),
            ("settle", 0.4 * shapes),
            ("cleanup", 0.2 * shapes),
            ("strokes", if stages.strokes { self.of(STROKES) } else { 0. }),
        ]
    }

    /// The steps of the passes over the whole scene.
    pub(crate) fn passes(self, stages: Stages) -> [Step; 6] {
        let m = self.megapixels;
        let on = |on: bool, w: f64| if on { w } else { 0. };
        // the share of a busy picture's time the linework and effects take
        let busy = self.hundreds * 100. / (self.hundreds * 100. + BUSY * m).max(1.);
        [
            ("linework", on(stages.strokes, LINEWORK * m * busy)),
            ("effects", on(stages.effects, EFFECTS * m * busy)),
            ("marks", 0.),
            ("instancing", on(stages.instancing, self.of(INSTANCING))),
            ("economy", self.of(ECONOMY)),
            ("polish", self.of(POLISH)),
        ]
    }
}
