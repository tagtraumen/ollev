//! A stroke's fit: its outline sampled into a strip and moved by the coverage gradient.

use super::*;

/// A piece of a stroke's outline: its points, each with the centerline sample it comes from
/// and its offset from the sample's point.
pub(super) struct Piece {
    pub(super) points: Vec<Vec2>,
    pub(super) sources: Vec<(usize, Vec2)>,
}

impl Piece {
    /// The piece from its sources, consecutive repeats dropped and wound positively.
    pub(super) fn new(sources: Vec<(usize, Vec2)>, samples: &[Sample]) -> Self {
        let mut piece = Self::with_winding(sources, samples);
        let points = &piece.points;
        let area: f32 = (0..points.len())
            .map(|i| points[i].perp_dot(points[(i + 1) % points.len()]))
            .sum();
        if area < 0. {
            piece.points.reverse();
            piece.sources.reverse();
        }
        piece
    }

    /// The piece from its sources, consecutive repeats dropped, wound as they run.
    pub(super) fn with_winding(sources: Vec<(usize, Vec2)>, samples: &[Sample]) -> Self {
        let mut points: Vec<Vec2> = Vec::with_capacity(sources.len());
        let mut kept: Vec<(usize, Vec2)> = Vec::with_capacity(sources.len());
        for (i, o) in sources {
            let p = samples[i].point + o;
            if points.last().is_none_or(|q| q.distance(p) > 1e-4) {
                points.push(p);
                kept.push((i, o));
            }
        }
        while points.len() > 1 && points[0].distance(points[points.len() - 1]) <= 1e-4 {
            points.pop();
            kept.pop();
        }
        Self {
            points,
            sources: kept,
        }
    }
}

/// Fits `stroke` to `cost`: centerline control points and width descend the error's gradient
/// (through the offsets and their turn), steps scaled by `off_by`. Returns the error reached,
/// infinite once the width passes [`MAX_WIDTH`]: a band that wide is a shape, not a stroke.
pub fn fit(cost: &CoverageCost, stroke: &mut Stroke, origin: Vec2, off_by: f32) -> f32 {
    let mut fitting = Fitting::new(stroke, origin, off_by);
    fitting.run(cost, crate::refine::MAX_STEPS);
    let (error, fitted) = fitting.finish();
    if error.is_finite() {
        *stroke = fitted;
    }
    error
}

/// Steps of a screening fit: a quarter of a whole one, after which the candidates are ranked
/// and the best fitted on ([`fit_screened`]).
pub(super) const SCREEN_STEPS: usize = crate::refine::MAX_STEPS / 4;
/// Steps of a fit between two checkpoints.
const CHECK_STEPS: usize = 5;

/// A stroke's fit in progress ([`fit`] is one run of all its steps), kept between runs.
pub(super) struct Fitting {
    stroke: Stroke,
    origin: Vec2,
    params: Vec<f32>,
    adam: crate::render::Adam,
    best: (f32, Vec<f32>),
    since_best: usize,
    steps_done: usize,
    stopped: bool,
    flattening: Vec<usize>,
    // a dashed stroke keeps its dashes where they start on the centerline's parameters
    mappings: Option<Vec<Vec<(usize, f32, f32)>>>,
}

impl Fitting {
    pub(super) fn new(stroke: &Stroke, origin: Vec2, off_by: f32) -> Self {
        use crate::refine::LEARNING_RATE;
        let params = stroke.params(origin);
        Self {
            // The steps scale with how far the start is expected to be off, as a centerline
            // simplified at a tolerance of 4 px is
            adam: crate::render::Adam::new(LEARNING_RATE * off_by, 0.),
            best: (f32::INFINITY, params.clone()),
            params,
            since_best: 0,
            steps_done: 0,
            stopped: false,
            flattening: stroke.steps(),
            mappings: stroke.dashes.as_ref().map(|_| stroke.dash_mappings()),
            stroke: stroke.clone(),
            origin,
        }
    }

    /// The best error so far: infinite before the first step and once the width ran past
    /// [`MAX_WIDTH`].
    pub(super) fn error(&self) -> f32 {
        self.best.0
    }

    /// Starts the steps afresh from the best point reached: a restart settles lower than
    /// steps already shrunk on a plateau.
    pub(super) fn restart(&mut self) {
        self.stroke = self.stroke.with_params(&self.best.1, self.origin);
        self.params = self.stroke.params(self.origin);
        self.flattening = self.stroke.steps();
        self.mappings = self
            .stroke
            .dashes
            .as_ref()
            .map(|_| self.stroke.dash_mappings());
        let rate = self.adam.lr_points;
        self.adam = crate::render::Adam::new(rate, 0.);
        self.best.0 = f32::INFINITY;
        self.since_best = 0;
        self.steps_done = 0;
        self.stopped = false;
    }

    /// [`Fitting::run`] with a checkpoint every few steps (see [`crate::progress`]): a large
    /// stroke's step renders it whole.
    async fn run_checked(&mut self, cost: &CoverageCost, until: usize) {
        for at in (CHECK_STEPS..until).step_by(CHECK_STEPS).chain([until]) {
            crate::progress::poll().await;
            self.run(cost, at);
        }
    }

    /// The best error and stroke reached.
    pub(super) fn finish(self) -> (f32, Stroke) {
        let error = self.error();
        (error, self.stroke.with_params(&self.best.1, self.origin))
    }

    /// Runs the fit on until `until` steps are done in all, or it has settled.
    pub(super) fn run(&mut self, cost: &CoverageCost, until: usize) {
        use crate::refine::{MIN_STEP_SCALE, PLATEAU_GAIN, PLATEAU_STEPS};
        use crate::render::ParamKind;
        let kinds = vec![ParamKind::Point; self.params.len()];
        let points = (self.params.len() - 1) / 2;
        while !self.stopped && self.steps_done < until {
            self.steps_done += 1;
            let current = self.stroke.with_params(&self.params, self.origin);
            if current.width > MAX_WIDTH {
                self.best.0 = f32::INFINITY;
                self.stopped = true;
                break;
            }
            let (samples, pieces) = match &self.mappings {
                Some(m) => current.geometry_of(m),
                None => current.geometry(Some(&self.flattening)),
            };
            let (value, grad) = cost.evaluate(&path_of(self.origin, &pieces));
            if value < self.best.0 * (1. - PLATEAU_GAIN) {
                self.since_best = 0;
            } else {
                self.since_best += 1;
            }
            if value < self.best.0 {
                self.best = (value, self.params.clone());
            }
            if self.since_best >= PLATEAU_STEPS {
                self.adam.lr_scale *= 0.5;
                self.since_best = 0;
                if self.adam.lr_scale < MIN_STEP_SCALE {
                    self.stopped = true;
                    break;
                }
            }
            let flat = gradient(
                &current,
                &samples,
                &pieces,
                &grad,
                points,
                self.params.len(),
            );
            self.adam.step_flat(&mut self.params, &kinds, &flat);
        }
    }
}

/// The error's gradient on the centerline's control points and the width, carried back from
/// the subpath points (an anchor and the two thirds of each edge) to the pieces' corners, then
/// to the samples and the width.
fn gradient(
    current: &Stroke,
    samples: &[Sample],
    pieces: &[Piece],
    grad: &[Vec<Vec2>],
    points: usize,
    len: usize,
) -> Vec<f32> {
    let mut d_samples = vec![Vec2::ZERO; samples.len()];
    // How the error changes as each sample's offsets turn with the tangent, which on a wide
    // stroke outweighs the sample's own motion
    let mut d_turn = vec![0f32; samples.len()];
    let mut d_width = 0f32;
    for (piece, d) in pieces.iter().zip(grad) {
        let n = piece.points.len();
        let mut d_corner = vec![Vec2::ZERO; n];
        let mut base = 0;
        for i in 0..n {
            let (a, b) = (i, (i + 1) % n);
            if piece.points[a].distance(piece.points[b]) <= 1e-4 {
                continue;
            }
            if base + 2 >= d.len() {
                break;
            }
            d_corner[a] += d[base] + d[base + 1] * (2. / 3.) + d[base + 2] * (1. / 3.);
            d_corner[b] += d[base + 1] * (1. / 3.) + d[base + 2] * (2. / 3.);
            base += 3;
        }
        for ((i, offset), dc) in piece.sources.iter().zip(&d_corner) {
            d_samples[*i] += *dc;
            d_width += dc.dot(*offset) / current.width;
            d_turn[*i] += dc.dot(offset.perp());
        }
    }
    let mut flat = vec![0f32; len];
    for ((s, ds), dt) in samples.iter().zip(&d_samples).zip(&d_turn) {
        let (k, t) = (s.cubic, s.t);
        let u = 1. - t;
        let weights = [u * u * u, 3. * u * u * t, 3. * u * t * t, t * t * t];
        // The tangent's angle turns with the derivative's component across it: by the
        // derivative's Bernstein weights over its length
        let c = &current.centerline[k];
        let derivative = (c[1] - c[0]) * (3. * u * u)
            + (c[2] - c[1]) * (6. * u * t)
            + (c[3] - c[2]) * (3. * t * t);
        let speed = derivative.length();
        let across = if speed > 1e-6 {
            derivative.perp() / (speed * speed)
        } else {
            Vec2::ZERO
        };
        let turn_weights = [
            -3. * u * u,
            3. * u * u - 6. * u * t,
            6. * u * t - 3. * t * t,
            3. * t * t,
        ];
        for (j, w) in weights.iter().enumerate() {
            let index = if current.closed {
                (3 * k + j) % points
            } else {
                3 * k + j
            };
            let g = *ds * *w + across * (dt * turn_weights[j]);
            flat[2 * index] += g.x;
            flat[2 * index + 1] += g.y;
        }
    }
    flat[len - 1] = d_width;
    flat
}

/// Screens `candidates` (each with its starting step scale) for [`SCREEN_STEPS`] and fits the
/// best by `rank` (the error with any price) on from where its screen stopped, the steps'
/// scale restored; one candidate alone is fitted whole. `None` where none fits.
pub(super) async fn fit_screened(
    cost: &CoverageCost,
    origin: Vec2,
    candidates: Vec<(Stroke, f32)>,
    rank: impl Fn(f32, &Stroke) -> f64,
) -> Option<(f32, Stroke)> {
    let mut fittings: Vec<Fitting> = candidates
        .iter()
        .map(|(s, off_by)| Fitting::new(s, origin, *off_by))
        .collect();
    let best = if fittings.len() > 1 {
        for f in &mut fittings {
            f.run_checked(cost, SCREEN_STEPS).await;
        }
        // the first of equals, in the candidates' order
        let key = |f: &Fitting| rank(f.error(), &f.stroke);
        let mut best: Option<Fitting> = None;
        for f in fittings.into_iter().filter(|f| f.error().is_finite()) {
            if best.as_ref().is_none_or(|b| key(&f) < key(b)) {
                best = Some(f);
            }
        }
        let mut best = best?;
        best.restart();
        best.run_checked(cost, crate::refine::MAX_STEPS - SCREEN_STEPS)
            .await;
        best
    } else {
        let mut only = fittings.pop()?;
        only.run_checked(cost, crate::refine::MAX_STEPS).await;
        only
    };
    let (error, stroke) = best.finish();
    error.is_finite().then_some((error, stroke))
}

/// A point of the flattened centerline, and whether it is a corner joint, where the join goes.
#[derive(Clone, Copy)]
pub(super) struct Sample {
    pub(super) point: Vec2,
    pub(super) corner: bool,
    /// Which cubic of the centerline the sample lies on, and where.
    pub(super) cubic: usize,
    pub(super) t: f32,
}

/// The unit tangent of `c` at `t`, along the chord where the derivative vanishes.
pub(crate) fn tangent(c: &Cubic, t: f32) -> Vec2 {
    let u = 1. - t;
    let d =
        (c[1] - c[0]) * (3. * u * u) + (c[2] - c[1]) * (6. * u * t) + (c[3] - c[2]) * (3. * t * t);
    if d.length_squared() > 1e-8 {
        d.normalize()
    } else {
        (c[3] - c[0]).normalize_or_zero()
    }
}

/// The miter offset at a corner between unit normals `n1` and `n2` at distance `half`, or
/// None beyond the miter limit.
pub(super) fn miter(n1: Vec2, n2: Vec2, half: f32, limit: f32) -> Option<Vec2> {
    let bisector = n1 + n2;
    let cos_half = bisector.length() / 2.;
    if cos_half < 1e-4 {
        return None;
    }
    let length = half / cos_half;
    if length > limit * half {
        return None;
    }
    Some(bisector.normalize() * length)
}

/// Offsets along the arc from `from` to `to` about the origin, the shorter way, both ends
/// included.
pub(super) fn arc(from: Vec2, to: Vec2) -> Vec<Vec2> {
    let center = Vec2::ZERO;
    let radius = from.length();
    let (a0, a1) = (from.y.atan2(from.x), to.y.atan2(to.x));
    let mut sweep = a1 - a0;
    while sweep > std::f32::consts::PI {
        sweep -= 2. * std::f32::consts::PI;
    }
    while sweep < -std::f32::consts::PI {
        sweep += 2. * std::f32::consts::PI;
    }
    let steps = ((sweep.abs() / std::f32::consts::FRAC_PI_2) * ROUND_STEPS as f32).ceil() as usize;
    let steps = steps.max(1);
    (0..=steps)
        .map(|i| {
            let a = a0 + sweep * i as f32 / steps as f32;
            center + Vec2::new(a.cos(), a.sin()) * radius
        })
        .collect()
}

/// The length of `c`'s control polygon, at least its arc length.
pub(crate) fn hull_length(c: &Cubic) -> f32 {
    (c[1] - c[0]).length() + (c[2] - c[1]).length() + (c[3] - c[2]).length()
}

/// The straight cubic from `a` to `b`, its handles on the thirds.
pub(crate) fn line(a: Vec2, b: Vec2) -> Cubic {
    [a, a.lerp(b, 1. / 3.), a.lerp(b, 2. / 3.), b]
}

/// A closed polygon as line cubics.
pub(super) fn polygon(points: &[Vec2]) -> Vec<Cubic> {
    let n = points.len();
    (0..n)
        .filter(|&i| points[i].distance(points[(i + 1) % n]) > 1e-4)
        .map(|i| line(points[i], points[(i + 1) % n]))
        .collect()
}
