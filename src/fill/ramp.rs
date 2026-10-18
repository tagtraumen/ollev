//! Multi-stop ramps and focal radial gradients fitted to a region's samples.

use super::*;

/// Sums over the samples in one stretch of a gradient's parameter `t`, enough for the least
/// squares ramp over them.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Moments {
    pub(super) n: f64,
    pub(super) t: f64,
    pub(super) tt: f64,
    pub(super) c: DVec3,
    pub(super) tc: DVec3,
    pub(super) cc: f64,
}

impl Moments {
    pub(super) fn add(&self, o: &Self) -> Self {
        Self {
            n: self.n + o.n,
            t: self.t + o.t,
            tt: self.tt + o.tt,
            c: self.c + o.c,
            tc: self.tc + o.tc,
            cc: self.cc + o.cc,
        }
    }

    pub(super) fn sub(&self, o: &Self) -> Self {
        Self {
            n: self.n - o.n,
            t: self.t - o.t,
            tt: self.tt - o.tt,
            c: self.c - o.c,
            tc: self.tc - o.tc,
            cc: self.cc - o.cc,
        }
    }
}

/// The samples a bin of a ramp's profile holds at least, on average.
pub(super) const SAMPLES_PER_BIN: usize = 4;

/// How much the fit pulls each stop's color toward the two-stop ramp, per sample: a stop with
/// no samples either side (a singular system) takes the ramp's; elsewhere the pull is tiny.
pub(super) const RIDGE: f64 = 1e-6;

/// The samples' colors along a gradient's parameter, binned a pixel apart along the gradient,
/// with prefix sums, so the least squares ramp through stops at any bin boundaries comes from
/// a few sums per stop.
pub(super) struct Profile {
    pub(super) bins: usize,
    /// `prefix[k]` sums the bins before `k`.
    pub(super) prefix: Vec<Moments>,
    /// The two-stop ramp: its colors at 0 and 1.
    pub(super) line: [DVec3; 2],
}

/// The sums a piece of the ramp from `a` to `a + w` contributes to the normal equations, with
/// `u = (t - a) / w` running from 0 to 1 over it: the sums of `(1 - u)^2`, `u (1 - u)`, `u^2`,
/// and of the colors times `1 - u` and `u`.
pub(super) fn piece(m: &Moments, a: f64, w: f64) -> ([f64; 3], [DVec3; 2]) {
    let su = (m.t - m.n * a) / w;
    let suu = (m.tt - 2. * a * m.t + m.n * a * a) / (w * w);
    let suc = (m.tc - m.c * a) / w;
    ([m.n - 2. * su + suu, su - suu, suu], [m.c - suc, suc])
}

impl Profile {
    pub(super) fn new(samples: &[(Vec2, Vec3)], gradient: &Gradient) -> Option<Self> {
        // A pixel along the gradient per bin, so a stop lands within a pixel of where it fits,
        // and a few samples per bin, so the search compares stretches of the ramp rather than
        // single pixels where the samples are few
        let length = match *gradient {
            Gradient::Linear { start, end, .. } => start.distance(end),
            Gradient::Radial { radius, .. } => radius,
            Gradient::Path { near, far, .. } => (far - near).abs(),
        };
        let bins = (length.ceil() as usize)
            .min(samples.len() / SAMPLES_PER_BIN)
            .max(2);
        let mut moments = vec![Moments::default(); bins];
        for (p, c) in samples {
            let t = gradient.t(*p) as f64;
            let c = c.as_dvec3();
            let m = &mut moments[((t * bins as f64) as usize).min(bins - 1)];
            m.n += 1.;
            m.t += t;
            m.tt += t * t;
            m.c += c;
            m.tc += c * t;
            m.cc += c.length_squared();
        }
        let mut prefix = vec![Moments::default(); bins + 1];
        for k in 0..bins {
            prefix[k + 1] = prefix[k].add(&moments[k]);
        }
        // The two-stop ramp, by regression of the colors on t
        let all = prefix[bins];
        if all.n < 2. {
            return None;
        }
        let var = all.tt - all.t * all.t / all.n;
        if var < 1e-12 * all.n {
            return None;
        }
        let slope = (all.tc - all.c * (all.t / all.n)) / var;
        let at_zero = (all.c - slope * all.t) / all.n;
        Some(Self {
            bins,
            prefix,
            line: [at_zero, at_zero + slope],
        })
    }

    /// The least squares colors at the stops at bin boundaries `knots` (increasing, from 0 to
    /// the bin count), the ramp piecewise linear between them, and its squared error.
    pub(super) fn solve(&self, knots: &[usize]) -> (Vec<DVec3>, f64) {
        let (x, sse) = self.solve_in_place(knots);
        (x[..knots.len()].to_vec(), sse)
    }

    /// [`Profile::solve`] into an array, without allocating, for the searches that solve many
    /// times.
    pub(super) fn solve_in_place(&self, knots: &[usize]) -> ([DVec3; MAX_STOPS + 2], f64) {
        const K: usize = MAX_STOPS + 2;
        let k = knots.len();
        let tau = |i: usize| knots[i] as f64 / self.bins as f64;
        // A piece of no width holds nothing
        let mut pieces = [([0f64; 3], [DVec3::ZERO; 2]); K];
        for s in 0..k - 1 {
            if knots[s + 1] > knots[s] {
                let m = self.prefix[knots[s + 1]].sub(&self.prefix[knots[s]]);
                pieces[s] = piece(&m, tau(s), tau(s + 1) - tau(s));
            }
        }
        // The normal equations of the hat functions: tridiagonal, `d` and `off` (between i
        // and i + 1), right side `r`; the ridge pulls toward the two-stop ramp
        let total = self.prefix[self.bins];
        let ridge = RIDGE * total.n / k as f64;
        let mut d = [ridge; K];
        let mut off = [0f64; K];
        let mut r = [DVec3::ZERO; K];
        for (i, r) in r.iter_mut().enumerate().take(k) {
            *r = self.line[0].lerp(self.line[1], tau(i)) * ridge;
        }
        for (s, (q, c)) in pieces.iter().enumerate().take(k - 1) {
            d[s] += q[0];
            off[s] += q[1];
            d[s + 1] += q[2];
            r[s] += c[0];
            r[s + 1] += c[1];
        }
        // Thomas' algorithm
        for i in 1..k {
            let w = off[i - 1] / d[i - 1];
            d[i] -= w * off[i - 1];
            r[i] -= r[i - 1] * w;
        }
        let mut x = [DVec3::ZERO; K];
        x[k - 1] = r[k - 1] / d[k - 1];
        for i in (0..k - 1).rev() {
            x[i] = (r[i] - x[i + 1] * off[i]) / d[i];
        }
        // The error about the ramp, piece by piece: c^2 - 2 c . ramp + ramp^2
        let mut sse = total.cc;
        for (s, (q, c)) in pieces.iter().enumerate().take(k - 1) {
            let (p0, p1) = (x[s], x[s + 1]);
            sse += -2. * (p0.dot(c[0]) + p1.dot(c[1]))
                + p0.length_squared() * q[0]
                + 2. * p0.dot(p1) * q[1]
                + p1.length_squared() * q[2];
        }
        (x, sse.max(0.))
    }

    /// The squared error over bins `a..b` of the piece of ramp from `p` at bin boundary `a` to
    /// `q` at `b`.
    pub(super) fn piece_error(&self, a: usize, b: usize, p: DVec3, q: DVec3) -> f64 {
        if b == a {
            return 0.;
        }
        let m = self.prefix[b].sub(&self.prefix[a]);
        let bins = self.bins as f64;
        let (w, c) = piece(&m, a as f64 / bins, (b - a) as f64 / bins);
        m.cc - 2. * (p.dot(c[0]) + q.dot(c[1]))
            + p.length_squared() * w[0]
            + 2. * p.dot(q) * w[1]
            + q.length_squared() * w[2]
    }

    /// The knots with one more stop where it helps most (each free bin boundary judged by the
    /// piece it splits, ends held, in closed form), or None without room. Only bends: a hard
    /// step is an edge, which a shape draws.
    pub(super) fn insertion(&self, knots: &[usize], colors: &[DVec3]) -> Option<Vec<usize>> {
        if knots.len() >= MAX_STOPS + 2 {
            return None;
        }
        let bins = self.bins as f64;
        let mut best: Option<(usize, f64)> = None;
        for s in 0..knots.len() - 1 {
            let (a, b) = (knots[s], knots[s + 1]);
            if b - a < 2 {
                continue;
            }
            let (p, q) = (colors[s], colors[s + 1]);
            let before = self.piece_error(a, b, p, q);
            for m in a + 1..b {
                let left = self.prefix[m].sub(&self.prefix[a]);
                let right = self.prefix[b].sub(&self.prefix[m]);
                let (wl, cl) = piece(&left, a as f64 / bins, (m - a) as f64 / bins);
                let (wr, cr) = piece(&right, m as f64 / bins, (b - m) as f64 / bins);
                // The error of the two halves is quadratic in the new stop's color x, the
                // first half's u^2 and the second's (1 - v)^2 weighing it against the colors
                // and the held ends
                let weight = wl[2] + wr[0];
                if weight < 1e-12 {
                    continue;
                }
                let x = (cl[1] + cr[0] - p * wl[1] - q * wr[1]) / weight;
                let after = self.piece_error(a, m, p, x) + self.piece_error(m, b, x, q);
                let gain = before - after;
                if best.is_none_or(|(_, g)| gain > g) {
                    best = Some((m, gain));
                }
            }
        }
        let (m, _) = best?;
        let mut more = knots.to_vec();
        more.insert(knots.partition_point(|&k| k < m), m);
        Some(more)
    }

    /// Moves each stop between the ends to where the error is least with the others held, by
    /// a pattern search in bins from an eighth of the profile down to one; sweeps until nothing
    /// moves. Returns the error.
    pub(super) fn polish(&self, knots: &mut [usize], mut sse: f64) -> f64 {
        for _ in 0..POLISH_SWEEPS {
            let mut moved = false;
            for i in 1..knots.len() - 1 {
                let mut step = (self.bins / 8).max(1);
                loop {
                    let mut improved = false;
                    for candidate in [knots[i].saturating_sub(step), knots[i] + step] {
                        if candidate <= knots[i - 1] || candidate >= knots[i + 1] {
                            continue;
                        }
                        let old = knots[i];
                        knots[i] = candidate;
                        let (_, e) = self.solve_in_place(knots);
                        if e < sse {
                            (sse, improved, moved) = (e, true, true);
                        } else {
                            knots[i] = old;
                        }
                    }
                    if !improved {
                        if step == 1 {
                            break;
                        }
                        step /= 2;
                    }
                }
            }
            if !moved {
                break;
            }
        }
        sse
    }
}

/// How many times the stops are swept over when moved to where they fit best: each sweep
/// moves every stop given the others, and the second rarely moves anything.
pub(super) const POLISH_SWEEPS: usize = 3;

/// The ramp along `gradient` fitting `samples` best: stops go in where they help most while each
/// lowers the error by more than `stop_cost`, then are polished. Returns the start color and the
/// gradient, unclamped, or None for samples that do not spread along it.
pub fn ramp(
    samples: &[(Vec2, Vec3)],
    gradient: &Gradient,
    stop_cost: f32,
) -> Option<(Vec3, Gradient)> {
    let profile = Profile::new(samples, gradient)?;
    let mut knots = vec![0, profile.bins];
    let (mut colors, mut sse) = profile.solve(&knots);
    // A stop is judged before the stops move (which only lowers the error), to save polishing
    // for the final stop that does not pay
    while let Some(mut more) = profile.insertion(&knots, &colors) {
        let (_, e) = profile.solve(&more);
        if sse - e <= stop_cost as f64 {
            break;
        }
        profile.polish(&mut more, e);
        knots = more;
        (colors, sse) = profile.solve(&knots);
    }
    let mut gradient = *gradient;
    *gradient.to_mut() = crate::render::opaque(colors[colors.len() - 1].as_vec3().to_array());
    let stops: Vec<(f32, [f32; 3])> = (1..knots.len() - 1)
        .map(|i| {
            (
                knots[i] as f32 / profile.bins as f32,
                colors[i].as_vec3().to_array(),
            )
        })
        .collect();
    *gradient.stops_mut() = Stops::opaque(&stops);
    Some((colors[0].as_vec3(), gradient))
}

/// `fill` with its gradient's ramp refitted through the stops that pay `stop_cost` (see
/// [`ramp`]), its colors clamped to what can be painted. A solid fill, or a gradient along
/// which the samples do not spread, comes back as it is.
pub(super) fn with_stops(samples: &[(Vec2, Vec3)], fill: &Fill, stop_cost: f32) -> Fill {
    let Some(gradient) = fill.gradient else {
        return *fill;
    };
    let Some((from, mut gradient)) = ramp(samples, &gradient, stop_cost) else {
        return *fill;
    };
    let paint = |[r, g, b, a]: [f32; 4]| {
        let [r, g, b] = Vec3::new(r, g, b).clamp(Vec3::ZERO, Vec3::ONE).to_array();
        [r, g, b, a]
    };
    *gradient.to_mut() = paint(gradient.to());
    for c in gradient.stops_mut().colors_mut() {
        *c = paint(*c);
    }
    Fill {
        color: from.clamp(Vec3::ZERO, Vec3::ONE),
        gradient: Some(gradient),
        ..*fill
    }
}

/// At most this many rounds of the focus search, see [`focal`].
pub(super) const FOCAL_ROUNDS: usize = 8;

/// How far from the center a radial gradient's focus may go, as a share of its radius.
pub(super) const FOCUS_LIMIT: f32 = 0.95;

/// `concentric` with its focus moved off center where the isolines are off-center circles (a
/// sphere lit near its edge), searched from the sample nearest the start color with center and
/// radius in turn. None when the focus stays at the center or leaves the `extent`.
pub(super) fn focal(samples: &[(Vec2, Vec3)], concentric: &Fill, extent: &[Vec2]) -> Option<Fill> {
    let Some(Gradient::Radial { center, radius, .. }) = concentric.gradient else {
        return None;
    };
    let stride = samples.len().div_ceil(CENTER_SAMPLES);
    let few: Vec<(Vec2, Vec3)> = samples.iter().step_by(stride).copied().collect();
    let error = |center: Vec2, focus: Vec2, radius: f32| {
        let g = Gradient::Radial {
            center,
            focus: focus.clamp_length_max(FOCUS_LIMIT * radius),
            transform: glam::Mat2::IDENTITY,
            radius,
            to: [0.; 4],
            stops: Stops::NONE,
        };
        t_regression_error(&few, |p| g.t(p))
    };
    let start = concentric.color;
    let brightest = few
        .iter()
        .min_by(|a, b| a.1.distance(start).total_cmp(&b.1.distance(start)))?
        .0;
    let (mut center, mut focus, mut radius) = (center, brightest - center, radius);
    let size = {
        let (min, max) = few.iter().fold((Vec2::MAX, Vec2::MIN), |(lo, hi), (p, _)| {
            (lo.min(*p), hi.max(*p))
        });
        (max - min).max_element()
    };
    let concentric_error = error(center, Vec2::ZERO, radius);
    // Focus, center and radius trade off along a valley the searches crawl along: rounds go
    // on while one improves, each at half the steps of the one before
    let mut best = f32::INFINITY;
    let mut reach = size / 4.;
    for _ in 0..FOCAL_ROUNDS {
        let before = best;
        let f = search2(focus, reach, |f| error(center, f, radius));
        focus = f.0;
        // The focal point stays where it is while the center moves
        let c = search2(center, reach / 2., |c| {
            error(c, focus - (c - center), radius)
        });
        focus -= c.0 - center;
        (center, best) = c;
        let mut step = reach / 2.;
        while step >= 0.25 {
            let mut moved = false;
            for r in [radius - step, radius + step] {
                if r > 1. {
                    let e = error(center, focus, r);
                    if e < best {
                        (radius, best, moved) = (r, e, true);
                    }
                }
            }
            if !moved {
                step /= 2.;
            }
        }
        if best >= before {
            break;
        }
        reach = (reach / 2.).max(1.);
    }
    let focus = focus.clamp_length_max(FOCUS_LIMIT * radius);
    if focus.length() < 0.5 || best >= concentric_error {
        return None;
    }
    // The focal point is on the shape: off it the start color is extrapolated from far away,
    // as in `pixel_1up`'s ring, whose focus in the hole spilled over the rim
    let focal_point = center + focus;
    if !extent.iter().any(|p| p.distance_squared(focal_point) <= 1.) {
        return None;
    }
    Some(Fill {
        gradient: Some(Gradient::Radial {
            center,
            focus,
            transform: glam::Mat2::IDENTITY,
            radius,
            to: concentric.gradient.map_or([0.; 4], |g| g.to()),
            stops: Stops::NONE,
        }),
        ..*concentric
    })
}

/// Most samples the search for a radial gradient's ellipse reads.
const STRETCH_SAMPLES: usize = 2000;

/// The radius, axis angle and minor to major ratio of the ellipse about `center` whose radial
/// gradient (the colour linear in the elliptical radius up to the rim, held beyond) fits
/// `samples` least, by a pattern search from the circle of `radius`; the circle where no
/// ellipse fits better.
pub fn stretch(samples: &[(Vec2, Vec3)], center: Vec2, radius: f32) -> (f32, f32, f32) {
    let stride = samples.len().div_ceil(STRETCH_SAMPLES);
    let few: Vec<(Vec2, Vec3)> = samples.iter().step_by(stride).copied().collect();
    let error = |r: f32, angle: f32, squash: f32| {
        let back =
            glam::Mat2::from_diagonal(Vec2::new(1., 1. / squash)) * glam::Mat2::from_angle(-angle);
        t_regression_error(&few, |p| ((back * (p - center)).length() / r).min(1.))
    };
    // Started from the circle or an ellipse along one of four axes, whichever fits best: from
    // the circle alone the axis cannot move (b007-kiln-0401's tall arches)
    let starts = (0..4).map(|k| (radius, k as f32 * std::f32::consts::FRAC_PI_4, 0.75));
    let (mut best, mut least) = std::iter::once((radius, 0f32, 1f32))
        .chain(starts)
        .map(|c| (c, error(c.0, c.1, c.2)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("a start");
    let (mut step_r, mut step_a, mut step_s) = (0.25 * radius, std::f32::consts::FRAC_PI_4, 0.25);
    while step_s > 1. / 64. {
        let (r, a, s) = best;
        let mut moved = false;
        for (dr, da, ds) in [
            (step_r, 0., 0.),
            (-step_r, 0., 0.),
            (0., step_a, 0.),
            (0., -step_a, 0.),
            (0., 0., step_s),
            (0., 0., -step_s),
        ] {
            let c = (r + dr, a + da, (s + ds).min(1.));
            if c.0 <= 1. || c.2 <= step_s {
                continue;
            }
            let e = error(c.0, c.1, c.2);
            if e < least {
                (best, least, moved) = (c, e, true);
            }
        }
        if !moved {
            (step_r, step_a, step_s) = (0.5 * step_r, 0.5 * step_a, 0.5 * step_s);
        }
    }
    best
}

/// The plane map of the ellipse of axis `angle` and minor to major ratio `squash` (see
/// [`stretch`]), the identity for a circle.
pub fn ellipse_transform(angle: f32, squash: f32) -> glam::Mat2 {
    match squash < 1. {
        true => glam::Mat2::from_angle(angle) * glam::Mat2::from_diagonal(Vec2::new(1., squash)),
        false => glam::Mat2::IDENTITY,
    }
}

/// `concentric` stretched into the ellipse that fits `samples` best (an oval lit from its
/// middle, `b004-radial-05`), or None where the circle fits best.
pub(super) fn elliptical(samples: &[(Vec2, Vec3)], concentric: &Fill) -> Option<Fill> {
    let Some(Gradient::Radial {
        center, radius, to, ..
    }) = concentric.gradient
    else {
        return None;
    };
    let (radius, angle, squash) = stretch(samples, center, radius);
    (squash < 1.).then(|| Fill {
        gradient: Some(Gradient::Radial {
            center,
            focus: Vec2::ZERO,
            radius,
            transform: ellipse_transform(angle, squash),
            to,
            stops: Stops::NONE,
        }),
        ..*concentric
    })
}

/// The residual of the regression of the samples' colors on `t`.
pub(super) fn t_regression_error(samples: &[(Vec2, Vec3)], t: impl Fn(Vec2) -> f32) -> f32 {
    let n = samples.len() as f32;
    let ts: Vec<f32> = samples.iter().map(|(p, _)| t(*p)).collect();
    let t_mean = ts.iter().sum::<f32>() / n;
    let mean = samples.iter().map(|(_, c)| *c).sum::<Vec3>() / n;
    let (mut stt, mut stc, mut scc) = (0f32, Vec3::ZERO, 0f32);
    for ((_, c), t) in samples.iter().zip(&ts) {
        let (dt, dc) = (t - t_mean, *c - mean);
        stt += dt * dt;
        stc += dc * dt;
        scc += dc.length_squared();
    }
    if stt < 1e-9 {
        return scc;
    }
    (scc - stc.length_squared() / stt).max(0.)
}

/// The point that gives the least `error`, by a pattern search in eight directions from
/// `start` with steps from `step` halving down to a quarter of a pixel, and its error.
pub(super) fn search2(start: Vec2, mut step: f32, error: impl Fn(Vec2) -> f32) -> (Vec2, f32) {
    let (mut at, mut best) = (start, error(start));
    while step >= 0.25 {
        let next = (0..8)
            .map(|k| at + Vec2::from_angle(k as f32 * std::f32::consts::FRAC_PI_4) * step)
            .map(|p| (p, error(p)))
            .min_by(|x, y| x.1.total_cmp(&y.1));
        match next {
            Some((p, e)) if e < best => (at, best) = (p, e),
            _ => step /= 2.,
        }
    }
    (at, best)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Costs with a stop and a focus at half a gradient, as the segment costs have them.
    pub(super) fn costs(gradient: f32) -> Costs {
        Costs {
            gradient,
            stop: gradient / 2.,
            focus: gradient / 2.,
            segment: gradient / 2.,
        }
    }

    fn grid(f: impl Fn(Vec2) -> Vec3) -> Vec<(Vec2, Vec3)> {
        (0..32)
            .flat_map(|y| (0..32).map(move |x| Vec2::new(x as f32 + 0.5, y as f32 + 0.5)))
            .map(|p| (p, f(p)))
            .collect()
    }

    #[test]
    fn flat_stays_solid() {
        let samples = grid(|_| Vec3::splat(0.4));
        let (fill, sse) = fit(&samples, &[], Vec3::splat(0.4), None, costs(0.1), &[]);
        assert!(fill.gradient.is_none());
        assert!(sse < 1e-6);
    }

    #[test]
    fn vertical_ramp_is_linear() {
        let samples = grid(|p| Vec3::new(0.2, 0.3, 0.1) + Vec3::new(0.02, 0.01, 0.0) * p.y);
        let (fill, sse) = fit(&samples, &[], Vec3::splat(0.5), None, costs(0.1), &[]);
        let Some(Gradient::Linear { start, end, .. }) = fill.gradient else {
            panic!("expected a linear gradient, got {fill:?}");
        };
        assert!(sse < 1e-4, "sse {sse}");
        assert!(
            (start.x - end.x).abs() < 1e-3,
            "axis not vertical: {start:?} {end:?}"
        );
        for (p, c) in &samples {
            assert!(fill.at(*p).distance(*c) < 1e-3);
        }
    }

    #[test]
    fn partial_ring_finds_its_center() {
        // The lower right quarter of a ring lit from the ring's center
        let center = Vec2::new(16., 16.);
        let samples: Vec<_> =
            grid(|p| Vec3::new(1., 0.9, 0.6) - Vec3::new(0., 0.02, 0.03) * p.distance(center))
                .into_iter()
                .filter(|(p, _)| {
                    let r = p.distance(center);
                    r > 8. && r < 15. && p.x > center.x && p.y > center.y
                })
                .collect();
        let (fill, _) = fit(&samples, &[], Vec3::splat(0.5), None, costs(0.1), &[]);
        let Some(Gradient::Radial { center: found, .. }) = fill.gradient else {
            panic!("expected a radial gradient, got {fill:?}");
        };
        assert!(found.distance(center) < 0.5, "center found at {found}");
    }

    #[test]
    fn disc_is_radial() {
        let center = Vec2::new(16., 16.);
        let samples: Vec<_> =
            grid(|p| Vec3::new(1., 0.9, 0.6) - Vec3::new(0., 0.02, 0.03) * p.distance(center))
                .into_iter()
                .filter(|(p, _)| p.distance(center) < 15.)
                .collect();
        let (fill, _) = fit(&samples, &[], Vec3::splat(0.5), None, costs(0.1), &[]);
        assert!(
            matches!(fill.gradient, Some(Gradient::Radial { .. })),
            "{fill:?}"
        );
        for (p, c) in &samples {
            assert!(
                fill.at(*p).distance(*c) < 2e-2,
                "{p} {c} vs {}",
                fill.at(*p)
            );
        }
    }
}

#[cfg(test)]
mod stop_tests {
    use super::tests::costs;
    use super::*;

    fn rows(height: usize, f: impl Fn(f32) -> Vec3) -> Vec<(Vec2, Vec3)> {
        (0..height)
            .flat_map(|y| (0..64).map(move |x| Vec2::new(x as f32 + 0.5, y as f32 + 0.5)))
            .map(|p| (p, f(p.y / height as f32)))
            .collect()
    }

    /// Red to white to blue down the image, the turn halfway, as `b004-stop-count-01`.
    #[test]
    fn three_colors_take_one_stop() {
        let (red, white, blue) = (
            Vec3::new(0.9, 0.22, 0.27),
            Vec3::new(0.95, 0.98, 0.93),
            Vec3::new(0.27, 0.48, 0.62),
        );
        let samples = rows(200, |t| {
            if t < 0.5 {
                red.lerp(white, t / 0.5)
            } else {
                white.lerp(blue, (t - 0.5) / 0.5)
            }
        });
        let (fill, sse) = fit(&samples, &[], Vec3::splat(0.5), None, costs(1.), &[]);
        let gradient = fill.gradient.expect("a gradient");
        assert!(matches!(gradient, Gradient::Linear { .. }), "{fill:?}");
        assert_eq!(gradient.stops().len(), 1, "{fill:?}");
        let (offset, color) = gradient.stops().get(0);
        assert!((offset - 0.5).abs() < 0.02, "stop at {offset}");
        assert!(
            glam::Vec4::from(color).truncate().distance(white) < 0.02,
            "{color:?}"
        );
        assert!(sse < 0.5, "sse {sse}");
    }

    /// A disc lit from near its edge, as `b004-radial-06`: the focus is found.
    #[test]
    fn off_center_light_has_a_focus() {
        let (center, radius, focal) = (Vec2::new(64., 64.), 50., Vec2::new(100., 64.));
        let truth = Gradient::Radial {
            center,
            focus: focal - center,
            radius,
            transform: glam::Mat2::IDENTITY,
            to: [0.48, 0.17, 0.75, 1.],
            stops: Stops::NONE,
        };
        let from = Vec3::new(1., 0.95, 0.);
        let samples: Vec<(Vec2, Vec3)> = (0..128)
            .flat_map(|y| (0..128).map(move |x| Vec2::new(x as f32 + 0.5, y as f32 + 0.5)))
            .filter(|p| p.distance(center) < radius - 1.)
            .map(|p| (p, truth.color(from, truth.t(p))))
            .collect();
        let (fill, sse) = fit(&samples, &[], Vec3::splat(0.5), None, costs(1.), &[]);
        let Some(Gradient::Radial {
            center: c, focus, ..
        }) = fill.gradient
        else {
            panic!("expected a radial gradient, got {fill:?}");
        };
        assert!(
            (c + focus).distance(focal) < 3.,
            "focal point at {}",
            c + focus
        );
        assert!(sse < 1., "sse {sse}");
    }
}

#[cfg(test)]
mod padded_tests {
    use super::tests::costs;
    use super::*;

    /// A disc lit from off its center by a gradient that reaches its end color at six tenths
    /// of the radius and holds it beyond, as a padded radial gradient paints.
    #[test]
    fn padded_radial_recovers_its_end() {
        let (center, limit) = (Vec2::new(40., 36.), 60.);
        let (from, to) = (Vec3::new(0.9, 1.0, 0.56), Vec3::new(0.29, 0.08, 0.55));
        let samples: Vec<(Vec2, Vec3)> = (0..128)
            .flat_map(|y| (0..128).map(move |x| Vec2::new(x as f32 + 0.5, y as f32 + 0.5)))
            .filter(|p| p.distance(Vec2::new(64., 64.)) < 60.)
            .map(|p| (p, from.lerp(to, (p.distance(center) / limit).min(1.))))
            .collect();
        let (fill, sse) = fit(&samples, &[], Vec3::splat(0.5), None, costs(1.), &[]);
        let Some(Gradient::Radial {
            center: found,
            radius,
            ..
        }) = fill.gradient
        else {
            panic!("expected a radial gradient, got {fill:?}");
        };
        assert!(found.distance(center) < 1., "center {found}");
        assert!((radius - limit).abs() < 2., "radius {radius}");
        assert!(sse < 1., "sse {sse}");
    }
}

#[cfg(test)]
mod ellipse_tests {
    use super::tests::costs;
    use super::*;

    /// A tall oval lit from its middle, as `b007-kiln-0401`'s arches: the fit stretches the
    /// circle into an upright ellipse and draws the oval within a few levels.
    #[test]
    fn a_tall_oval_is_an_ellipse() {
        let (center, axes) = (Vec2::new(64., 64.), Vec2::new(30., 54.));
        let (from, to) = (Vec3::new(1., 0.8, 0.3), Vec3::new(0.48, 0.17, 0.07));
        let samples: Vec<(Vec2, Vec3)> = (0..128)
            .flat_map(|y| (0..128).map(move |x| Vec2::new(x as f32 + 0.5, y as f32 + 0.5)))
            .map(|p| (p, ((p - center) / axes).length()))
            .filter(|(_, r)| *r < 1.)
            .map(|(p, r)| (p, from.lerp(to, r)))
            .collect();
        let (fill, _) = fit(&samples, &[], Vec3::splat(0.5), None, costs(1.), &[]);
        let Some(Gradient::Radial { transform, .. }) = fill.gradient else {
            panic!("expected a radial gradient, got {fill:?}");
        };
        assert_ne!(transform, glam::Mat2::IDENTITY);
        for (p, c) in &samples {
            assert!(
                fill.at(*p).distance(*c) < 6. / 255.,
                "{p} {c} {}",
                fill.at(*p)
            );
        }
    }
}
