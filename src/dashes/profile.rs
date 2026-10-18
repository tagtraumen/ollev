//! Dash patterns read off the image: the coverage profile along a centerline, its period seeded
//! by the profile's autocorrelation, the dashes of one period from the profile folded there, and
//! period and phases by least squares through every dash seen along the profile.

use glam::Vec3;

use crate::translucency::NOISE;

use super::*;

/// Arc length between profile samples: a quarter pixel, so a dash's ends and the period are
/// read to a fraction of the anti-aliasing step.
const PROFILE_STEP: f32 = 0.25;
/// Samples averaged into one for the autocorrelation's seed, a pixel's worth.
const DECIMATE: usize = 4;
/// The shortest period read: a dash and a gap of two anti-aliasing steps each, the narrowest
/// pattern whose dashes and gaps both reach their full contrast.
const MIN_PERIOD: f32 = 4. * crate::edges::SIGMA as f32;
/// Least contrast between a pattern's dashes and gaps, in units of the noise, as linework asks
/// of a line against its ground.
const CONTRAST: f32 = 3.;

/// A pattern read off a profile: its period, and the painted runs of one period (where each
/// starts, as arc length modulo the period from the centerline's start, and its length).
pub(super) struct Reading {
    pub(super) period: f32,
    pub(super) runs: Vec<(f32, f32)>,
    /// The painted runs seen along the profile, as arc length spans.
    pub(super) seen: Vec<(f32, f32)>,
}

/// Points along the centerline every [`PROFILE_STEP`] of arc length, from its start to its end.
fn walk(stroke: &Stroke) -> Vec<Vec2> {
    let mut out = vec![stroke.centerline[0][0]];
    let (mut walked, mut next) = (0f32, PROFILE_STEP);
    for c in &stroke.centerline {
        let length = (c[1] - c[0]).length() + (c[2] - c[1]).length() + (c[3] - c[2]).length();
        let steps = ((length / PROFILE_STEP * 2.).ceil() as usize).clamp(1, 1 << 16);
        let mut from = c[0];
        for i in 1..=steps {
            let to = eval(c, i as f32 / steps as f32);
            let d = from.distance(to);
            while d > 0. && walked + d >= next {
                out.push(from.lerp(to, (next - walked) / d));
                next += PROFILE_STEP;
            }
            walked += d;
            from = to;
        }
    }
    out
}

/// `pixels` (a `width` wide image) at the point `p`, bilinear between pixel centers.
fn bilinear(pixels: &[Vec3], width: usize, height: usize, p: Vec2) -> Vec3 {
    let x = (p.x - 0.5).clamp(0., (width - 1) as f32);
    let y = (p.y - 0.5).clamp(0., (height - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |x: usize, y: usize| pixels[y * width + x];
    at(x0, y0)
        .lerp(at(x1, y0), fx)
        .lerp(at(x0, y1).lerp(at(x1, y1), fx), fy)
}

/// How far the image along the centerline goes from what lies below toward `colour`, per
/// [`PROFILE_STEP`] of arc length (1 where it is the colour, 0 where it is the ground), with the
/// noise in those units (the noise over the median contrast of colour and ground).
pub(super) fn profile(
    ctx: &Context,
    place: &crate::refine::Place,
    stroke: &Stroke,
    colour: Vec3,
) -> (Vec<f32>, f32) {
    let w = place.window;
    let origin = w.origin();
    let mut contrasts = Vec::new();
    let values = walk(stroke)
        .into_iter()
        .map(|p| {
            let t = bilinear(&ctx.target, ctx.width, ctx.height, p);
            let g = bilinear(&place.below, w.width, w.height, p - origin);
            let d = colour - g;
            let dd = d.length_squared();
            contrasts.push(dd.sqrt());
            if dd < 1e-8 { 0. } else { (t - g).dot(d) / dd }
        })
        .collect();
    contrasts.sort_by(f32::total_cmp);
    let contrast = contrasts.get(contrasts.len() / 2).copied().unwrap_or(0.);
    (values, NOISE / contrast.max(1e-6))
}

/// The profile folded at `period`: its mean over each of the period's bins.
fn fold(x: &[f32], period: f32) -> Vec<f32> {
    let bins = ((period / PROFILE_STEP).round() as usize).max(2);
    let width = period as f64 / bins as f64;
    let bin = |i: usize| {
        (((i as f64 * PROFILE_STEP as f64).rem_euclid(period as f64) / width) as usize)
            .min(bins - 1)
    };
    let mut sums = vec![0f64; bins];
    let mut counts = vec![0usize; bins];
    for (i, &v) in x.iter().enumerate() {
        let b = bin(i);
        sums[b] += v as f64;
        counts[b] += 1;
    }
    sums.iter()
        .zip(&counts)
        .map(|(s, &c)| if c > 0 { (s / c as f64) as f32 } else { 0. })
        .collect()
}

/// The lags, in pixels, of the profile's autocorrelation peaks that may be its period: the
/// highest peak among those overlapping the profile by a shortest period, and every lower peak it
/// is a whole multiple of (a pattern repeats at its period's multiples as well), read on the
/// profile averaged to a pixel per sample.
fn seeds(x: &[f32], closed: bool) -> Vec<f32> {
    let coarse: Vec<f32> = x
        .chunks(DECIMATE)
        .map(|c| c.iter().sum::<f32>() / c.len() as f32)
        .collect();
    let n = coarse.len();
    let mean = coarse.iter().sum::<f32>() / n.max(1) as f32;
    let y: Vec<f32> = coarse.iter().map(|v| v - mean).collect();
    let lo = (MIN_PERIOD.floor() as usize).max(1);
    let hi = if closed {
        n / MIN_DASHES
    } else {
        n.saturating_sub(lo)
    };
    if hi < lo + 2 {
        return Vec::new();
    }
    let r: Vec<f32> = (lo - 1..=hi + 1)
        .map(|l| {
            let count = if closed { n } else { n.saturating_sub(l) };
            // over the whole length, so a lag seen on a short overlap counts for less
            let s: f32 = (0..count).map(|i| y[i] * y[(i + l) % n]).sum();
            s / n as f32
        })
        .collect();
    // r[k] is the lag lo - 1 + k; a peak's lag refined by the parabola through its neighbours
    let peaks: Vec<(f32, f32)> = (1..r.len() - 1)
        .filter(|&k| r[k] > 0. && r[k] > r[k - 1] && r[k] >= r[k + 1])
        .map(|k| {
            let (a, b, c) = (r[k - 1], r[k], r[k + 1]);
            let bend = a - 2. * b + c;
            let shift = if bend < 0. {
                (0.5 * (a - c) / bend).clamp(-0.5, 0.5)
            } else {
                0.
            };
            ((lo - 1 + k) as f32 + shift, b)
        })
        .collect();
    let Some(&(top, _)) = peaks.iter().max_by(|a, b| a.1.total_cmp(&b.1)) else {
        return Vec::new();
    };
    peaks
        .iter()
        .map(|&(lag, _)| lag)
        .filter(|&lag| {
            let k = (top / lag).round();
            lag <= top && (top - k * lag).abs() <= 1.
        })
        .collect()
}

/// The patterns along a profile of `x` (see [`profile`], `noise` in its units) at each period
/// [`seeds`] proposes, where dashes and gaps differ by [`CONTRAST`] noise. The profile folded at
/// the seed gives the dashes of one period; each dash seen along the profile is matched to one
/// of those, and the period and the dashes' phases are the least squares line through the
/// seen centres, which a drift of a fraction of a pixel per period does not escape.
pub(super) fn read(x: &[f32], noise: f32, closed: bool, quantum: f32) -> Vec<Reading> {
    let n = x.len();
    let total = (n - 1) as f32 * PROFILE_STEP;
    // a closed profile's last sample is its first again
    let samples = if closed { &x[..n - 1] } else { x };
    let mut readings = Vec::new();
    for rough in seeds(x, closed) {
        let means = fold(samples, rough);
        let hi = means.iter().copied().fold(f32::MIN, f32::max);
        let lo = means.iter().copied().fold(f32::MAX, f32::min);
        if hi - lo < CONTRAST * noise {
            continue;
        }
        let mid = (hi + lo) / 2.;
        let width = rough / means.len() as f32;
        let Some(slots) = crossings(&means, mid, width, true) else {
            continue;
        };
        // each slot's centre in the period, a bin's mean sitting at its middle
        let phases: Vec<f32> = slots
            .iter()
            .map(|&(a, l)| a + width / 2. + l / 2.)
            .collect();
        let seen = crossings(samples, mid, PROFILE_STEP, closed).unwrap_or_default();
        // (slot, period index, centre, length) of each seen dash whole within the profile
        let mut matched: Vec<(usize, f32, f32, f32)> = Vec::new();
        for &(a, l) in &seen {
            if !closed && (a <= 0. || a + l >= total) {
                continue;
            }
            let c = a + l / 2.;
            let best = phases
                .iter()
                .enumerate()
                .map(|(s, &p)| {
                    let k = ((c - p) / rough).round();
                    (s, k, (c - p - k * rough).abs())
                })
                .min_by(|a, b| a.2.total_cmp(&b.2));
            if let Some((s, k, miss)) = best
                && miss <= rough / 4.
            {
                matched.push((s, k, c, l));
            }
        }
        let Some(period) = common_period(&matched, phases.len()) else {
            continue;
        };
        let period = if closed {
            total / (total / period).round().max(MIN_DASHES as f32)
        } else {
            period
        };
        if period < MIN_PERIOD {
            continue;
        }
        // each slot's dash: its centre's mean phase at the period, its median length; no
        // reading where the period leaves a dash seen more than `quantum` from its place
        let mut runs: Vec<(f32, f32)> = Vec::new();
        let mut periodic = true;
        for s in 0..phases.len() {
            let mine: Vec<&(usize, f32, f32, f32)> = matched.iter().filter(|m| m.0 == s).collect();
            // two dashes fix a period and phase; two more must bear it out (lacebobbins: three
            // scattered dots on a dome read as a dotted line)
            periodic &= mine.len() >= 2 * MIN_DASHES;
            if mine.is_empty() {
                continue;
            }
            let phase = mine.iter().map(|m| m.2 - m.1 * period).sum::<f32>() / mine.len() as f32;
            periodic &= mine
                .iter()
                .all(|m| (m.2 - m.1 * period - phase).abs() <= quantum);
            let mut lengths: Vec<f32> = mine.iter().map(|m| m.3).collect();
            lengths.sort_by(f32::total_cmp);
            let l = lengths[lengths.len() / 2];
            runs.push(((phase - l / 2.).rem_euclid(period), l));
        }
        if periodic && !runs.is_empty() {
            let seen = seen.iter().map(|&(a, l)| (a, a + l)).collect();
            readings.push(Reading { period, runs, seen });
        }
    }
    readings
}

/// The period of the least squares fit `centre = phase[slot] + index * period` through the
/// matched dashes (slot, index, centre, length), each slot with its own phase; none where no
/// slot has dashes at two indices.
fn common_period(matched: &[(usize, f32, f32, f32)], slots: usize) -> Option<f32> {
    let (mut sxy, mut sxx) = (0f64, 0f64);
    for s in 0..slots {
        let mine: Vec<(f64, f64)> = matched
            .iter()
            .filter(|m| m.0 == s)
            .map(|m| (m.1 as f64, m.2 as f64))
            .collect();
        if mine.len() < 2 {
            continue;
        }
        let kx = mine.iter().map(|m| m.0).sum::<f64>() / mine.len() as f64;
        let cy = mine.iter().map(|m| m.1).sum::<f64>() / mine.len() as f64;
        for (k, c) in mine {
            sxy += (k - kx) * (c - cy);
            sxx += (k - kx) * (k - kx);
        }
    }
    (sxx > 0.).then(|| (sxy / sxx) as f32)
}

/// The runs (start, length) where `values`, samples `width` apart, lie above `mid`, their
/// ends where the line between neighbouring samples crosses it; around the end and back to
/// the start with `circular`. None where the values never cross.
fn crossings(values: &[f32], mid: f32, width: f32, circular: bool) -> Option<Vec<(f32, f32)>> {
    let n = values.len();
    let at = |i: usize| values[i % n];
    let cross = |i: usize| {
        let (a, b) = (at(i), at(i + 1));
        (i as f32 + (mid - a) / (b - a)) * width
    };
    let last = if circular { n } else { n - 1 };
    let mut ups = Vec::new();
    let mut downs = Vec::new();
    for i in 0..last {
        let (a, b) = (at(i) > mid, at(i + 1) > mid);
        if !a && b {
            ups.push(cross(i));
        } else if a && !b {
            downs.push(cross(i));
        }
    }
    if ups.is_empty() && downs.is_empty() {
        return None;
    }
    let span = n as f32 * width;
    let mut runs = Vec::new();
    if circular {
        for &u in &ups {
            let d = downs
                .iter()
                .map(|&d| if d < u { d + span } else { d })
                .fold(f32::MAX, f32::min);
            runs.push((u, d - u));
        }
    } else {
        // an open profile on at its start or end has runs cut by its ends
        let starts_on = at(0) > mid;
        let mut ups = ups;
        if starts_on {
            ups.insert(0, 0.);
        }
        for (k, &u) in ups.iter().enumerate() {
            let d = downs
                .iter()
                .copied()
                .find(|&d| d >= u)
                .unwrap_or((n - 1) as f32 * width);
            if k + 1 < ups.len() && ups[k + 1] < d {
                continue;
            }
            runs.push((u, d - u));
        }
    }
    Some(runs)
}

/// Whether the picture along a dashed stroke shows its gaps as the ground beside it: every gap
/// long enough to reach it, and the gaps nearer the ground on one side than to `colour`, so a
/// line in pieces (niabot's shaded rails, whose gaps are the line) has none.
pub(super) fn gaps_seen(ctx: &Context, stroke: &Stroke, colour: Vec3) -> bool {
    // a gap reaches the ground's full contrast over two anti-aliasing steps (see MIN_PERIOD);
    // one the pattern closed below that is a solid line in pieces (niabot 1052's hair)
    let closed = stroke.dashes.as_ref().is_none_or(|d| {
        d.array
            .iter()
            .skip(1)
            .step_by(2)
            .any(|&g| g < MIN_PERIOD / 2.)
    });
    if closed {
        return false;
    }
    let ends = stroke.arc_lengths();
    let reach = if stroke.cap == Cap::Butt {
        0.
    } else {
        stroke.width / 2.
    };
    let spans = stroke.dash_spans(&ends);
    let on = |i: usize| {
        let s = i as f32 * PROFILE_STEP;
        spans.iter().any(|&(a, b)| a - reach <= s && s <= b + reach)
    };
    let points = walk(stroke);
    let (Some(first), Some(last)) = (
        (0..points.len()).position(on),
        (0..points.len()).rposition(on),
    ) else {
        return false;
    };
    // the gaps between the first dash and the last, and the ground beside them past the
    // stroke's anti-aliased edge on either side
    let at = |p: Vec2| bilinear(&ctx.target, ctx.width, ctx.height, p);
    let beside = stroke.width / 2. + 2. * crate::edges::SIGMA as f32;
    let (mut gaps, mut sides) = (Vec::new(), [Vec::new(), Vec::new()]);
    for i in (first..=last).filter(|&i| !on(i)) {
        let along = (points[(i + 1).min(points.len() - 1)] - points[i - 1]).normalize_or_zero();
        let normal = along.perp() * beside;
        gaps.push(at(points[i]));
        sides[0].push(at(points[i] + normal));
        sides[1].push(at(points[i] - normal));
    }
    if gaps.is_empty() {
        return false;
    }
    let median = |v: &mut Vec<Vec3>| {
        let mut channel = |k: usize| {
            v.sort_by(|a, b| a[k].total_cmp(&b[k]));
            v[v.len() / 2][k]
        };
        Vec3::new(channel(0), channel(1), channel(2))
    };
    // how far a colour lies along the line from a ground to the paint
    let toward = |t: Vec3, ground: Vec3| {
        let d = colour - ground;
        (t - ground).dot(d) / d.length_squared()
    };
    let shown = median(&mut gaps);
    let Some(ground) = sides
        .iter_mut()
        .map(median)
        .filter(|g| g.distance(colour) >= CONTRAST * NOISE)
        .min_by(|a, b| toward(shown, *a).total_cmp(&toward(shown, *b)))
    else {
        return false;
    };
    if toward(shown, ground) >= 0.5 {
        return false;
    }
    // and each gap's mean nearer that ground
    let mut run = (0f32, 0usize);
    for (i, &p) in points.iter().enumerate().take(last + 1).skip(first) {
        if !on(i) {
            run = (run.0 + toward(at(p), ground), run.1 + 1);
        } else {
            let (sum, n) = std::mem::take(&mut run);
            if n > 0 && sum / n as f32 >= 0.5 {
                return false;
            }
        }
    }
    true
}

/// The dash pattern drawing `reading`'s runs with `reach` of paint past each dash's ends (a
/// round or square cap's), each dash centred on its run and at least `least` long.
pub(super) fn dashes_of(reading: &Reading, reach: f32, least: f32) -> Dashes {
    let p = reading.period;
    let mut dashes: Vec<(f32, f32)> = reading
        .runs
        .iter()
        .map(|&(a, l)| {
            let d = (l - 2. * reach).max(least);
            ((a + (l - d) / 2.).rem_euclid(p), d)
        })
        .collect();
    dashes.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut array = Vec::new();
    for k in 0..dashes.len() {
        let (a, d) = dashes[k];
        let next = if k + 1 < dashes.len() {
            dashes[k + 1].0
        } else {
            dashes[0].0 + p
        };
        array.push(d);
        array.push((next - a - d).max(0.));
    }
    Dashes {
        array,
        offset: (-dashes[0].0).rem_euclid(p),
    }
}

/// The centerline spans of the dashes `reading` saw along the profile, each painted run
/// shrunk by `reach` at both ends as [`dashes_of`] does, for checking a pattern drawn.
pub(super) fn seen_spans(reading: &Reading, reach: f32, least: f32) -> Vec<(f32, f32)> {
    reading
        .seen
        .iter()
        .map(|&(a, b)| {
            let (c, d) = ((a + b) / 2., (b - a - 2. * reach).max(least));
            (c - d / 2., c + d / 2.)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4 on 12 off pattern starting 3 px in, 27 periods long, as binary's dotted rows: the
    /// period read to a hundredth of a pixel, so the last dash stays in step, and the phase.
    #[test]
    fn reads_period_and_phase() {
        let (period, on, start) = (16f32, 4f32, 3f32);
        let x: Vec<f32> = (0..(27. * period / PROFILE_STEP) as usize)
            .map(|i| {
                let s = (i as f32 * PROFILE_STEP - start).rem_euclid(period);
                if s < on { 1. } else { 0. }
            })
            .collect();
        let reading = read(&x, NOISE, false, WIDTH_QUANTUM)
            .pop()
            .expect("a reading");
        assert!((reading.period - period).abs() < 0.01, "{}", reading.period);
        let dashes = dashes_of(&reading, 0., DOT_LENGTH);
        let first = (-dashes.offset).rem_euclid(period);
        assert!((first - start).abs() <= PROFILE_STEP, "{first}");
        assert!(
            (dashes.array[0] - on).abs() <= PROFILE_STEP,
            "{:?}",
            dashes.array
        );
    }

    /// A row of grey dashes on a light ground, the gaps showing the ground or a lighter grey
    /// (a shaded line in pieces): only the first shows its gaps.
    #[test]
    fn gaps_show_the_ground_beside() {
        let (w, h) = (64, 24);
        let stroke = Stroke {
            centerline: vec![[
                Vec2::new(4., 12.),
                Vec2::new(22., 12.),
                Vec2::new(40., 12.),
                Vec2::new(58., 12.),
            ]],
            closed: false,
            width: 4.,
            cap: Cap::Butt,
            join: crate::stroke::Join::Miter,
            corners: vec![false],
            dashes: Some(crate::stroke::Dashes {
                array: vec![6., 4.],
                offset: 0.,
            }),
        };
        let seen = |gap: u8| {
            let mut rgb = vec![230u8; 3 * w * h];
            for y in 10..14 {
                for x in 4..58 {
                    let v = if (x - 4) % 10 < 6 { 80 } else { gap };
                    rgb[3 * (y * w + x)..3 * (y * w + x) + 3].fill(v);
                }
            }
            let ctx = Context::new(&rgb, w, h, 1.);
            gaps_seen(&ctx, &stroke, Vec3::splat(80. / 255.))
        };
        assert!(seen(230));
        assert!(!seen(110));
    }
}
