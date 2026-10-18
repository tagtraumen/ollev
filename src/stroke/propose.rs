//! A stroke proposed from a skeleton chain: the edges either side, width, caps and corners.

use super::*;

/// The band's edges as the image shows them: per window pixel, how much of the shape's fill
/// the image holds over what lies below, `None` where hidden, free or without contrast. The
/// anti-aliasing locates an edge within a fraction of a pixel, unlike the segmentation's mask.
pub(crate) struct Edges {
    pub(super) window: Window,
    pub(super) coverage: Vec<Option<f32>>,
}

impl Edges {
    pub(crate) fn new(ctx: &Context, place: &Place, fill: Fill) -> Self {
        let window = place.window;
        let coverage = window
            .pixels(ctx.width)
            .enumerate()
            .map(|(k, i)| {
                let hidden = ctx.is_free(i)
                    || place
                        .above
                        .as_ref()
                        .is_some_and(|a| a.transmittance[k].max_element() < 0.5);
                if hidden {
                    return None;
                }
                let (x, y) = (i % ctx.width, i / ctx.width);
                let own = fill.at_pixel(x, y);
                let contrast = own - place.below[k];
                let span = contrast.length_squared();
                (span > 1e-4).then(|| (ctx.target[i] - place.below[k]).dot(contrast) / span)
            })
            .collect();
        Self { window, coverage }
    }

    /// The coverage at an image point, interpolated between the four pixels around it,
    /// `None` where any of them is unknown.
    pub(super) fn at(&self, p: Vec2) -> Option<f32> {
        let (fx, fy) = (
            p.x - 0.5 - self.window.x as f32,
            p.y - 0.5 - self.window.y as f32,
        );
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let (w, h) = (self.window.width as i64, self.window.height as i64);
        let get = |x: i64, y: i64| -> Option<f32> {
            (x >= 0 && y >= 0 && x < w && y < h)
                .then(|| self.coverage[(y * w + x) as usize])
                .flatten()
        };
        let (x0, y0) = (x0 as i64, y0 as i64);
        let (c00, c10, c01, c11) = (
            get(x0, y0)?,
            get(x0 + 1, y0)?,
            get(x0, y0 + 1)?,
            get(x0 + 1, y0 + 1)?,
        );
        Some((c00 * (1. - tx) + c10 * tx) * (1. - ty) + (c01 * (1. - tx) + c11 * tx) * ty)
    }

    /// What a ray from `origin` (an image position) along `direction` meets within `reach`:
    /// the band's edge, where coverage falls through a half with its gradient along the ray
    /// within the kink; another edge (a cap's, join's or branch's) or none; or unknown pixels.
    pub(super) fn edge(&self, origin: Vec2, direction: Vec2, reach: f32) -> Edge {
        const STEP: f32 = 0.25;
        let Some(mut previous) = self.at(origin) else {
            return Edge::Hidden;
        };
        if previous < 0.5 {
            return Edge::Other;
        }
        let mut s = 0f32;
        while s + STEP <= reach {
            let next = s + STEP;
            let Some(c) = self.at(origin + direction * next) else {
                return Edge::Hidden;
            };
            if c < 0.5 {
                let t = (previous - 0.5) / (previous - c).max(1e-6);
                let distance = s + STEP * t;
                let q = origin + direction * distance;
                let (Some(gx1), Some(gx0), Some(gy1), Some(gy0)) = (
                    self.at(q + Vec2::X * 0.5),
                    self.at(q - Vec2::X * 0.5),
                    self.at(q + Vec2::Y * 0.5),
                    self.at(q - Vec2::Y * 0.5),
                ) else {
                    return Edge::Hidden;
                };
                let gradient = Vec2::new(gx1 - gx0, gy1 - gy0).normalize_or_zero();
                return if gradient.dot(direction).abs() >= KINK.cos() {
                    Edge::At(distance)
                } else {
                    Edge::Other
                };
            }
            previous = c;
            s = next;
        }
        Edge::Other
    }
}

/// What a ray across the band meets, see [`Edges::edge`].
pub(super) enum Edge {
    At(f32),
    Hidden,
    Other,
}

/// A chain pixel's place between the band's edges, read off the image.
#[derive(Clone, Copy)]
pub(super) struct Crossing {
    /// The midpoint between the two edges (the pixel itself where neither shows).
    pub(super) mid: Vec2,
    /// The distance between the edges, where both show.
    pub(super) width: Option<f32>,
}

/// Each chain pixel moved along the chain's normal to the midpoint between the band's two
/// edges, which count only where both run along the chain and stand the width apart within
/// its slack; with one edge hidden, half a width from the other. Otherwise the pixel stays.
pub(super) fn cross(
    points: &[Vec2],
    closed: bool,
    width: f32,
    edge: &dyn Fn(Vec2, Vec2) -> Edge,
) -> Vec<Crossing> {
    let n = points.len();
    // The heading over half a width either way, at least the outline's corner window, so a
    // thin band's ray does not follow the pixel staircase
    let k = ((width / 2.).ceil() as usize).max(crate::outline::FIT.corner_window);
    let pass = |along: &[Vec2]| -> Vec<Crossing> {
        (0..n)
            .map(|i| {
                let (a, b) = if closed {
                    ((i + n - k) % n, (i + k) % n)
                } else {
                    (i.saturating_sub(k), (i + k).min(n - 1))
                };
                let heading = (along[b] - along[a]).normalize_or_zero();
                if heading == Vec2::ZERO {
                    return Crossing {
                        mid: points[i],
                        width: None,
                    };
                }
                side_crossing(points[i], heading, width, edge)
            })
            .collect()
    };
    // Twice: the headings of the second pass come from the first pass's midpoints, off the
    // staircase
    let first = pass(points);
    let mids: Vec<Vec2> = first.iter().map(|c| c.mid).collect();
    pass(&mids)
}

/// The crossing at `point` across the band heading `heading`, see [`cross`].
pub(super) fn side_crossing(
    point: Vec2,
    heading: Vec2,
    width: f32,
    edge: &dyn Fn(Vec2, Vec2) -> Edge,
) -> Crossing {
    let normal = heading.perp();
    let slack = (WIDTH_SLACK * width).max(WIDTH_QUANTUM);
    match (edge(point, normal), edge(point, -normal)) {
        (Edge::At(forward), Edge::At(backward)) if (forward + backward - width).abs() > slack => {
            Crossing {
                mid: point,
                width: None,
            }
        }
        (Edge::At(forward), Edge::At(backward)) => Crossing {
            mid: point + normal * (forward - backward) / 2.,
            width: Some(forward + backward),
        },
        // One edge known, but farther than the band's own side can lie: not the band's side
        // (as a ray running along a crossing line of a network)
        (Edge::At(forward), Edge::Hidden) if forward <= width + slack => Crossing {
            mid: point + normal * (forward - width / 2.),
            width: None,
        },
        (Edge::Hidden, Edge::At(backward)) if backward <= width + slack => Crossing {
            mid: point - normal * (backward - width / 2.),
            width: None,
        },
        _ => Crossing {
            mid: point,
            width: None,
        },
    }
}

/// Where the band's sides stop at an open chain's end: from the end along its heading (or the
/// end piece's `line`), outward while both edges still show at the width, or inward until they
/// do, since the thinning's chain may stop short of the end or hook into the cap.
pub(super) fn band_end(
    points: &[Vec2],
    at_start: bool,
    width: f32,
    edge: &dyn Fn(Vec2, Vec2) -> Edge,
    line: Option<(Vec2, Vec2)>,
) -> Vec2 {
    let n = points.len();
    let k = ((width / 2.).ceil() as usize).max(2).min(n - 1);
    let (end, inner) = if at_start {
        (points[0], points[k])
    } else {
        (points[n - 1], points[n - 1 - k])
    };
    // Along the end piece's line where it has one, from the end's projection on it, since the
    // chain's last pixels hook toward a cap's corners
    let (end, outward) = match line {
        Some((p, d)) => {
            let along = p + d * (end - p).dot(d);
            (along, d * (end - inner).dot(d).signum())
        }
        None => (end, (end - inner).normalize_or_zero()),
    };
    if outward == Vec2::ZERO {
        return end;
    }
    let sides_at = |s: f32| -> bool {
        side_crossing(end + outward * s, outward, width, edge)
            .width
            .is_some()
    };
    const STEP: f32 = 0.5;
    let limit = width * MITER_LIMIT;
    let mut s = 0f32;
    if sides_at(0.) {
        while s + STEP <= limit && sides_at(s + STEP) {
            s += STEP;
        }
    } else {
        while s - STEP >= -limit && !sides_at(s - STEP) {
            s -= STEP;
        }
        s -= STEP;
    }
    end + outward * s
}

/// A proposed stroke, caps and joins still to choose, with whether each end of its centerline
/// sits at the band's extent (a cap reaching beyond then pulls the end in by its reach) or half
/// a width short of a butt end, as a skeleton's does.
pub(crate) struct Proposal {
    pub(crate) stroke: Stroke,
    /// What the centerline is built from, for the simplification candidates.
    pub(crate) spec: Centerline,
    pub(crate) ends_at_extent: [bool; 2],
    /// The centerline's ends are where the sides stop running parallel: the end itself for a
    /// butt or round cap, half a width past it for a square cap, whose sides run on.
    pub(crate) from_pairs: bool,
}

/// The first estimate of a skeleton's stroke width, where it could be one: the thinned
/// chain's distance to the mask's edge, the median over the chain's middle half (toward an
/// open chain's ends it falls off), between a pixel and [`MAX_WIDTH`], the chain longer
/// than its caps. Read before any placing, so a shape no stroke could be is not placed.
pub(super) fn first_width(skeleton: &Skeleton) -> Option<f32> {
    let n = skeleton.chain.len();
    let mut middle: Vec<f32> = if skeleton.closed {
        skeleton.widths.clone()
    } else {
        skeleton.widths[n / 4..n - n / 4].to_vec()
    };
    middle.sort_by(f32::total_cmp);
    let width = middle.get(middle.len() / 2).copied()?;
    let margin = if skeleton.closed {
        0
    } else {
        (width / 2.).ceil() as usize
    };
    ((1. ..=MAX_WIDTH).contains(&width) && n > 2 * margin + 2).then_some(width)
}

/// A stroke for one skeleton of a shape, see [`fit_at`].
pub(super) fn propose(skeleton: Skeleton, edges: &Edges) -> Option<Proposal> {
    let n = skeleton.chain.len();
    // Rays in box coordinates (a chain pixel's center at its index) against the image
    let offset = skeleton.top_left.as_vec2() + 0.5;
    let edge = |p: Vec2, d: Vec2| -> Edge { edges.edge(p + offset, d, MAX_WIDTH) };
    // The width estimate sets the crossings' reach and heading window
    let width = first_width(&skeleton)?;
    let chain: Vec<Vec2> = skeleton.chain.iter().map(|p| p.as_vec2()).collect();
    let crossings = cross(&chain, skeleton.closed, width, &edge);
    let margin = if skeleton.closed {
        0
    } else {
        (width / 2.).ceil() as usize
    };
    let inner: Vec<f32> = (margin..n - margin)
        .filter_map(|i| crossings[i].width)
        .collect();
    let within = inner
        .iter()
        .filter(|&&v| (v - width).abs() <= (WIDTH_SLACK * width).max(WIDTH_QUANTUM))
        .count();
    if inner.is_empty() || (within as f32) < WIDTH_SHARE * inner.len() as f32 {
        return None;
    }
    let points: Vec<Vec2> = crossings.iter().map(|c| c.mid).collect();
    let n = points.len();
    let mut band_ends: [Option<Vec2>; 2] = [None, None];
    let reach = (width / 2.).ceil() as usize;
    // A corner turns by a kink over half a width each side, or over three times that by half a
    // kink more than its neighbours (a polyline's mild corner); of candidates within the miter
    // reach or joined by turning, the sharpest, so a hairpin's tip gives one
    let span = reach.max(crate::outline::FIT.corner_window);
    // None where a point repeats one a window away (no direction to turn from)
    let turn_over = |i: usize, w: usize| -> Option<f32> {
        let (a, b) = if skeleton.closed {
            ((i + n - w) % n, (i + w) % n)
        } else if i >= w && i + w < n {
            (i - w, i + w)
        } else {
            return Some(0.);
        };
        crate::tracer::angle::turn(
            (points[i] - points[a]).normalize_or_zero(),
            (points[b] - points[i]).normalize_or_zero(),
        )
        .map(f32::abs)
    };
    // a point without a turn is no corner by its own turn
    let turns: Vec<f32> = (0..n).map(|i| turn_over(i, span).unwrap_or(0.)).collect();
    let wide = 3 * span;
    let wide_turns: Vec<Option<f32>> = (0..n).map(|i| turn_over(i, wide)).collect();
    // None where one of the three has no turn: no excess to tell
    let excess = |i: usize| -> Option<f32> {
        let (a, b) = if skeleton.closed {
            ((i + n - 2 * wide) % n, (i + 2 * wide) % n)
        } else if i >= 2 * wide && i + 2 * wide < n {
            (i - 2 * wide, i + 2 * wide)
        } else {
            return Some(0.);
        };
        Some(wide_turns[i]? - (wide_turns[a]? + wide_turns[b]?) / 2.)
    };
    let score: Vec<f32> = (0..n)
        .map(|i| match excess(i) {
            _ if turns[i] > KINK => turns[i],
            Some(e) if e > KINK / 2. => wide_turns[i].unwrap_or(0.),
            _ => 0.,
        })
        .collect();
    let cluster = ((width * MITER_LIMIT / 2.).ceil() as usize).max(span);
    // Candidates joined by a run of turning (the wide turn above half a kink all the way
    // between them) belong to one corner too: around a hairpin's tip the chain turns over
    // a stretch of its legs' overlap, far longer than a corner's reach
    let joined = |a: usize, b: usize| -> bool {
        let (lo, hi) = (a.min(b), a.max(b));
        let d = if skeleton.closed {
            (hi - lo).min(n - (hi - lo))
        } else {
            hi - lo
        };
        if d <= cluster {
            return true;
        }
        if skeleton.closed && hi - lo != d {
            (hi..hi + d).all(|k| wide_turns[k % n].is_some_and(|t| t > KINK / 2.))
        } else {
            (lo..=hi).all(|k| wide_turns[k].is_some_and(|t| t > KINK / 2.))
        }
    };
    let mut by_score: Vec<usize> = (0..n).filter(|&i| score[i] > 0.).collect();
    by_score.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
    let mut corner_at: Vec<usize> = Vec::new();
    for i in by_score {
        if corner_at.iter().all(|&c| !joined(c, i)) {
            corner_at.push(i);
        }
    }
    corner_at.sort_unstable();
    // A corner within half a width of an open chain's end is the thinning's hook there, not
    // a corner of the stroke
    corner_at.retain(|&i| skeleton.closed || (i > reach && i + reach < n));
    // Between corners, a piece whose middle lies on a line within the tolerance becomes that
    // line, and a corner between two lines goes where they meet, which the medial axis never
    // reaches; curved pieces keep their fitted cubics. Pieces are index ranges between corners
    let mut bounds: Vec<usize> = Vec::new();
    if !skeleton.closed {
        bounds.push(0);
    }
    bounds.extend(corner_at.iter().copied());
    if !skeleton.closed {
        bounds.push(points.len() - 1);
    }
    bounds.sort_unstable();
    bounds.dedup();
    let line_of = |from: usize, to: usize| -> Option<(Vec2, Vec2)> {
        // The middle half of the piece: at an acute corner the skeleton bends off over
        // half a width over the sine of half the angle, well beyond half a width
        let span = to.saturating_sub(from);
        let (lo, hi) = (from + span / 4, to.saturating_sub(span / 4));
        if hi <= lo + 2 || span < 2 * reach {
            return None;
        }
        let pca = |set: &[Vec2]| -> Option<(Vec2, Vec2)> {
            let mean = set.iter().sum::<Vec2>() / set.len() as f32;
            let (mut xx, mut xy, mut yy) = (0f32, 0f32, 0f32);
            for p in set {
                let d = *p - mean;
                xx += d.x * d.x;
                xy += d.x * d.y;
                yy += d.y * d.y;
            }
            if (xx - yy).abs() < 1e-6 && xy.abs() < 1e-6 {
                return None;
            }
            let top = 0.5 * (xx + yy) + (0.25 * (xx - yy).powi(2) + xy * xy).sqrt();
            let axis = if xy.abs() > 1e-6 {
                Vec2::new(xy, top - xx).normalize()
            } else if xx >= yy {
                Vec2::X
            } else {
                Vec2::Y
            };
            Some((mean, axis))
        };
        // The line through the middle half, then through everything of the piece within the
        // tolerance of it, which leaves out the bends toward the corners
        let first = pca(&points[lo..=hi])?;
        let all = &points[from..=to];
        let inliers: Vec<Vec2> = all
            .iter()
            .copied()
            .filter(|p| (*p - first.0).perp_dot(first.1).abs() <= CENTERLINE_TOLERANCE)
            .collect();
        if inliers.len() * 2 < all.len() {
            return None;
        }
        pca(&inliers)
    };
    let mut lines: Vec<Option<(Vec2, Vec2)>> = if skeleton.closed || bounds.len() < 2 {
        Vec::new()
    } else {
        bounds.windows(2).map(|w| line_of(w[0], w[1])).collect()
    };
    // An end piece shorter than two spans that is no line is the thinning's hook at a butt
    // end: it goes, and the piece inside takes the end
    if !skeleton.closed {
        while bounds.len() > 2 && lines[0].is_none() && bounds[1] - bounds[0] < 2 * span {
            bounds.remove(1);
            lines.remove(0);
            lines[0] = line_of(bounds[0], bounds[1]);
        }
        while bounds.len() > 2
            && lines[lines.len() - 1].is_none()
            && bounds[bounds.len() - 1] - bounds[bounds.len() - 2] < 2 * span
        {
            let last = bounds.len() - 1;
            bounds.remove(last - 1);
            lines.pop();
            let k = lines.len() - 1;
            lines[k] = line_of(bounds[k], bounds[k + 1]);
        }
    }
    if !skeleton.closed {
        band_ends = [
            Some(band_end(
                &points,
                true,
                width,
                &edge,
                lines.first().copied().flatten(),
            )),
            Some(band_end(
                &points,
                false,
                width,
                &edge,
                lines.last().copied().flatten(),
            )),
        ];
    }
    // The corner points: where two lines meet, else the chain's corner pixel; an end where
    // the band's sides stop
    let meet = |l1: (Vec2, Vec2), l2: (Vec2, Vec2)| -> Option<Vec2> {
        let det = l1.1.perp_dot(l2.1);
        if det.abs() < 1e-4 {
            return None;
        }
        let t = (l2.0 - l1.0).perp_dot(l2.1) / det;
        Some(l1.0 + l1.1 * t)
    };
    let spec = if skeleton.closed {
        Centerline {
            pieces: vec![Stretch {
                samples: points.iter().map(|p| *p + offset).collect(),
                line: false,
                start: End::Free,
                end: End::Free,
            }],
            closed: true,
            corners: corner_at.clone(),
        }
    } else {
        let mut knots: Vec<Vec2> = Vec::with_capacity(bounds.len());
        for (j, &b) in bounds.iter().enumerate() {
            let before = if j > 0 { lines[j - 1] } else { None };
            let after = if j + 1 < bounds.len() { lines[j] } else { None };
            let q = points[b];
            let knot = match (before, after) {
                (Some(l1), Some(l2)) => meet(l1, l2)
                    .filter(|m| {
                        // The medial axis ends half a width over the sine of half the angle
                        // short of the corner, so the corner may lie that far off, plus a
                        // width of slack
                        let along = |l: (Vec2, Vec2), from: usize, to: usize| {
                            l.1 * (points[bounds[to]] - points[bounds[from]])
                                .dot(l.1)
                                .signum()
                        };
                        let (d1, d2) = (along(l1, j - 1, j), along(l2, j, j + 1));
                        // lines without a direction: the widest reach
                        let half_sine = crate::tracer::angle::turn(d1, d2)
                            .map_or(1e-3, |t| (t.abs() / 2.).cos().max(1e-3));
                        let reach = (width / 2. / half_sine + width).max(width * MITER_LIMIT);
                        m.distance(q) <= reach
                    })
                    .unwrap_or(q),
                _ if j == 0 => band_ends[0].unwrap_or(q),
                _ if j + 1 == bounds.len() => band_ends[1].unwrap_or(q),
                _ => q,
            };
            knots.push(knot + offset);
        }
        let m = lines.len();
        let pieces = (0..m)
            .map(|j| {
                let line = lines[j].is_some();
                // A curved piece's end at a corner goes to the knot; at an open end it
                // extends along its own continuation to where the band stops
                let end_at = |k: usize, open: bool| {
                    if line || !open {
                        End::Knot(knots[k])
                    } else {
                        End::Band(knots[k])
                    }
                };
                Stretch {
                    samples: points[bounds[j]..=bounds[j + 1]]
                        .iter()
                        .map(|p| *p + offset)
                        .collect(),
                    line,
                    start: end_at(j, j == 0),
                    end: end_at(j + 1, j + 1 == m),
                }
            })
            .collect();
        Centerline {
            pieces,
            closed: false,
            corners: Vec::new(),
        }
    };
    let (centerline, corners) = spec.build(CENTERLINE_TOLERANCE)?;
    Some(Proposal {
        stroke: Stroke {
            centerline,
            closed: skeleton.closed,
            width,
            cap: Cap::Butt,
            join: Join::Miter,
            corners,
            dashes: None,
        },
        spec,
        ends_at_extent: [false, false],
        from_pairs: true,
    })
}

/// `simpler`, a simplification of the centerline `previous`, with each cubic refitted by least
/// squares to the stretch of `previous` between its ends, keeping its ends and tangents.
pub(super) fn polish(simpler: &[Cubic], previous: &[Cubic], closed: bool) -> Vec<Cubic> {
    const SAMPLES: usize = 16;
    const ITERATIONS: usize = 8;
    let m = previous.len();
    let total = m * SAMPLES;
    let sample = |s: usize| {
        eval(
            &previous[(s / SAMPLES) % m],
            (s % SAMPLES) as f32 / SAMPLES as f32,
        )
    };
    // The sample of `previous` nearest to a point, as a position along it
    let nearest = |q: Vec2| -> usize {
        (0..=total)
            .min_by(|&a, &b| sample(a).distance(q).total_cmp(&sample(b).distance(q)))
            .unwrap_or(0)
    };
    simpler
        .iter()
        .map(|c| {
            let (from, mut to) = (nearest(c[0]), nearest(c[3]));
            if to <= from {
                if closed {
                    to += total;
                } else {
                    return *c;
                }
            }
            let points: Vec<Vec2> = (from..=to).map(sample).collect();
            let t0 = tangent(c, 0.);
            let t1 = -tangent(c, 1.);
            let mut fitted = crate::tracer::fit::fit_cubic(&points, t0, t1, ITERATIONS);
            fitted[0] = c[0];
            fitted[3] = c[3];
            fitted
        })
        .collect()
}

/// `cubics`, a fit of `samples` at some tolerance, with each cubic fitted again by least
/// squares to the samples between its ends, keeping its ends and tangents: a fit at a
/// tolerance stops at the first cubic within it, a poor start for the optimizer.
pub(crate) fn refit(cubics: Vec<Cubic>, samples: &[Vec2]) -> Vec<Cubic> {
    const ITERATIONS: usize = 8;
    let nearest = |q: Vec2| -> usize {
        (0..samples.len())
            .min_by(|&a, &b| samples[a].distance(q).total_cmp(&samples[b].distance(q)))
            .unwrap_or(0)
    };
    let n = samples.len();
    cubics
        .iter()
        .map(|c| {
            let (from, mut to) = (nearest(c[0]), nearest(c[3]));
            if to < from {
                // Around a ring's seam (its last sample repeats its first)
                to += n - 1;
            }
            if to < from + 3 {
                return *c;
            }
            let range: Vec<Vec2> = (from..=to)
                .map(|i| {
                    if i < n {
                        samples[i]
                    } else {
                        samples[i - (n - 1)]
                    }
                })
                .collect();
            let mut fitted =
                crate::tracer::fit::fit_cubic(&range, tangent(c, 0.), -tangent(c, 1.), ITERATIONS);
            fitted[0] = c[0];
            fitted[3] = c[3];
            fitted
        })
        .collect()
}
