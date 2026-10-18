//! Dotted lines: chains of dots as one dotted stroke.

use super::*;

/// A disc of the scene that may be a dot of a dotted stroke.
pub(super) struct Dot {
    pub(super) position: usize,
    pub(super) center: Vec2,
    /// Across the path and along it: a dot drawn as a round-capped dash is the width across
    /// and the dash's length more along.
    pub(super) width: f32,
    pub(super) extent: f32,
    /// The image's colour of a dot read off the picture, which links only to alike ones.
    pub(super) color: Option<glam::Vec3>,
}

/// The dot a shape is, as its center, width and extent (see [`Dot`]): a circle, ellipse or
/// stadium (as a primitive, a fit of its single outline, or a stroke as long as it is wide)
/// whose two diameters are alike by the width's slack; longer, it is a dash.
pub(super) fn dot_of(shape: &Shape) -> Option<(Vec2, f32, f32)> {
    use crate::tracer::primitives::Primitive;
    if shape.fill.gradient.is_some() {
        return None;
    }
    let alike =
        |width: f32, extent: f32| extent - width <= (WIDTH_SLACK * extent).max(WIDTH_QUANTUM);
    if let Some(stroke) = stroke_of(shape) {
        if stroke.closed || stroke.dashes.is_some() {
            return None;
        }
        // A round or square cap reaches half the width past each end, a width in all
        let along = arc_table(&stroke);
        let length = along[along.len() - 1].0;
        let reach = if stroke.cap == Cap::Butt {
            0.
        } else {
            stroke.width
        };
        let (a, b) = (stroke.width, length + reach);
        let (width, extent) = (a.min(b), a.max(b));
        let middle = along
            .iter()
            .min_by(|p, q| {
                (p.0 - length / 2.)
                    .abs()
                    .total_cmp(&(q.0 - length / 2.).abs())
            })
            .map_or(stroke.centerline[0][0], |p| p.1);
        return (alike(width, extent) && extent <= MAX_WIDTH).then_some((middle, width, extent));
    }
    let disc = |p: Primitive| {
        let (center, a, b) = match p {
            Primitive::Circle { center, radius } => (center, radius, radius),
            Primitive::Ellipse { center, radii, .. } => (center, radii.x, radii.y),
            Primitive::RoundedRect {
                center,
                half,
                radius,
                ..
            } if radius >= half.min_element() => (center, half.x, half.y),
            Primitive::RoundedRect { .. } => return None,
        };
        let (width, extent) = (2. * a.min(b), 2. * a.max(b));
        alike(width, extent).then_some((center, width, extent))
    };
    let (center, width, extent) = match shape.primitive {
        Some((_, primitive)) => disc(primitive)?,
        None => {
            if shape.loops.len() != 1 {
                return None;
            }
            // Sampled as the primitive fit samples outlines
            const SAMPLES: usize = 8;
            let all: Vec<Vec2> = shape.loops[0]
                .iter()
                .flat_map(|c| (0..SAMPLES).map(move |i| eval(c, i as f32 / SAMPLES as f32)))
                .collect();
            disc(crate::tracer::primitives::fit_primitive(
                &all,
                &all,
                crate::complete::PRIMITIVE_TOLERANCE,
            )?)?
        }
    };
    (width > 0. && extent <= MAX_WIDTH).then_some((center, width, extent))
}

/// The chains of dots, each with the dotted stroke through them, chained by place alone: each
/// dot takes, cheapest first, the two links to neighbors on either side with alike spacings
/// that turn the least, so it has at most one link before it and one after.
pub(super) fn dot_chains(scene: &Scene, taken: &[bool]) -> Vec<(Vec<usize>, Stroke)> {
    let dots: Vec<Dot> = scene
        .shapes
        .iter()
        .enumerate()
        .filter(|&(position, shape)| shape.clip.is_none() && !taken[position])
        .filter_map(|(position, shape)| {
            let (center, width, extent) = dot_of(shape)?;
            Some(Dot {
                position,
                center,
                width,
                extent,
                color: None,
            })
        })
        .collect();
    chain_dots(&dots)
}

/// The chains among `dots` (see [`dot_chains`]), each as its dots' positions and the dotted
/// stroke through them.
pub(super) fn chain_dots(dots: &[Dot]) -> Vec<(Vec<usize>, Stroke)> {
    if dots.len() < MIN_DOTS {
        return Vec::new();
    }
    let alike = |a: f32, b: f32| (a - b).abs() <= (WIDTH_SLACK * a.max(b)).max(WIDTH_QUANTUM);
    // Neighbors: dots of alike widths with no other dot inside the circle on the two as its
    // diameter (the Gabriel graph), which along a dotted line leaves only adjacent dots,
    // whatever the spacing, with no distance to choose
    let n = dots.len();
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (&dots[i], &dots[j]);
            if !alike(a.width, b.width) {
                continue;
            }
            if let (Some(ca), Some(cb)) = (a.color, b.color)
                && ca.distance(cb) > ALIKE_COLOR
            {
                continue;
            }
            let mid = (a.center + b.center) / 2.;
            let radius = a.center.distance(b.center) / 2.;
            let between = dots
                .iter()
                .enumerate()
                .any(|(k, d)| k != i && k != j && d.center.distance(mid) < radius);
            if !between {
                neighbors[i].push(j);
                neighbors[j].push(i);
            }
        }
    }
    // A dot `b` between `a` and `c`: spacings alike by the width slack and the path running on
    // through `b` (not the centerline's kink bound, since dots sample a curve sparsely and
    // turn by a whole spacing's curvature). The cost prefers alike spacings and small turns.
    let triple = |a: usize, b: usize, c: usize| -> Option<f32> {
        let (u, v) = (
            dots[b].center - dots[a].center,
            dots[c].center - dots[b].center,
        );
        let (s1, s2) = (u.length(), v.length());
        if !alike(s1, s2) || u.dot(v) <= 0. {
            return None;
        }
        let turn = u.perp_dot(v).atan2(u.dot(v)).abs();
        Some((s1 - s2).abs() / s1.max(s2) + turn / std::f32::consts::FRAC_PI_2)
    };
    let mut triples: Vec<(f32, usize, usize, usize)> = Vec::new();
    for (b, around) in neighbors.iter().enumerate() {
        for (x, &a) in around.iter().enumerate() {
            for &c in &around[x + 1..] {
                if let Some(cost) = triple(a, b, c) {
                    triples.push((cost, a, b, c));
                }
            }
        }
    }
    triples.sort_by(|x, y| x.0.total_cmp(&y.0));
    // Each dot takes the cheapest triple around it that its neighbors' choices allow: a dot
    // with a triple has exactly its two links, and a dot without one at most one link (an
    // end of a chain), or a second where the two make a triple of their own
    let mut middle: Vec<Option<(usize, usize)>> = vec![None; n];
    let mut links: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(_, a, b, c) in &triples {
        if middle[b].is_some() || links[b].iter().any(|&x| x != a && x != c) {
            continue;
        }
        let mut becomes: Vec<(usize, (usize, usize))> = Vec::new();
        let ok = [a, c].into_iter().all(|x| {
            if links[x].contains(&b) {
                true
            } else if middle[x].is_some() {
                false
            } else if let [y] = links[x][..] {
                let fits = triple(y, x, b).is_some();
                if fits {
                    becomes.push((x, (y, b)));
                }
                fits
            } else {
                links[x].is_empty()
            }
        });
        if !ok {
            continue;
        }
        middle[b] = Some((a, c));
        for (x, pair) in becomes {
            middle[x] = Some(pair);
        }
        for x in [a, c] {
            if !links[x].contains(&b) {
                links[x].push(b);
                links[b].push(x);
            }
        }
    }
    // Chains: from a dot with one link along the links; what is left closes on itself
    let mut used = vec![false; n];
    let walk = |start: usize, used: &mut Vec<bool>| -> Vec<usize> {
        let mut chain = vec![start];
        used[start] = true;
        let mut at = start;
        while let Some(&next) = links[at].iter().find(|&&x| !used[x]) {
            used[next] = true;
            chain.push(next);
            at = next;
        }
        chain
    };
    let mut chains: Vec<(Vec<usize>, bool)> = Vec::new();
    for start in 0..n {
        if !used[start] && links[start].len() == 1 {
            chains.push((walk(start, &mut used), false));
        }
    }
    for start in 0..n {
        if !used[start] && links[start].len() == 2 {
            chains.push((walk(start, &mut used), true));
        }
    }
    chains
        .into_iter()
        .filter(|(chain, _)| chain.len() >= MIN_DOTS)
        .filter_map(|(chain, closed)| {
            let members: Vec<&Dot> = chain.iter().map(|&i| &dots[i]).collect();
            let dotted = dotted_stroke(&members, closed)?;
            Some((chain.iter().map(|&i| dots[i].position).collect(), dotted))
        })
        .collect()
}

/// One round-capped stroke through `dots`, in order, with a dash at each: the
/// centerline fitted through a smooth curve through the centers, the width and the dash the
/// dots' medians, and the pattern read off where the centers lie along the centerline.
pub(super) fn dotted_stroke(dots: &[&Dot], closed: bool) -> Option<Stroke> {
    let centers: Vec<Vec2> = dots.iter().map(|d| d.center).collect();
    let n = centers.len();
    // A Catmull-Rom curve through the centers sampled about a pixel apart, so a ring is not
    // chopped by lines at every dot; a closed chain starts halfway between its last dot and
    // its first, so the seam lies in a gap
    let at = |i: isize| -> Vec2 {
        if closed {
            centers[i.rem_euclid(n as isize) as usize]
        } else if i < 0 {
            2. * centers[0] - centers[1]
        } else if i as usize >= n {
            2. * centers[n - 1] - centers[n - 2]
        } else {
            centers[i as usize]
        }
    };
    let mut samples: Vec<Vec2> = Vec::new();
    let mut first_steps = 0;
    let segments: Vec<isize> = if closed {
        (0..n as isize).map(|k| k - 1).collect()
    } else {
        (0..n as isize - 1).collect()
    };
    for (s, &i) in segments.iter().enumerate() {
        let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
        let steps = (p1.distance(p2).ceil() as usize).max(1);
        if s == 0 {
            first_steps = steps;
        }
        for j in 0..steps {
            let t = j as f32 / steps as f32;
            let (t2, t3) = (t * t, t * t * t);
            samples.push(
                0.5 * (2. * p1
                    + (p2 - p0) * t
                    + (2. * p0 - 5. * p1 + 4. * p2 - p3) * t2
                    + (3. * p1 - p0 - 3. * p2 + p3) * t3),
            );
        }
    }
    if closed {
        samples.rotate_left(first_steps / 2);
    } else {
        samples.push(centers[n - 1]);
    }
    if samples.len() < 4 {
        return None;
    }
    // Held tighter than the skeleton's pixel: a dot's center is fitted to its whole outline,
    // and a curve a pixel off it leaves a crescent of the dot's width on either side
    let centerline = fit_centerline(&samples, closed, WIDTH_QUANTUM / 4.)?;
    let median = |v: Vec<f32>| {
        let mut v = v;
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    let width = median(dots.iter().map(|d| d.width).collect());
    let extent = median(dots.iter().map(|d| d.extent).collect());
    let m = centerline.len();
    let mut stroke = Stroke {
        centerline,
        closed,
        width,
        cap: Cap::Round,
        join: crate::stroke::Join::Round,
        corners: vec![false; m],
        dashes: None,
    };
    // A dot drawn as a round-capped dash of length d at width w reaches d + w along the path;
    // a round dot is a dash of no length
    let length = (extent - width).max(0.);
    // the centerline runs on past each end dot by DOT_MARGIN, so the pattern draws it
    // (a dash on the very end is none of its spans)
    let reach = DOT_MARGIN;
    let along = arc_table(&stroke);
    let mut spans: Vec<(f32, f32)> = centers
        .iter()
        .map(|&c| {
            let s = nearest_arc(&along, c);
            (s - length / 2. - reach, s + length / 2. + reach)
        })
        .collect();
    let ends = stroke.arc_lengths();
    let total = ends[ends.len() - 1];
    extend_to_spans(&mut stroke, &mut spans, total);
    for span in &mut spans {
        (span.0, span.1) = (span.0 + reach, span.1 - reach);
    }
    stroke.dashes = Some(pattern_of_spans(&mut spans, closed, total));
    Some(stroke)
}
