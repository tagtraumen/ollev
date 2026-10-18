//! A layer's extent carried under the opaque parts above it: the region stops where something
//! opaque lies over the layer, though the layer runs on beneath, so the region's outline is
//! continued across those parts.

use super::completion::{
    End, V2, add, components, contour_ends, cross, dot, mul, norm, pair_ends, sub,
};
use super::raster::{DVec3, length_px, neighbours4};

/// What the carrying did, for the log.
#[derive(Default, Debug)]
pub struct Report {
    /// the region's contour ends at the parts and the pairs joined
    pub ends: usize,
    pub pairs: usize,
    /// the ends with no partner continued along their tangents onto the region
    pub rays: usize,
    /// pieces of the parts enclosed by the region, and their pixels
    pub pieces: usize,
    pub added_px: usize,
    /// the joining curves drawn into the parts
    pub barrier: Vec<bool>,
    /// the parts as crossed, with what they enclose
    pub parts: Vec<bool>,
    /// the curves and the ends, for the picture
    pub curves: Vec<[V2; 4]>,
    pub ends_at: Vec<End>,
}

/// The region carried under `parts`: its contour ends at the parts ([`contour_ends`], and with
/// `lines` [`junction_ends`]) paired across a part and joined by tangent cubics as barriers; a
/// piece they cut off, bounded by counterpart curves and the region alone, joins the extent.
pub fn carry_under(
    region: &[bool],
    parts: &[bool],
    lines: Option<(&[u32], &[u32])>,
    w: usize,
    h: usize,
) -> (Vec<bool>, Report) {
    let mut parts: Vec<bool> = parts.iter().zip(region).map(|(p, r)| *p && !*r).collect();
    // what the parts enclose is under them as much as they are: a piece of no part that
    // touches only the parts and the region (a highlight on a part the relation reads as the
    // ground would cut it into slivers)
    {
        let rest: Vec<bool> = (0..w * h).map(|i| !parts[i] && !region[i]).collect();
        let (comp, n) = components(&rest, w, h);
        let mut open = vec![false; n];
        let mut by_part = vec![false; n];
        for i in 0..w * h {
            if !rest[i] {
                continue;
            }
            let c = comp[i] as usize;
            let (x, y) = (i % w, i / w);
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                open[c] = true;
            }
            if neighbours4(i, w, h).any(|j| parts[j]) {
                by_part[c] = true;
            }
        }
        for i in 0..w * h {
            if rest[i] && !open[comp[i] as usize] && by_part[comp[i] as usize] {
                parts[i] = true;
            }
        }
    }
    let mut report = Report::default();
    if !parts.iter().any(|p| *p) {
        return (region.to_vec(), report);
    }
    // the region's own picture: white on black, two pieces
    let picture: Vec<DVec3> = region
        .iter()
        .map(|&r| if r { DVec3::ONE } else { DVec3::ZERO })
        .collect();
    let label: Vec<u32> = region.iter().map(|&r| r as u32).collect();
    let (mut ends, _) = contour_ends(&picture, &label, &parts, w, h);
    if let Some((lines, objects)) = lines {
        let (part_of, _) = components(&parts, w, h);
        for mut e in junction_ends(region, lines, objects, w, h) {
            // the end belongs to the part it points into
            let q = [e.p[0] + e.d[0], e.p[1] + e.d[1]];
            let (x, y) = (q[0].floor(), q[1].floor());
            if x < 0.0 || y < 0.0 || x >= w as f64 || y >= h as f64 {
                continue;
            }
            let i = y as usize * w + x as usize;
            if !parts[i] {
                continue;
            }
            // one end per place and direction: the crack reading may have it already
            if ends
                .iter()
                .any(|f| norm(sub(f.p, e.p)) <= 1.5 && dot(f.d, e.d) > 0.9)
            {
                continue;
            }
            e.hole = part_of[i];
            ends.push(e);
        }
    }
    let (curves, mut barrier) = pair_ends(&ends, &parts, w, h);
    report.ends = ends.len();
    report.pairs = curves.len();
    report.curves = curves;
    report.ends_at = ends;
    // An end with no partner continues along its tangent while it stays in the part; where it
    // comes out on the region, the ray is a barrier too. Every curve and ray links the two
    // region pieces at its ends.
    let (rcomp, _) = components(region, w, h);
    // the region piece an end stands on: the nearest region pixel within the anti-aliasing
    // scale, as a junction end is read up to the scale off the region
    let near = length_px(w, 1.5) as isize;
    let piece_at = |p: V2| -> Option<u32> {
        let (x, y) = (p[0].floor() as isize, p[1].floor() as isize);
        (-near..=near)
            .flat_map(|dy| (-near..=near).map(move |dx| (x + dx, y + dy)))
            .filter(|&(xx, yy)| xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h)
            .map(|(xx, yy)| yy as usize * w + xx as usize)
            .filter(|&i| region[i])
            .min_by_key(|&i| {
                let (xx, yy) = ((i % w) as isize, (i / w) as isize);
                (xx - x) * (xx - x) + (yy - y) * (yy - y)
            })
            .map(|i| rcomp[i])
    };
    // the pixels off the region from `p` along `back` within the anti-aliasing scale, where
    // a curve or ray is sealed to the region behind its end
    let behind = |p: V2, back: V2| -> Vec<usize> {
        let mut out = Vec::new();
        let mut t = 0.5;
        while t <= near as f64 + 0.5 {
            let q = add(p, mul(back, t));
            let (x, y) = (q[0].floor(), q[1].floor());
            if x < 0.0 || y < 0.0 || x >= w as f64 || y >= h as f64 {
                break;
            }
            let i = y as usize * w + x as usize;
            if region[i] {
                break;
            }
            out.push(i);
            t += 0.5;
        }
        out
    };
    let mut curve_of = vec![u32::MAX; w * h];
    // the pixels each curve or ray draws, assigned to `curve_of` once the counterparts are
    // known: a curve with no counterpart cuts nothing and must not overwrite one that does
    let mut curve_px: Vec<Vec<usize>> = Vec::new();
    // every pixel a curve passes through, part or not: the curve closes the cracks it crosses,
    // so an inlet of the rest closed off by the curve and the region is under the parts too
    let mut closed = vec![false; w * h];
    let mut links: Vec<(u32, u32)> = Vec::new();
    for c in &report.curves {
        let (Some(a), Some(b)) = (piece_at(c[0]), piece_at(c[3])) else {
            links.push((u32::MAX, u32::MAX));
            curve_px.push(Vec::new());
            continue;
        };
        links.push((a.min(b), a.max(b)));
        curve_px.push(Vec::new());
        let eval = |t: f64| -> V2 {
            let u = 1.0 - t;
            [
                c[0][0] * u * u * u
                    + 3.0 * c[1][0] * u * u * t
                    + 3.0 * c[2][0] * u * t * t
                    + c[3][0] * t * t * t,
                c[0][1] * u * u * u
                    + 3.0 * c[1][1] * u * u * t
                    + 3.0 * c[2][1] * u * t * t
                    + c[3][1] * t * t * t,
            ]
        };
        let samples = (norm(sub(c[3], c[0])) * 8.0).ceil().max(8.0) as usize;
        for k in 0..=samples {
            let q = eval(k as f64 / samples as f64);
            let (x, y) = (q[0].floor(), q[1].floor());
            if x >= 0.0 && y >= 0.0 && x < w as f64 && y < h as f64 {
                let i = y as usize * w + x as usize;
                closed[i] = true;
                if barrier[i] {
                    curve_px.last_mut().unwrap().push(i);
                } else if !parts[i] {
                    // the pairing draws a sample outside the parts onto the part pixels
                    // beside it (see pair_ends): those are this curve's too
                    for j in neighbours4(i, w, h) {
                        if barrier[j] {
                            curve_px.last_mut().unwrap().push(j);
                        }
                    }
                }
            }
        }
        // the curve sealed to the region at both ends: a junction end stands up to the
        // anti-aliasing scale off the region, and the piece behind the curve would run around
        // its end through the part pixels between
        for (p, d) in [(c[0], sub(c[1], c[0])), (c[3], sub(c[2], c[3]))] {
            let n = norm(d);
            if n < 1e-9 {
                continue;
            }
            for i in behind(p, mul(d, -1.0 / n)) {
                closed[i] = true;
                if parts[i] {
                    barrier[i] = true;
                    curve_px.last_mut().unwrap().push(i);
                }
            }
        }
    }
    let used = |p: [f64; 2]| report.curves.iter().any(|c| (c[0] == p) || (c[3] == p));
    let mut starts: Vec<[f64; 2]> = Vec::new();
    for e in report.ends_at.iter().filter(|e| !used(e.p)) {
        let mut ray: Vec<usize> = Vec::new();
        let mut t = 0.5;
        let lands = loop {
            let q = [e.p[0] + e.d[0] * t, e.p[1] + e.d[1] * t];
            let (x, y) = (q[0].floor(), q[1].floor());
            if x < 0.0 || y < 0.0 || x >= w as f64 || y >= h as f64 {
                break None;
            }
            let i = y as usize * w + x as usize;
            if region[i] {
                break Some(rcomp[i]);
            }
            if !parts[i] {
                break None;
            }
            if ray.last() != Some(&i) {
                ray.push(i);
            }
            t += 0.5;
        };
        if let Some(b) = lands
            && !ray.is_empty()
        {
            report.rays += 1;
            starts.push(e.p);
            let a = piece_at(e.p).unwrap_or(b);
            links.push((a.min(b), a.max(b)));
            curve_px.push(Vec::new());
            for i in ray {
                barrier[i] = true;
                curve_px.last_mut().unwrap().push(i);
            }
            // and the ray sealed to the region behind its end, as the curves are
            for i in behind(e.p, mul(e.d, -1.0)) {
                closed[i] = true;
                if parts[i] {
                    barrier[i] = true;
                    curve_px.last_mut().unwrap().push(i);
                }
            }
        }
    }
    // the curves' ends, where a cut-off piece meets the region's outline: the pixels of the
    // outside within the anti-aliasing scale of an end are where the barrier meets the
    // crack, not something the piece reaches
    let ends: Vec<[f64; 2]> = report
        .curves
        .iter()
        .flat_map(|c| [c[0], c[3]])
        .chain(starts)
        .collect();
    let at_end = |i: usize| {
        let q = [(i % w) as f64 + 0.5, (i / w) as f64 + 0.5];
        ends.iter()
            .any(|e| ((q[0] - e[0]).powi(2) + (q[1] - e[1]).powi(2)).sqrt() <= 1.5)
    };
    // Only the curves with a counterpart (another curve or ray linking the same two pieces
    // of the region) cut the parts into pieces: a curve alone encloses nothing by the rule
    // below, and as a barrier it would only cut the band between its counterparts
    let mut count = crate::HashMap::<(u32, u32), usize>::default();
    for l in &links {
        if l.0 != u32::MAX {
            *count.entry(*l).or_default() += 1;
        }
    }
    let multi: Vec<bool> = links
        .iter()
        .map(|l| l.0 != u32::MAX && count[l] >= 2)
        .collect();
    for (k, px) in curve_px.iter().enumerate() {
        if multi[k] {
            for &i in px {
                curve_of[i] = k as u32;
            }
        }
    }
    let barrier: Vec<bool> = (0..w * h)
        .map(|i| barrier[i] && curve_of[i] != u32::MAX)
        .collect();
    let open: Vec<bool> = parts.iter().zip(&barrier).map(|(p, b)| *p && !*b).collect();
    let (comp, n) = components(&open, w, h);
    // per piece: does it touch the region, does it reach anything else
    let mut touches = vec![false; n];
    let mut touched: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut reaches = vec![false; n];
    for i in 0..w * h {
        if !open[i] {
            continue;
        }
        let c = comp[i] as usize;
        let (x, y) = (i % w, i / w);
        // the frame is something else: a part running off the picture is not enclosed
        if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
            reaches[c] = true;
        }
        for j in neighbours4(i, w, h) {
            if parts[j] {
                continue;
            }
            if region[j] {
                if !touched[c].contains(&rcomp[j]) {
                    touched[c].push(rcomp[j]);
                }
                touches[c] = true;
            } else if !at_end(j) && !closed[j] {
                reaches[c] = true;
            }
        }
    }
    // A piece is carried only when the curves enclosing it are each other's counterparts: at
    // least two, all linking the same two region pieces. A piece closed by one curve, or by
    // curves to different places, is a bay of the outline bridged, not the layer continued.
    let mut bound: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in 0..w * h {
        if !barrier[i] || curve_of[i] == u32::MAX {
            continue;
        }
        for j in neighbours4(i, w, h) {
            if open[j] {
                let c = comp[j] as usize;
                if !bound[c].contains(&curve_of[i]) {
                    bound[c].push(curve_of[i]);
                }
            }
        }
    }
    // and the piece runs from the one linked region piece to the other: the layer continues
    // under the part from A to B (two curves along one edge enclose a lens touching one side)
    let counterparts = |c: usize| -> bool {
        bound[c].len() >= 2
            && bound[c]
                .iter()
                .all(|&k| links[k as usize] == links[bound[c][0] as usize])
            && {
                // from one piece of the region to another: a curve from a piece back to
                // itself bridges a bay of its own outline, which the layer's does not
                let (a, b) = links[bound[c][0] as usize];
                a != b && touched[c].contains(&a) && touched[c].contains(&b)
            }
    };
    let inside: Vec<bool> = (0..n)
        .map(|c| touches[c] && !reaches[c] && counterparts(c))
        .collect();
    report.pieces = inside.iter().filter(|v| **v).count();
    let mut out = region.to_vec();
    for i in 0..w * h {
        if open[i] && inside[comp[i] as usize] {
            out[i] = true;
            report.added_px += 1;
        }
    }
    // Only what the curves enclose together with the region is carried, and the barriers
    // themselves where they bound an enclosed piece: the layer's continued outline is its edge
    for i in 0..w * h {
        if !barrier[i] || out[i] {
            continue;
        }
        if neighbours4(i, w, h).any(|j| open[j] && inside[comp[j] as usize])
        {
            out[i] = true;
            report.added_px += 1;
        }
    }
    report.barrier = barrier;
    report.parts = parts;
    (out, report)
}

/// The region's contour ends where the grain line outside it changes from X to Y: if the X/Y
/// boundary continues X's stretch (chords over the reach), X occludes and Y's stretch ends
/// there, and conversely; neither collinear is the region's own corner, no end.
pub fn junction_ends(
    region: &[bool],
    lines: &[u32],
    objects: &[u32],
    w: usize,
    h: usize,
) -> Vec<End> {
    // only a line of the grain can be the occluder at a junction: a ground beside the region
    // read as one would point the region's edge down the side of what it meets
    let on_grain: crate::HashSet<u32> = (0..w * h)
        .filter(|&i| objects[i] != u32::MAX)
        .map(|i| lines[i])
        .collect();
    let reach = length_px(w, 4.0) as f64;
    let offset_tol = |dist: f64| 1.5 + 0.2 * dist;
    let at = |x: isize, y: isize| -> Option<usize> {
        (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h)
            .then(|| y as usize * w + x as usize)
    };
    // the line outside a crack, read past the anti-aliasing: the last pixel outside the
    // region within the scale along the crack's normal
    let probe = length_px(w, 1.5).max(1) as isize;
    let outside_at = |x: isize, y: isize, dx: isize, dy: isize| -> Option<u32> {
        let mut last = None;
        for s in 1..=probe {
            match at(x + s * dx, y + s * dy) {
                Some(j) if !region[j] => last = Some(lines[j]),
                _ => break,
            }
        }
        last
    };
    // the region's cracks against the outside, by the line outside: midpoint and the region
    // pixel's centre; and the cracks between two outside lines, by the pair
    let mut edge: Vec<(u32, V2, V2)> = Vec::new();
    let mut between: Vec<((u32, u32), V2)> = Vec::new();
    for y in 0..h as isize {
        for x in 0..w as isize {
            let i = (y * w as isize + x) as usize;
            for (dx, dy) in [(1isize, 0isize), (0, 1)] {
                let Some(j) = at(x + dx, y + dy) else {
                    continue;
                };
                let mid = [
                    x as f64 + 0.5 + dx as f64 * 0.5,
                    y as f64 + 0.5 + dy as f64 * 0.5,
                ];
                match (region[i], region[j]) {
                    (true, false) => {
                        if let Some(s) = outside_at(x, y, dx, dy) {
                            edge.push((s, mid, [x as f64 + 0.5, y as f64 + 0.5]));
                        }
                    }
                    (false, true) => {
                        if let Some(s) = outside_at(x + dx, y + dy, -dx, -dy) {
                            edge.push((s, mid, [(x + dx) as f64 + 0.5, (y + dy) as f64 + 0.5]));
                        }
                    }
                    (false, false) if lines[i] != lines[j] => {
                        let (a, b) = (lines[i].min(lines[j]), lines[i].max(lines[j]));
                        between.push(((a, b), mid));
                    }
                    _ => {}
                }
            }
        }
    }
    // the junction points: a region pixel with two different lines outside it along the
    // eight directions, read past the anti-aliasing; one point per pair within the scale
    let mut points: Vec<(u32, u32, V2)> = Vec::new();
    for y in 0..h as isize {
        for x in 0..w as isize {
            let i = (y * w as isize + x) as usize;
            if !region[i] {
                continue;
            }
            let mut outside: Vec<u32> = Vec::new();
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if (dx != 0 || dy != 0)
                        && let Some(s) = outside_at(x, y, dx, dy)
                        && !outside.contains(&s)
                    {
                        outside.push(s);
                    }
                }
            }
            let r = [x as f64 + 0.5, y as f64 + 0.5];
            for a in 0..outside.len() {
                for b in a + 1..outside.len() {
                    let (xa, yb) = (outside[a].min(outside[b]), outside[a].max(outside[b]));
                    // the junction point is where the X/Y boundary meets the outline: its
                    // crack nearest this pixel
                    let Some(p) = between
                        .iter()
                        .filter(|(pair, _)| *pair == (xa, yb))
                        .map(|(_, m)| *m)
                        .filter(|m| norm(sub(*m, r)) <= reach)
                        .min_by(|a, b| norm(sub(*a, r)).total_cmp(&norm(sub(*b, r))))
                    else {
                        continue;
                    };
                    if !points.iter().any(|(u, v, q)| {
                        *u == xa && *v == yb && norm(sub(*q, p)) <= length_px(w, 1.5) as f64
                    }) {
                        points.push((xa, yb, p));
                    }
                }
            }
        }
    }
    // a stretch of cracks as a line: its centroid and its axis (the principal direction),
    // the axis pointing from the centroid toward `p`
    let axis = |p: V2, pts: &[V2]| -> Option<(V2, V2)> {
        if pts.len() < 2 {
            return None;
        }
        let mut c = [0.0, 0.0];
        for q in pts {
            c = add(c, *q);
        }
        c = mul(c, 1.0 / pts.len() as f64);
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for q in pts {
            let d = sub(*q, c);
            sxx += d[0] * d[0];
            sxy += d[0] * d[1];
            syy += d[1] * d[1];
        }
        let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let mut d = [theta.cos(), theta.sin()];
        if dot(d, sub(p, c)) < 0.0 {
            d = mul(d, -1.0);
        }
        (norm(sub(p, c)) >= reach / 4.0).then_some((c, d))
    };
    let mut ends: Vec<End> = Vec::new();
    for (sx, sy, p) in points {
        let near = |line: u32| -> Vec<V2> {
            edge.iter()
                .filter(|(s, m, _)| *s == line && norm(sub(*m, p)) <= reach)
                .map(|(_, m, _)| *m)
                .collect()
        };
        let (Some((cx, dx)), Some((cy, dy))) = (axis(p, &near(sx)), axis(p, &near(sy))) else {
            continue;
        };
        // the X/Y boundary leaving the junction: its far point over the reach
        let far: Vec<V2> = between
            .iter()
            .filter(|(pair, m)| *pair == (sx, sy) && norm(sub(*m, p)) <= reach)
            .map(|(_, m)| *m)
            .collect();
        if far.is_empty() {
            continue;
        }
        let along = |c: V2, d: V2| -> bool {
            // the X/Y boundary's cracks ahead of the junction along the stretch, over the
            // reach, lie on the stretch's line. Only those ahead count: a strand's edge runs
            // through the junction both ways, its centroid then within reach of any stretch
            let ahead: Vec<V2> = far
                .iter()
                .copied()
                .filter(|m| dot(d, sub(*m, p)) > 0.0)
                .collect();
            if ahead.len() < 2 {
                return false;
            }
            let f = mul(
                ahead.iter().fold([0.0, 0.0], |c, q| add(c, *q)),
                1.0 / ahead.len() as f64,
            );
            let l = norm(sub(f, p));
            l >= reach / 4.0 && cross(d, sub(f, c)).abs() <= offset_tol(l)
        };
        let (stem_dir, stem_line) = if on_grain.contains(&sx) && along(cx, dx) {
            (dy, sy)
        } else if on_grain.contains(&sy) && along(cy, dy) {
            (dx, sx)
        } else {
            continue;
        };
        // the region's side of the stem: the region pixels of the stem's cracks
        let mut side = 0.0;
        for (s, m, r) in &edge {
            if *s == stem_line && norm(sub(*m, p)) <= reach {
                side += cross(stem_dir, sub(*r, p)).signum();
            }
        }
        let (left, right) = if side > 0.0 {
            (DVec3::ONE, DVec3::ZERO)
        } else {
            (DVec3::ZERO, DVec3::ONE)
        };
        ends.push(End {
            p,
            d: stem_dir,
            hole: 0,
            left,
            right,
        });
    }
    ends
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A band crossed by an opaque bar: the band is carried under the bar between its two
    /// edges, and the bar's ends beyond the band stay out.
    #[test]
    fn band_carried_under_a_crossing_bar() {
        // at the reference width, so the completion's reach (4 units) is 4 px
        let (w, h) = (330usize, 120usize);
        let band = |x: usize, y: usize| (45..75).contains(&y) && (20..300).contains(&x);
        let bar = |x: usize, _y: usize| (150..175).contains(&x);
        let region: Vec<bool> = (0..w * h)
            .map(|i| band(i % w, i / w) && !bar(i % w, i / w))
            .collect();
        let parts: Vec<bool> = (0..w * h).map(|i| bar(i % w, i / w)).collect();
        let (out, report) = carry_under(&region, &parts, None, w, h);
        assert!(report.pairs >= 2, "{report:?}");
        for y in 47..73 {
            for x in 150..175 {
                assert!(out[y * w + x], "under the bar at ({x}, {y})");
            }
        }
        for y in [2usize, 30, 90, 117] {
            assert!(!out[y * w + 160], "the bar beyond the band at y {y}");
        }
    }

    /// A band whose edges border a wide line all along crossed by a second line: the junction
    /// reading gives the band's edges their ends at the crossing, and the band is carried
    /// under it.
    #[test]
    fn band_along_a_line_crossed_by_another() {
        let (w, h) = (330usize, 120usize);
        let band = |x: usize, y: usize| (45..75).contains(&y) && (20..300).contains(&x);
        let bar = |x: usize, _y: usize| (150..175).contains(&x);
        let region: Vec<bool> = (0..w * h)
            .map(|i| band(i % w, i / w) && !bar(i % w, i / w))
            .collect();
        // the lines: 1 above and below the band (one wide line each side), 2 the bar
        let lines: Vec<u32> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                if bar(x, y) {
                    2
                } else if y < 45 {
                    1
                } else {
                    3
                }
            })
            .collect();
        let parts: Vec<bool> = (0..w * h).map(|i| !region[i]).collect();
        let objects: Vec<u32> = lines
            .iter()
            .map(|&l| if l == 3 { u32::MAX } else { l })
            .collect();
        let (out, report) = carry_under(&region, &parts, Some((&lines, &objects)), w, h);
        assert!(report.pairs >= 2, "{report:?}");
        for y in 47..73 {
            for x in 150..175 {
                assert!(out[y * w + x], "under the bar at ({x}, {y})");
            }
        }
        assert!(!out[20 * w + 160] && !out[100 * w + 160]);
    }
}
