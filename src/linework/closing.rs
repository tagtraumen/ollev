//! Cuts along a kept line and the paths that close a cut.

use super::*;

/// A cut candidate at its places in the stack: the best. Its price is a path's outline with
/// one segment for its paint as a mark's, and one more for a gradient's direction.
pub(super) fn try_cut(
    ctx: &crate::Context,
    scene: &crate::Scene,
    img: &Image,
    field: &EdgeField,
    lambda: f64,
    c: &CutFit,
) -> Option<Choice> {
    let (w, h) = (img.width, img.height);
    let (shape, extra) = cut_shape(c);
    let fill = shape.fill;
    let shapes = vec![shape];
    let window = window_of(&shapes, w, h);
    if window.is_empty() {
        return None;
    }
    let cov = joint_coverage(&shapes, window);
    let cubics: Vec<Bezier> = c.outline.iter().flatten().copied().collect();
    let price = extra * ctx.segment_cost + ctx.path_cost;
    let mut best: Option<(f64, usize)> = None;
    for position in positions(scene, window) {
        let site = Site::new(scene, window, position);
        let at = |j: usize| {
            let (x, y) = (window.x + j % window.width, window.y + j / window.width);
            fill.at_pixel(x, y).as_dvec3()
        };
        let (plain, edge) = site.saving(img, field, lambda, &cov, &at);
        let gain = plain + c.smooth * edge - price;
        if best.as_ref().is_none_or(|b| gain > b.0) {
            best = Some((gain, position));
        }
    }
    let (gain, position) = best?;
    Some(Choice {
        net: gain - (cubics.len() as f64 + 1.0) * ctx.segment_cost,
        gain,
        extra: price,
        width: 0.0,
        kind: crate::translucency::vocabulary::MarkKind::Cut,
        shapes,
        cubics,
        position,
    })
}

/// A line drawn over a cut along it: the area on each side (closed by [`closed_sides`] where
/// the sides are one) refitted by [`fit_cut`] and drawn on top with the line over it, so a
/// merged leg becomes its own shape; weighed jointly with `line`, priced as two marks.
#[allow(clippy::too_many_arguments)]
pub(super) fn line_cuts(
    ctx: &crate::Context,
    scene: &crate::Scene,
    img: &Image,
    comp: &[DVec3],
    field: &EdgeField,
    opt: &Options,
    f: &Fit,
    line: &Choice,
    areas: &mut Areas,
    fitted: &mut crate::HashMap<u32, Option<CutFit>>,
) -> Vec<Choice> {
    let (w, h) = (img.width, img.height);
    let through: Vec<DVec2> = f.flat.iter().flatten().copied().collect();
    if through.len() < 2 {
        return Vec::new();
    }
    let parts: Vec<u32> = match sides(areas, &through, w, h) {
        Some(s) => s.to_vec(),
        None => {
            let nrm = normals(&through);
            let mut v: Vec<u32> = through
                .iter()
                .zip(&nrm)
                .step_by(2)
                .flat_map(|(p, n)| [*p + 2.0 * *n, *p - 2.0 * *n])
                .filter(|q| q.x >= 0.0 && q.y >= 0.0 && (q.x as usize) < w && (q.y as usize) < h)
                .map(|q| areas.label[q.y as usize * w + q.x as usize])
                .filter(|&l| l > 0)
                .collect();
            v.sort_unstable();
            v.dedup();
            let mut out = Vec::new();
            for l in v {
                if let Some([pa, pb]) = closed_sides(areas, l, &through, w, h) {
                    let first = areas.pixels.len() as u32;
                    areas.pixels.push(pa);
                    areas.pixels.push(pb);
                    out.extend([first, first + 1]);
                    break;
                }
            }
            out
        }
    };
    let top = scene.shapes.len();
    let mut out = Vec::new();
    for a in parts {
        let Some(c) = fitted
            .entry(a)
            .or_insert_with(|| fit_cut(img, comp, field, opt, areas, a, f.smooth))
            .clone()
        else {
            continue;
        };
        let (cut, extra) = cut_shape(&c);
        let mut shapes = vec![cut];
        shapes.extend(line.shapes.iter().cloned());
        let window = window_of(&shapes, w, h);
        if window.is_empty() {
            continue;
        }
        let (plain, edge) = joint_saving(
            scene,
            img,
            field,
            opt.edge,
            window,
            Variant::Same,
            (top, &shapes),
        );
        let mut cubics = line.cubics.clone();
        cubics.extend(c.outline.iter().flatten().copied());
        // two marks: the cut's paint (and a gradient's direction) and path beyond the line's
        let gain =
            plain + f.smooth * edge - line.extra - (1.0 + extra) * ctx.segment_cost - ctx.path_cost;
        out.push(Choice {
            net: gain - (cubics.len() as f64 + 1.0) * ctx.segment_cost,
            gain,
            extra: line.extra + (1.0 + extra) * ctx.segment_cost + ctx.path_cost,
            width: line.width,
            kind: crate::translucency::vocabulary::MarkKind::Line,
            shapes,
            cubics,
            position: top,
        });
    }
    out
}

/// A cut's shape, filled flat or by its linear gradient, and the segments its paint costs
/// beyond a mark's (one for a gradient's direction).
pub(super) fn cut_shape(c: &CutFit) -> (crate::Shape, f64) {
    let (fill, extra) = match c.fill {
        CutFill::Flat(col) => (crate::Fill::solid(col.as_vec3()), 0.0),
        CutFill::Linear { from, to, c0, c1 } => (
            crate::Fill {
                color: c0.as_vec3(),
                gradient: Some(crate::render::Gradient::Linear {
                    start: from.as_vec2(),
                    end: to.as_vec2(),
                    to: crate::render::opaque(c1.as_vec3().to_array()),
                    stops: crate::render::scene::Stops::NONE,
                }),
                opacity: 1.0,
            },
            1.0,
        ),
    };
    let mut shape = crate::Shape::new(
        c.outline
            .iter()
            .map(|l| l.iter().map(to_cubic).collect())
            .collect(),
        fill,
    );
    shape.smooth = true;
    (shape, extra)
}

/// The shortest 8-connected path from a step's end `from` through `inside`, within 45
/// degrees of the outward `heading`, to the first pixel off the area (an edge, another area,
/// the border), at most `reach` steps.
pub(super) fn closing_path(
    inside: &[bool],
    width: usize,
    height: usize,
    from: (usize, usize),
    heading: DVec2,
    reach: usize,
) -> Option<Vec<usize>> {
    use std::collections::VecDeque;
    let start = from.1 * width + from.0;
    let mut parent: crate::HashMap<usize, usize> = Default::default();
    let mut queue = VecDeque::from([(start, 0usize)]);
    parent.insert(start, start);
    let o = DVec2::new(from.0 as f64, from.1 as f64);
    while let Some((i, d)) = queue.pop_front() {
        if d >= reach {
            continue;
        }
        let (x, y) = ((i % width) as i64, (i / width) as i64);
        for (dx, dy) in [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ] {
            let (nx, ny) = (x + dx, y + dy);
            let off = nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64;
            let q = DVec2::new(nx as f64, ny as f64) - o;
            if q.length() > 0.0 && q.dot(heading) < std::f64::consts::FRAC_1_SQRT_2 * q.length() {
                continue;
            }
            if off || !inside[ny as usize * width + nx as usize] {
                // reached the closing edge: the path back to the end
                let mut path = vec![i];
                let mut k = i;
                while k != start {
                    k = parent[&k];
                    path.push(k);
                }
                return Some(path);
            }
            let j = ny as usize * width + nx as usize;
            if let std::collections::hash_map::Entry::Vacant(e) = parent.entry(j) {
                e.insert(i);
                queue.push_back((j, d + 1));
            }
        }
    }
    None
}

/// The two parts of area `label` either side of a step that fades out inside it (a shading
/// step between two legs), the step and its closures at both ends ([`closing_path`]) taken
/// out; none where they do not split it.
pub(super) fn closed_sides(
    areas: &Areas,
    label: u32,
    through: &[DVec2],
    width: usize,
    height: usize,
) -> Option<[Vec<usize>; 2]> {
    let px = &areas.pixels[label as usize];
    let mut inside = vec![false; width * height];
    for &i in px {
        if areas.label[i] == label {
            inside[i] = true;
        }
    }
    let n = through.len();
    if n < 2 {
        return None;
    }
    let pixel = |q: DVec2| -> Option<(usize, usize)> {
        let (x, y) = (q.x - 0.5, q.y - 0.5);
        (x >= 0.0 && y >= 0.0 && (x.round() as usize) < width && (y.round() as usize) < height)
            .then(|| (x.round() as usize, y.round() as usize))
    };
    let reach = arc_length(through).ceil() as usize;
    let back = (3 * (REACH + 1)).min(n - 1);
    let mut cut = vec![false; width * height];
    for q in through {
        if let Some((x, y)) = pixel(*q) {
            cut[y * width + x] = true;
        }
    }
    for (end, prev) in [(0, back), (n - 1, n - 1 - back)] {
        let heading = (through[end] - through[prev]).normalize_or_zero();
        let Some(from) = pixel(through[end]) else {
            continue;
        };
        if heading == DVec2::ZERO {
            continue;
        }
        if let Some(path) = closing_path(&inside, width, height, from, heading, reach) {
            for i in path {
                cut[i] = true;
            }
        }
    }
    let open: Vec<bool> = inside.iter().zip(&cut).map(|(a, c)| *a && !*c).collect();
    let (lab, _) = crate::translucency::raster::label(&open, width, height, Conn::Four);
    // the components either side, by the votes of the samples past the line
    let nrm = normals(through);
    let mut votes: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    for (p, nr) in through.iter().zip(&nrm).step_by(2) {
        for (s, sign) in [(0, 1.0), (1, -1.0)] {
            if let Some((x, y)) = pixel(*p + sign * 2.0 * *nr) {
                let l = lab[y * width + x];
                if l > 0 {
                    votes[s].push(l);
                }
            }
        }
    }
    let mode = |v: &mut Vec<u32>| -> u32 {
        v.sort_unstable();
        let (mut best, mut i) = ((0, 0), 0);
        while i < v.len() {
            let j = v[i..].iter().take_while(|&&l| l == v[i]).count();
            if j > best.1 {
                best = (v[i], j);
            }
            i += j;
        }
        best.0
    };
    let (a, b) = (mode(&mut votes[0]), mode(&mut votes[1]));
    if a == 0 || b == 0 || a == b {
        return None;
    }
    let part = |l: u32| -> Vec<usize> { (0..width * height).filter(|&i| lab[i] == l).collect() };
    Some([part(a), part(b)])
}

/// The cut candidates: the areas either side of each step, and where its two sides are one
/// area, the two parts its closure makes of it ([`closed_sides`]), added to `areas`.
#[allow(clippy::too_many_arguments)]
pub(super) fn closing_cut_candidates(
    img: &Image,
    comp: &[DVec3],
    field: &EdgeField,
    weak: &[bool],
    opt: &Options,
    chains: &[Chain],
    lines: &[&Linked],
    areas: &mut Areas,
    closed: &mut usize,
) -> Vec<CutFit> {
    let (w, h) = (img.width, img.height);
    let mut smooth: std::collections::BTreeMap<u32, f64> = Default::default();
    for line in lines {
        let (_, through) = polylines(chains, line);
        let sm = discount(&through, field, weak);
        let parts = match sides(areas, &through, w, h) {
            Some(s) => s,
            None => {
                // one area on both sides: the one the samples past the line land in most
                let nrm = normals(&through);
                let mut v: Vec<u32> = through
                    .iter()
                    .zip(&nrm)
                    .step_by(2)
                    .flat_map(|(p, n)| [*p + 2.0 * *n, *p - 2.0 * *n])
                    .filter(|q| {
                        q.x >= 0.0 && q.y >= 0.0 && (q.x as usize) < w && (q.y as usize) < h
                    })
                    .map(|q| areas.label[q.y as usize * w + q.x as usize])
                    .filter(|&l| l > 0)
                    .collect();
                v.sort_unstable();
                let mut best = (0u32, 0usize);
                let mut i = 0;
                while i < v.len() {
                    let j = v[i..].iter().take_while(|&&l| l == v[i]).count();
                    if j > best.1 {
                        best = (v[i], j);
                    }
                    i += j;
                }
                if best.0 == 0 {
                    continue;
                }
                let Some([pa, pb]) = closed_sides(areas, best.0, &through, w, h) else {
                    continue;
                };
                *closed += 1;
                let first = areas.pixels.len() as u32;
                areas.pixels.push(pa);
                areas.pixels.push(pb);
                [first, first + 1]
            }
        };
        for a in parts {
            let e = smooth.entry(a).or_insert(0.0);
            *e = e.max(sm);
        }
    }
    smooth
        .into_iter()
        .filter_map(|(a, s)| fit_cut(img, comp, field, opt, areas, a, s))
        .collect()
}
