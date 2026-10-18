//! Skeletons of a shape's mask: distance transform, thinning, the chains.

use super::*;

/// The skeleton of a shape's raster: the band's centerline as pixel chains, with the width
/// at each pixel.
#[derive(Clone)]
pub(super) struct Skeleton {
    /// Pixel chain from one end to the other, or around, in box coordinates.
    pub(super) chain: Vec<IVec2>,
    pub(super) closed: bool,
    /// Twice the distance to the outline at each chain pixel.
    pub(super) widths: Vec<f32>,
    /// Image position of the box's top left pixel.
    pub(super) top_left: IVec2,
}

/// The squared Euclidean distance to the nearest zero of `mask` for every pixel, by
/// Felzenszwalb's two passes.
pub(super) fn distance_squared(mask: &[bool], w: usize, h: usize) -> Vec<f32> {
    const INF: f32 = 1e12;
    let mut d: Vec<f32> = mask.iter().map(|&m| if m { INF } else { 0. }).collect();
    let mut f = vec![0f32; w.max(h)];
    let mut out = vec![0f32; w.max(h)];
    // Columns, then rows
    for x in 0..w {
        for y in 0..h {
            f[y] = d[y * w + x];
        }
        transform_1d(&f[..h], &mut out[..h]);
        for y in 0..h {
            d[y * w + x] = out[y];
        }
    }
    for y in 0..h {
        f[..w].copy_from_slice(&d[y * w..y * w + w]);
        transform_1d(&f[..w], &mut out[..w]);
        d[y * w..y * w + w].copy_from_slice(&out[..w]);
    }
    d
}

pub(super) fn transform_1d(f: &[f32], out: &mut [f32]) {
    let n = f.len();
    let mut v = vec![0usize; n];
    let mut z = vec![0f32; n + 1];
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    let sq = |q: usize| (q * q) as f32;
    for q in 1..n {
        loop {
            let p = v[k];
            let s = ((f[q] + sq(q)) - (f[p] + sq(p))) / (2. * (q as f32 - p as f32));
            if s <= z[k] && k > 0 {
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
        }
    }
    k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        *o = (q as f32 - p as f32).powi(2) + f[p];
    }
}

/// Whether Guo-Hall's sub-iteration `pass` deletes the set pixel `(x, y)` of `mask`: a test of
/// its eight neighbours alone.
fn deletable(mask: &[bool], w: usize, h: usize, x: isize, y: isize, pass: usize) -> bool {
    let at = |x: isize, y: isize| {
        x >= 0
            && y >= 0
            && (x as usize) < w
            && (y as usize) < h
            && mask[y as usize * w + x as usize]
    };
    let p2 = at(x, y - 1);
    let p3 = at(x + 1, y - 1);
    let p4 = at(x + 1, y);
    let p5 = at(x + 1, y + 1);
    let p6 = at(x, y + 1);
    let p7 = at(x - 1, y + 1);
    let p8 = at(x - 1, y);
    let p9 = at(x - 1, y - 1);
    let c = u8::from(!p2 && (p3 || p4))
        + u8::from(!p4 && (p5 || p6))
        + u8::from(!p6 && (p7 || p8))
        + u8::from(!p8 && (p9 || p2));
    let n1 = u8::from(p9 || p2) + u8::from(p3 || p4) + u8::from(p5 || p6) + u8::from(p7 || p8);
    let n2 = u8::from(p2 || p3) + u8::from(p4 || p5) + u8::from(p6 || p7) + u8::from(p8 || p9);
    let n = n1.min(n2);
    let m = if pass == 0 {
        (p6 || p7 || !p9) && p8
    } else {
        (p2 || p3 || !p5) && p4
    };
    c == 1 && (2..=3).contains(&n) && !m
}

/// Guo-Hall thinning of `mask` to one-pixel-wide lines, each sub-iteration deleting its pixels
/// together once found. A verdict reads only the eight neighbours, so after the first look a
/// sub-iteration looks again only beside pixels deleted since: time linear in the mask.
pub fn thin(mask: &mut [bool], w: usize, h: usize) {
    // per sub-iteration the set pixels to look at, and whether a pixel is listed
    let set: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();
    let mut todo = [set.clone(), set];
    let mut listed = [mask.to_vec(), mask.to_vec()];
    loop {
        let mut changed = false;
        for pass in 0..2 {
            let look = std::mem::take(&mut todo[pass]);
            for &i in &look {
                listed[pass][i] = false;
            }
            let remove: Vec<usize> = look
                .into_iter()
                .filter(|&i| {
                    mask[i] && deletable(mask, w, h, (i % w) as isize, (i / w) as isize, pass)
                })
                .collect();
            for &i in &remove {
                mask[i] = false;
            }
            for &i in &remove {
                let (x, y) = ((i % w) as isize, (i / w) as isize);
                for (dx, dy) in [
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                ] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    for p in 0..2 {
                        if mask[j] && !listed[p][j] {
                            listed[p][j] = true;
                            todo[p].push(j);
                        }
                    }
                }
            }
            changed |= !remove.is_empty();
        }
        if !changed {
            break;
        }
    }
}

/// The pixels of the shape, from its hierarchy region where it has one (which follows the
/// image, while a simplified outline may bulge) and its raster otherwise: the mask with a pixel
/// of margin, its width and height, and the image position of its top left pixel inside it.
pub(super) fn mask_of(
    shape: &Shape,
    hierarchy: Option<&Hierarchy>,
) -> (Vec<bool>, usize, usize, IVec2) {
    if let (Some(node), Some(hierarchy)) = (shape.node, hierarchy) {
        return node_mask(hierarchy, node);
    }
    let raster = shape.raster();
    let (w, h) = (raster.w + 2, raster.h + 2);
    let mut mask = vec![false; w * h];
    for y in 0..raster.h {
        for x in 0..raster.w {
            mask[(y + 1) * w + x + 1] = raster.values[y * raster.w + x] >= 0.5;
        }
    }
    (mask, w, h, IVec2::new(raster.x0 as i32, raster.y0 as i32))
}

/// The pixels of a region of the hierarchy, as [`mask_of`] has them.
pub(super) fn node_mask(hierarchy: &Hierarchy, node: u32) -> (Vec<bool>, usize, usize, IVec2) {
    let n = &hierarchy.nodes[node as usize];
    let (w, h) = (
        (n.max.x - n.min.x) as usize + 2,
        (n.max.y - n.min.y) as usize + 2,
    );
    let width = hierarchy.width();
    let mut mask = vec![false; w * h];
    for y in n.min.y..n.max.y {
        for x in n.min.x..n.max.x {
            let i = y as usize * width + x as usize;
            mask[(y - n.min.y + 1) as usize * w + (x - n.min.x + 1) as usize] =
                hierarchy.contains(node, i);
        }
    }
    (mask, w, h, n.min)
}

/// The skeleton of a shape's pixels (see [`mask_of`]): one chain or a network.
pub(super) fn skeletons_of_shape(
    shape: &Shape,
    hierarchy: Option<&Hierarchy>,
) -> Option<Skeletons> {
    let (mask, w, h, top_left) = mask_of(shape, hierarchy);
    SKELETONS.with_borrow_mut(|(known, px)| {
        let key = (top_left, w, h, mask);
        if let Some(found) = known.get(&key) {
            return found.clone();
        }
        let found = skeletons_of(&key.3, w, h, top_left);
        // the masks kept are bounded, forgotten all at once past the bound
        if *px + key.3.len() > MEMO_PX {
            known.clear();
            *px = 0;
        }
        *px += key.3.len();
        known.insert(key, found.clone());
        found
    })
}

/// How many mask pixels [`skeletons_of_shape`] remembers skeletons for.
const MEMO_PX: usize = 1 << 24;

type Memo = crate::HashMap<(IVec2, usize, usize, Vec<bool>), Option<Skeletons>>;

thread_local! {
    /// The skeletons of the masks seen, by mask: the branching test, the fit and the traces
    /// beneath a layer ask for the same shapes' skeletons again.
    static SKELETONS: std::cell::RefCell<(Memo, usize)> = std::cell::RefCell::default();
}

/// A shape's skeleton: one chain, or the lines of a network where it branches.
#[derive(Clone)]
pub(super) enum Skeletons {
    One(Skeleton),
    Network(Vec<Skeleton>),
}

/// The pixels set in `m` (`w` by `h`) among the eight around pixel `i`, edge neighbours first.
pub(super) fn neighbors(m: &[bool], w: usize, h: usize, i: usize) -> Vec<usize> {
    let (x, y) = ((i % w) as isize, (i / w) as isize);
    let mut out = Vec::new();
    for (dx, dy) in [
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
        (1, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
    ] {
        let (u, v) = (x + dx, y + dy);
        if u >= 0
            && v >= 0
            && (u as usize) < w
            && (v as usize) < h
            && m[v as usize * w + u as usize]
        {
            out.push(v as usize * w + u as usize);
        }
    }
    out
}

/// The set pixel of `m` a breadth-first walk from `from` (not through `without`) reaches
/// last, with each reached pixel's parent (`from` its own).
pub(super) fn farthest(
    m: &[bool],
    w: usize,
    h: usize,
    from: usize,
    without: Option<usize>,
) -> (usize, Vec<usize>) {
    let mut parent = vec![usize::MAX; w * h];
    let mut queue = std::collections::VecDeque::new();
    parent[from] = from;
    queue.push_back(from);
    let mut last = from;
    while let Some(i) = queue.pop_front() {
        last = i;
        for j in neighbors(m, w, h, i) {
            if parent[j] == usize::MAX && Some(j) != without {
                parent[j] = i;
                queue.push_back(j);
            }
        }
    }
    (last, parent)
}

/// The walk from `to` back along `parent` to the walk's start.
pub(super) fn walk_back(parent: &[usize], to: usize) -> Vec<usize> {
    let mut path = vec![to];
    let mut at = to;
    while parent[at] != at {
        at = parent[at];
        path.push(at);
    }
    path
}

/// The skeleton of a mask with a pixel of margin.
pub(super) fn skeletons_of(
    mask: &[bool],
    w: usize,
    h: usize,
    top_left: IVec2,
) -> Option<Skeletons> {
    if w < 3 || h < 3 {
        return None;
    }
    let distance = distance_squared(mask, w, h);
    let mut skeleton = mask.to_vec();
    thin(&mut skeleton, w, h);
    let neighbors = |m: &[bool], i: usize| neighbors(m, w, h, i);
    // Prune spurs: branches from an end shorter than the width there end at a cap or corner
    for _ in 0..3 {
        let ends: Vec<usize> = (0..w * h)
            .filter(|&i| skeleton[i] && neighbors(&skeleton, i).len() == 1)
            .collect();
        let mut pruned = false;
        for end in ends {
            let mut branch = vec![end];
            let mut current = end;
            let mut previous = usize::MAX;
            let junction = loop {
                let next: Vec<usize> = neighbors(&skeleton, current)
                    .into_iter()
                    .filter(|&j| j != previous && !branch.contains(&j))
                    .collect();
                if next.len() != 1 {
                    break next.len() > 1;
                }
                previous = current;
                current = next[0];
                if neighbors(&skeleton, current).len() > 2 {
                    break true;
                }
                branch.push(current);
                if branch.len() > 4 * w.max(h) {
                    break false;
                }
            };
            // A spur reaches from the axis to a corner of the outline, which lies at most the
            // miter limit times half the width away (the width at the end itself is small, so
            // the width where the branch joins the axis is the measure)
            let width = 2. * distance[current].sqrt() * MITER_LIMIT / 2.;
            if junction && (branch.len() as f32) < width.max(2.) {
                for i in &branch {
                    skeleton[*i] = false;
                }
                pruned = true;
            }
        }
        if !pruned {
            break;
        }
    }
    let pixels: Vec<usize> = (0..w * h).filter(|&i| skeleton[i]).collect();
    // A band under two widths long thins to forks toward its corners and little else, so its
    // chain runs along its principal axis instead
    let mut spread: Vec<f32> = pixels.iter().map(|&i| 2. * distance[i].sqrt()).collect();
    spread.sort_by(f32::total_cmp);
    let typical = spread.get(spread.len() / 2).copied().unwrap_or(0.);
    if pixels.len() < MIN_SKELETON || (pixels.len() as f32) < 2. * typical {
        return axis_chain(mask, &distance, w, h, top_left).map(Skeletons::One);
    }
    let ends: Vec<usize> = pixels
        .iter()
        .copied()
        .filter(|&i| neighbors(&skeleton, i).len() == 1)
        .collect();
    let closed = ends.is_empty();
    // The chain is the longest path through the skeleton: from an end to the pixel farthest
    // from it, and from there to the farthest again, which passes through the clusters the
    // thinning leaves at joints; around a loop, from a pixel's one neighbor to its other
    let farthest = |from: usize, without: Option<usize>| farthest(&skeleton, w, h, from, without);
    let path = |parent: &[usize], to: usize| {
        let mut path = walk_back(parent, to);
        path.reverse();
        path
    };
    let chain: Vec<usize> = if closed {
        let start = pixels
            .iter()
            .copied()
            .find(|&i| neighbors(&skeleton, i).len() == 2)
            .unwrap_or(pixels[0]);
        let around = neighbors(&skeleton, start);
        // A skeleton of one pixel has no neighbours to walk around
        if around.is_empty() {
            return None;
        }
        let (a, b) = (around[0], around[around.len() - 1]);
        let (_, parent) = farthest(a, Some(start));
        if parent[b] == usize::MAX {
            return None;
        }
        let mut chain = vec![start];
        chain.extend(path(&parent, b));
        chain
    } else {
        let (e1, _) = farthest(ends[0], None);
        let (e2, parent) = farthest(e1, None);
        path(&parent, e2)
    };
    if (chain.len() as f32) < CHAIN_SHARE * pixels.len() as f32 {
        // Much of the skeleton lies off the chain: a network of strokes, its lines
        let lines = network(&skeleton, &distance, w, h, top_left);
        return (!lines.is_empty()).then_some(Skeletons::Network(lines));
    }
    let widths = chain.iter().map(|&i| 2. * distance[i].sqrt()).collect();
    Some(Skeletons::One(Skeleton {
        chain: chain
            .iter()
            .map(|&i| IVec2::new((i % w) as i32 - 1, (i / w) as i32 - 1))
            .collect(),
        closed,
        widths,
        top_left,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guo-Hall by looking at every set pixel in each sub-iteration, as the definition reads.
    fn thin_by_scans(mask: &mut [bool], w: usize, h: usize) {
        loop {
            let mut changed = false;
            for pass in 0..2 {
                let remove: Vec<usize> = (0..w * h)
                    .filter(|&i| {
                        mask[i] && deletable(mask, w, h, (i % w) as isize, (i / w) as isize, pass)
                    })
                    .collect();
                for &i in &remove {
                    mask[i] = false;
                }
                changed |= !remove.is_empty();
            }
            if !changed {
                break;
            }
        }
    }

    /// Discs, bars and speckle at random sizes thin to the same pixels by both readings.
    #[test]
    fn thinning_looks_again_only_where_it_must() {
        let mut seed = 7u64;
        let mut next = |n: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % n
        };
        for case in 0..300 {
            let (w, h) = (8 + next(40) as usize, 8 + next(40) as usize);
            let mut mask = vec![false; w * h];
            for _ in 0..=next(4) {
                let (cx, cy, r) = (
                    next(w as u64) as f32,
                    next(h as u64) as f32,
                    1. + next(12) as f32,
                );
                let bar = next(2) == 0;
                for y in 0..h {
                    for x in 0..w {
                        let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                        let inside = if bar {
                            dx.abs() <= r && dy.abs() <= r / 3.
                        } else {
                            dx * dx + dy * dy <= r * r
                        };
                        mask[y * w + x] |= inside;
                    }
                }
            }
            if case % 3 == 0 {
                for _ in 0..w * h / 6 {
                    let i = next((w * h) as u64) as usize;
                    mask[i] = !mask[i];
                }
            }
            let (mut a, mut b) = (mask.clone(), mask);
            thin(&mut a, w, h);
            thin_by_scans(&mut b, w, h);
            assert_eq!(a, b, "case {case} ({w} x {h})");
        }
    }
}
