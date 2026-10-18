//! The grain: lines (three times as long as wide, two units to [`WIDEST_LINE`] wide) whose shared
//! boundary runs half the shorter's length are one object, slivers joined to their lines, so a
//! lock of hair traced as many strands is one occluder with one contour.

use std::collections::BTreeMap;

use super::raster::{area_px, unit_of};

/// The widest a line is, in units: twice the thin cue's reach ([`super::above`]), so a lock
/// of niabot's hair (7 units at 660) is still a line.
pub const WIDEST_LINE: f64 = 8.0;

/// The grain read on the opaque scene's shapes, per pixel.
pub struct Grain {
    /// the shape label with the slivers absorbed into their lines: every pixel keeps a label
    pub lines: Vec<u32>,
    /// the grain object (the label of one of its lines), `u32::MAX` off the grain
    pub objects: Vec<u32>,
}

/// The chamfer distance from each pixel to the nearest boundary pixel of its shape, in
/// pixels (the 3-4 chamfer over three); a boundary pixel is at 0.
fn boundary_distance(shapes: &[u32], w: usize, h: usize) -> Vec<f64> {
    let big = (3 * (w + h)) as u32;
    let mut d: Vec<u32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let l = shapes[i];
            let boundary = (x > 0 && shapes[i - 1] != l)
                || (x + 1 < w && shapes[i + 1] != l)
                || (y > 0 && shapes[i - w] != l)
                || (y + 1 < h && shapes[i + w] != l);
            if boundary { 0 } else { big }
        })
        .collect();
    // forward pass: from the left, the top and the two upper diagonals
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mut v = d[i];
            if x > 0 {
                v = v.min(d[i - 1] + 3);
            }
            if y > 0 {
                v = v.min(d[i - w] + 3);
                if x > 0 {
                    v = v.min(d[i - w - 1] + 4);
                }
                if x + 1 < w {
                    v = v.min(d[i - w + 1] + 4);
                }
            }
            d[i] = v;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            let mut v = d[i];
            if x + 1 < w {
                v = v.min(d[i + 1] + 3);
            }
            if y + 1 < h {
                v = v.min(d[i + w] + 3);
                if x + 1 < w {
                    v = v.min(d[i + w + 1] + 4);
                }
                if x > 0 {
                    v = v.min(d[i + w - 1] + 4);
                }
            }
            d[i] = v;
        }
    }
    d.iter().map(|&v| v as f64 / 3.0).collect()
}

/// The grain of `shapes` (a label per pixel, dense from 0) outside `region`.
pub fn grain_objects(shapes: &[u32], region: &[bool], w: usize, h: usize) -> Grain {
    let n = shapes.iter().max().map_or(0, |&l| l as usize + 1);
    let unit = unit_of(w);
    let dist = boundary_distance(shapes, w, h);
    let mut px = vec![0usize; n];
    let mut inside = vec![0usize; n];
    let mut interior = vec![0usize; n];
    let mut radius = vec![0.0f64; n];
    for i in 0..w * h {
        let l = shapes[i] as usize;
        px[l] += 1;
        if region[i] {
            inside[l] += 1;
        }
        if dist[i] > unit {
            interior[l] += 1;
        }
        radius[l] = radius[l].max(dist[i]);
    }
    // the shared boundary of each pair of shapes, in cracks; a sorted map, so the passes below
    // (order dependent: a root's length is the root shape's own) do not depend on a hash seed
    let mut shared: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    for i in 0..w * h {
        let (x, y) = (i % w, i / w);
        for j in [(x + 1 < w).then_some(i + 1), (y + 1 < h).then_some(i + w)]
            .into_iter()
            .flatten()
        {
            if shapes[i] != shapes[j] {
                let key = (shapes[i].min(shapes[j]), shapes[i].max(shapes[j]));
                *shared.entry(key).or_default() += 1;
            }
        }
    }
    let least = area_px(w, 4.0);
    let width: Vec<f64> = (0..n).map(|l| 2.0 * radius[l].max(0.5)).collect();
    let length: Vec<f64> = (0..n).map(|l| px[l] as f64 / width[l]).collect();
    let line: Vec<bool> = (0..n)
        .map(|l| {
            px[l] as f64 >= least
                && length[l] >= 3.0 * width[l]
                && width[l] >= 2.0 * unit
                && width[l] <= WIDEST_LINE * unit
                && 2 * inside[l] < px[l]
        })
        .collect();
    let sliver: Vec<bool> = (0..n)
        .map(|l| (interior[l] as f64) < least && !line[l])
        .collect();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(parent: &mut [u32], mut a: u32) -> u32 {
        while parent[a as usize] != a {
            let p = parent[a as usize];
            parent[a as usize] = parent[p as usize];
            a = p;
        }
        a
    }
    // the slivers, each into the line it shares the most boundary with
    let mut best: Vec<Option<(usize, u32)>> = vec![None; n];
    for (&(a, b), &c) in &shared {
        for (s, l) in [(a, b), (b, a)] {
            if sliver[s as usize]
                && line[l as usize]
                && best[s as usize].is_none_or(|(bc, _)| c > bc)
            {
                best[s as usize] = Some((c, l));
            }
        }
    }
    for (s, b) in best.iter().enumerate() {
        if let Some((_, l)) = *b {
            let (rs, rl) = (find(&mut parent, s as u32), find(&mut parent, l));
            if rs != rl {
                parent[rs as usize] = rl;
            }
        }
    }
    let lines: Vec<u32> = shapes.iter().map(|&l| find(&mut parent, l)).collect();
    // the lines alongside each other, until nothing merges
    loop {
        let mut by_root: BTreeMap<(u32, u32), usize> = BTreeMap::new();
        for (&(a, b), &c) in &shared {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb && line[ra as usize] && line[rb as usize] {
                *by_root.entry((ra.min(rb), ra.max(rb))).or_default() += c;
            }
        }
        // the longest shared boundaries first: the strongest adjacency decides the objects
        let mut pairs: Vec<((u32, u32), usize)> = by_root.into_iter().collect();
        pairs.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
        let mut merged = 0;
        for ((ra, rb), c) in pairs {
            if c as f64 >= 0.5 * length[ra as usize].min(length[rb as usize]) {
                let (pa, pb) = (find(&mut parent, ra), find(&mut parent, rb));
                if pa != pb {
                    parent[pb as usize] = pa;
                    merged += 1;
                }
            }
        }
        if merged == 0 {
            break;
        }
    }
    let objects: Vec<u32> = shapes
        .iter()
        .map(|&l| {
            let r = find(&mut parent, l);
            if line[r as usize] { r } else { u32::MAX }
        })
        .collect();
    Grain { lines, objects }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two strands side by side with a dark outline between them are one grain; a square
    /// beside them is none; a strand crossing them stays its own object.
    #[test]
    fn strands_alongside_are_one_grain_and_a_square_is_none() {
        let (w, h) = (330usize, 120usize);
        let mut shapes = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                shapes[i] = if (20..100).contains(&x) && (30..38).contains(&y) {
                    1
                } else if (20..100).contains(&x) && y == 38 {
                    2
                } else if (20..100).contains(&x) && (39..47).contains(&y) {
                    3
                } else if (150..190).contains(&x) && (20..60).contains(&y) {
                    4
                } else if (60..66).contains(&x) && (10..70).contains(&y) {
                    5
                } else {
                    0
                };
            }
        }
        let region = vec![false; w * h];
        let g = grain_objects(&shapes, &region, w, h);
        let at = |x: usize, y: usize| g.objects[y * w + x];
        assert_ne!(at(30, 33), u32::MAX, "the upper strand is a line");
        assert_eq!(
            at(30, 33),
            at(30, 43),
            "the strands alongside are one grain"
        );
        assert_eq!(at(30, 33), at(30, 38), "the outline joins its strand");
        assert_eq!(at(170, 40), u32::MAX, "the square is no line");
        assert_ne!(at(62, 15), u32::MAX, "the crossing strand is a line");
        assert_ne!(
            at(62, 15),
            at(30, 33),
            "the crossing strand is its own object"
        );
        assert_eq!(g.lines[38 * w + 30], g.lines[33 * w + 30]);
    }
}
