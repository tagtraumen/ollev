//! Outlines and polylines: crack loops, smoothing, resampling, a polyline's coverage.

use glam::DVec2;

use super::{Bezier, bez, fit_cubics};

/// The outline of a mask as closed loops through the midpoints of the cracks between pixels
/// in and out (SVG units), the mask on each loop's left (y down); pixels touching only
/// diagonally are kept apart.
pub fn crack_loops(mask: &[bool], width: usize, height: usize) -> Vec<Vec<DVec2>> {
    let inside = |x: isize, y: isize| {
        x >= 0
            && y >= 0
            && (x as usize) < width
            && (y as usize) < height
            && mask[y as usize * width + x as usize]
    };
    // directed cracks from corner to corner: a pixel in with its neighbour out gives the
    // crack between them, directed so the pixel lies on its left (y down: left of the
    // direction (dx, dy) is (dy, -dx))
    let corners = (width + 1) * (height + 1);
    let cid = |x: usize, y: usize| y * (width + 1) + x;
    let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); corners];
    for y in 0..height as isize {
        for x in 0..width as isize {
            if !inside(x, y) {
                continue;
            }
            let (xu, yu) = (x as usize, y as usize);
            // the four sides in one turning order (top right to left, bottom left to
            // right, left side down, right side up), so the cracks of a component chain
            // head to tail
            if !inside(x, y - 1) {
                out_edges[cid(xu + 1, yu)].push(cid(xu, yu));
            }
            if !inside(x, y + 1) {
                out_edges[cid(xu, yu + 1)].push(cid(xu + 1, yu + 1));
            }
            if !inside(x - 1, y) {
                out_edges[cid(xu, yu)].push(cid(xu, yu + 1));
            }
            if !inside(x + 1, y) {
                out_edges[cid(xu + 1, yu + 1)].push(cid(xu + 1, yu));
            }
        }
    }
    let mut used: Vec<Vec<bool>> = out_edges.iter().map(|e| vec![false; e.len()]).collect();
    let pos = |c: usize| DVec2::new((c % (width + 1)) as f64, (c / (width + 1)) as f64);
    let mut loops = Vec::new();
    for start in 0..corners {
        for e0 in 0..out_edges[start].len() {
            if used[start][e0] {
                continue;
            }
            let mut pts = Vec::new();
            let (mut cur, mut e) = (start, e0);
            loop {
                used[cur][e] = true;
                let nxt = out_edges[cur][e];
                pts.push((pos(cur) + pos(nxt)) / 2.0);
                // the next crack from nxt: of two, the one turning toward the mask (left
                // in image coordinates) keeps diagonal pixels apart
                let d = pos(nxt) - pos(cur);
                let mut pick = None;
                for (j, &q) in out_edges[nxt].iter().enumerate() {
                    if used[nxt][j] {
                        continue;
                    }
                    let d2 = pos(q) - pos(nxt);
                    // cross > 0 turns right on screen (y down); prefer the left turn
                    let cross = d.x * d2.y - d.y * d2.x;
                    let rank = if cross < 0.0 {
                        0
                    } else if cross == 0.0 {
                        1
                    } else {
                        2
                    };
                    if pick.is_none_or(|(_, r)| rank < r) {
                        pick = Some((j, rank));
                    }
                }
                match pick {
                    Some((j, _)) => {
                        cur = nxt;
                        e = j;
                    }
                    None => break,
                }
            }
            if pts.len() >= 4 {
                loops.push(pts);
            }
        }
    }
    loops
}

/// A closed loop smoothed along its arc by a Gaussian of `s` (the cracks' staircase is the
/// grid's, not the shape's).
pub fn smooth_loop(p: &[DVec2], s: f64) -> Vec<DVec2> {
    let n = p.len();
    if n < 3 || s <= 0.0 {
        return p.to_vec();
    }
    let r = (3.0 * s).ceil() as isize;
    let w: Vec<f64> = (-r..=r)
        .map(|i| (-0.5 * (i * i) as f64 / (s * s)).exp())
        .collect();
    let tot: f64 = w.iter().sum();
    (0..n as isize)
        .map(|i| {
            let mut a = DVec2::ZERO;
            for (j, wj) in (-r..=r).zip(&w) {
                a += *wj * p[(i + j).rem_euclid(n as isize) as usize];
            }
            a / tot
        })
        .collect()
}

/// A closed loop as cubics: cut into four arcs that share their ends (the open fit takes a
/// loop whose ends coincide for a point), each fitted.
pub fn fit_loop(p: &[DVec2], tol: f64) -> Vec<Bezier> {
    let n = p.len();
    if n < 4 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for q in 0..4 {
        let (a, b) = (q * n / 4, (q + 1) * n / 4);
        let mut arc: Vec<DVec2> = (a..=b).map(|i| p[i % n]).collect();
        arc.dedup();
        out.extend(fit_cubics(&arc, tol));
    }
    out
}

/// A closed polyline resampled at a spacing along its arc, back to its start.
pub fn resample(p: &[DVec2], step: f64) -> Vec<DVec2> {
    let mut pts = p.to_vec();
    if !pts.is_empty() {
        pts.push(pts[0]);
    }
    let mut arc = vec![0.0];
    for i in 1..pts.len() {
        arc.push(arc[i - 1] + (pts[i] - pts[i - 1]).length());
    }
    let total = *arc.last().unwrap_or(&0.0);
    if total <= 0.0 {
        return p.to_vec();
    }
    let n = (total / step).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(n + 1);
    let mut j = 0;
    for i in 0..n {
        let s = total * i as f64 / n as f64;
        while j + 1 < arc.len() - 1 && arc[j + 1] < s {
            j += 1;
        }
        let seg = (arc[j + 1] - arc[j]).max(1e-12);
        let t = ((s - arc[j]) / seg).clamp(0.0, 1.0);
        out.push(pts[j] + t * (pts[j + 1] - pts[j]));
    }
    out
}

/// Unit normals of a polyline (left of the direction in image coordinates).
pub(super) fn normals(p: &[DVec2]) -> Vec<DVec2> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let d = p[(i + 1).min(n - 1)] - p[i.saturating_sub(1)];
            let d = d / d.length().max(1e-9);
            DVec2::new(-d.y, d.x)
        })
        .collect()
}

/// Points along cubics, about one every half pixel.
pub fn flatten(cubics: &[Bezier]) -> Vec<DVec2> {
    let mut out = Vec::new();
    for c in cubics {
        let len = (c[1] - c[0]).length() + (c[2] - c[1]).length() + (c[3] - c[2]).length();
        let n = ((len / 0.5).ceil() as usize).max(1);
        let from = usize::from(!out.is_empty());
        for i in from..=n {
            out.push(bez(c, i as f64 / n as f64));
        }
    }
    out
}

/// A window of the image: columns x0..x1, rows y0..y1.
#[derive(Clone, Copy, Debug)]
pub(super) struct Win {
    pub(super) x0: usize,
    pub(super) y0: usize,
    pub(super) x1: usize,
    pub(super) y1: usize,
}

impl Win {
    pub(super) fn around(p: &[DVec2], pad: f64, width: usize, height: usize) -> Win {
        let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
        for q in p {
            lo = lo.min(*q);
            hi = hi.max(*q);
        }
        let x0 = (lo.x - pad).floor().max(0.0) as usize;
        let y0 = (lo.y - pad).floor().max(0.0) as usize;
        let x1 = ((hi.x + pad).ceil().max(0.0) as usize).min(width);
        let y1 = ((hi.y + pad).ceil().max(0.0) as usize).min(height);
        Win {
            x0,
            y0,
            x1: x1.max(x0),
            y1: y1.max(y0),
        }
    }
    pub(super) fn w(&self) -> usize {
        self.x1 - self.x0
    }
    pub(super) fn h(&self) -> usize {
        self.y1 - self.y0
    }
    pub(super) fn len(&self) -> usize {
        self.w() * self.h()
    }
    /// the image index of the window's i-th pixel
    pub(super) fn at(&self, i: usize, width: usize) -> usize {
        (self.y0 + i / self.w()) * width + self.x0 + i % self.w()
    }
}

/// The anti-aliased coverage of a stroke of `width` along a polyline over a window: the
/// half width less the pixel centre's distance, plus a half, clipped (the box filter's
/// approximation of the renderer).
pub(super) fn coverage(p: &[DVec2], width: f64, win: Win) -> Vec<f64> {
    // each segment is measured only near its bounding box: a pixel farther than the half
    // width and a half (plus slack) has zero coverage from it, so the result is exact
    let (ww, wh) = (win.w(), win.h());
    let centre = |i: usize| {
        DVec2::new(
            (win.x0 + i % ww) as f64 + 0.5,
            (win.y0 + i / ww) as f64 + 0.5,
        )
    };
    let mut dist = vec![f64::INFINITY; win.len()];
    if p.len() == 1 {
        for (i, d) in dist.iter_mut().enumerate() {
            *d = (centre(i) - p[0]).length();
        }
    }
    let reach = width / 2.0 + 1.5;
    // the window's columns or rows within reach of [a, b], all of them when that is not finite
    let span = |a: f64, b: f64, origin: usize, len: usize| -> (usize, usize) {
        let (a, b) = (a - reach - 1.0, b + reach + 1.0);
        if !(a.is_finite() && b.is_finite()) {
            return (0, len);
        }
        let from = (a.floor() - origin as f64).clamp(0.0, len as f64) as usize;
        let to = (b.ceil() - origin as f64).clamp(0.0, len as f64) as usize;
        (from, to.max(from))
    };
    for s in p.windows(2) {
        let (lo, hi) = (s[0].min(s[1]), s[0].max(s[1]));
        let (xa, xb) = span(lo.x, hi.x, win.x0, ww);
        let (ya, yb) = span(lo.y, hi.y, win.y0, wh);
        let ab = s[1] - s[0];
        for y in ya..yb {
            for x in xa..xb {
                let i = y * ww + x;
                let q = centre(i);
                let t = ((q - s[0]).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
                dist[i] = dist[i].min((q - (s[0] + t * ab)).length());
            }
        }
    }
    dist.iter()
        .map(|d| (width / 2.0 - d + 0.5).clamp(0.0, 1.0))
        .collect()
}
