use glam::Vec2;

use crate::render::dual::{Dual, Real};
use crate::render::scene::{FillRule, FlatPoint};

/// Signed winding (fractional at edges) of one path over its clamped pixel bounding box.
#[derive(Clone, Debug, Default)]
pub struct Winding {
    pub x0: i32,
    pub y0: i32,
    pub w: usize,
    pub h: usize,
    pub values: Vec<f32>,
}

/// Which pixel a line contribution lands in. Everything left of the box is merged into one bucket,
/// since only its vertical extent (the cover) matters to pixels inside the box.
#[derive(Clone, Copy)]
pub(crate) enum Column {
    Left,
    Inside(usize),
}

/// Converts the winding of a pixel to its coverage under the fill rule, along with the derivative
/// d coverage / d winding used in the backward pass.
pub fn fill_coverage(winding: f32, rule: FillRule) -> (f32, f32) {
    match rule {
        // Doubly wound regions are saturated, so they have no gradient
        FillRule::NonZero if winding.abs() >= 1. => (1., 0.),
        FillRule::NonZero => (winding.abs(), winding.signum()),
        // A triangle wave with period 2
        FillRule::EvenOdd => {
            // Same as `rem_euclid(2.)` but without the slow `fmodf` call
            let folded = winding - 2. * (0.5 * winding).floor();
            if folded < 1. {
                (folded, 1.)
            } else {
                (2. - folded, -1.)
            }
        }
    }
}

/// Clips the line `a -> b` to `min <= p[axis] < max`, keeping its direction. Only boundary
/// crossings are interpolated, which keeps derivatives of nearly axis-aligned lines well
/// conditioned (clipping in parameter space loses them to cancellation).
#[inline]
fn clip<S: Real>(
    a: [S; 2],
    b: [S; 2],
    axis: usize,
    min: f32,
    max: f32,
) -> Option<([S; 2], [S; 2])> {
    let (av, bv) = (a[axis].val(), b[axis].val());
    // A line parallel to the boundaries is either fully inside or outside, so horizontal lines
    // keep their derivative with respect to being tilted
    if av == bv {
        return (av >= min && av < max).then_some((a, b));
    }
    if av.max(bv) <= min || av.min(bv) >= max {
        return None;
    }
    let other = 1 - axis;
    let at = |v: f32| {
        let mut p = [S::cst(v); 2];
        p[other] = a[other] + (S::cst(v) - a[axis]) * (b[other] - a[other]) / (b[axis] - a[axis]);
        p
    };
    let clamp = |p: [S; 2]| match p[axis].val() {
        v if v < min => at(min),
        v if v > max => at(max),
        _ => p,
    };
    Some((clamp(a), clamp(b)))
}

/// Walks the pixels a line touches inside the box and emits the signed trapezoid area to the right
/// of the line within the pixel, and the signed height (cover) the pixel passes on to everything
/// to its right. The winding of a pixel is its area plus the cover of all pixels left of it.
#[inline]
pub(crate) fn line_contributions<S: Real>(
    p0: [S; 2],
    p1: [S; 2],
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    mut emit: impl FnMut(usize, Column, S, S),
) {
    // Upward lines (in y-down coordinates) add to the winding, downward ones subtract
    let cover = |a: [S; 2], b: [S; 2]| a[1] - b[1];

    let (ymin, ymax) = (p0[1].val().min(p1[1].val()), p0[1].val().max(p1[1].val()));
    let row_start = (ymin.floor() as i32).max(y0);
    let row_end = (ymax.floor() as i32).min(y1 - 1);
    for row in row_start..=row_end {
        let Some((ra, rb)) = clip(p0, p1, 1, row as f32, row as f32 + 1.) else {
            continue;
        };
        let row_idx = (row - y0) as usize;

        let (xa, xb) = (ra[0].val(), rb[0].val());
        let col_start = xa.min(xb).floor() as i32;
        let col_end = (xa.max(xb).floor() as i32).min(x1 - 1);

        if col_start < x0
            && let Some((a, b)) = clip(ra, rb, 0, f32::NEG_INFINITY, x0 as f32)
        {
            emit(row_idx, Column::Left, S::cst(0.), cover(a, b));
        }
        for col in col_start.max(x0)..=col_end {
            let right = col as f32 + 1.;
            if let Some((a, b)) = clip(ra, rb, 0, col as f32, right) {
                let h = cover(a, b);
                let x_mid = (a[0] + b[0]) * S::cst(0.5);
                emit(
                    row_idx,
                    Column::Inside((col - x0) as usize),
                    h * (S::cst(right) - x_mid),
                    h,
                );
            }
        }
    }
}

pub(crate) fn lines(polys: &[Vec<FlatPoint>]) -> impl Iterator<Item = (&FlatPoint, &FlatPoint)> {
    polys
        .iter()
        .flat_map(|poly| poly.iter().zip(poly.iter().cycle().skip(1)))
}

/// The box `[x0, x1, y0, y1]` of the pixels the flattened path touches in a `width` x
/// `height` frame, or `None` when it misses the frame.
pub(crate) fn pixel_box(polys: &[Vec<FlatPoint>], width: i32, height: i32) -> Option<[i32; 4]> {
    let (min, max) = lines(polys).fold(
        (Vec2::INFINITY, Vec2::NEG_INFINITY),
        |(min, max), (p, _)| (min.min(p.pos), max.max(p.pos)),
    );
    let x0 = (min.x.floor() as i32).max(0);
    let y0 = (min.y.floor() as i32).max(0);
    let x1 = (max.x.floor() as i32 + 1).min(width);
    let y1 = (max.y.floor() as i32 + 1).min(height);
    (x1 > x0 && y1 > y0).then_some([x0, x1, y0, y1])
}

pub fn rasterize(polys: &[Vec<FlatPoint>], width: u32, height: u32) -> Winding {
    let Some([x0, x1, y0, y1]) = pixel_box(polys, width as i32, height as i32) else {
        return Winding::default();
    };
    let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);

    let mut values = vec![0.; w * h];
    // Cover per pixel, with an extra leading column for everything left of the box
    let mut cover = vec![0.; (w + 1) * h];
    for (a, b) in lines(polys) {
        line_contributions(
            a.pos.to_array(),
            b.pos.to_array(),
            x0,
            x1,
            y0,
            y1,
            |row, col, area, h| {
                let cover_idx = match col {
                    Column::Left => 0,
                    Column::Inside(col) => {
                        values[row * w + col] += area;
                        col + 1
                    }
                };
                cover[row * (w + 1) + cover_idx] += h;
            },
        );
    }

    for row in 0..h {
        let mut acc = 0.;
        for col in 0..w {
            acc += cover[row * (w + 1) + col];
            values[row * w + col] += acc;
        }
    }

    Winding {
        x0,
        y0,
        w,
        h,
        values,
    }
}

/// Backpropagates `d_winding` (laid out like `winding.values`) to the flattened points.
pub fn rasterize_backward(
    polys: &[Vec<FlatPoint>],
    winding: &Winding,
    d_winding: &[f32],
) -> Vec<Vec<Vec2>> {
    let Winding { x0, y0, w, h, .. } = *winding;
    let mut d_points: Vec<Vec<Vec2>> = polys
        .iter()
        .map(|poly| vec![Vec2::ZERO; poly.len()])
        .collect();
    if w == 0 {
        return d_points;
    }

    // The cover of a pixel reaches every pixel to its right, so its gradient is the suffix sum
    let mut d_cover = vec![0.; (w + 1) * h];
    for row in 0..h {
        let mut acc = 0.;
        for col in (0..w).rev() {
            acc += d_winding[row * w + col];
            d_cover[row * (w + 1) + col] = acc;
        }
    }

    let (x1, y1) = (x0 + w as i32, y0 + h as i32);
    for (poly, d_poly) in polys.iter().zip(&mut d_points) {
        let n = poly.len();
        for i in 0..n {
            let (a, b) = (poly[i].pos, poly[(i + 1) % n].pos);
            let p0 = [Dual::<4>::var(a.x, 0), Dual::var(a.y, 1)];
            let p1 = [Dual::var(b.x, 2), Dual::var(b.y, 3)];
            let mut d = [0.; 4];
            line_contributions(p0, p1, x0, x1, y0, y1, |row, col, area, h| {
                let (d_area, d_h) = match col {
                    Column::Left => (0., d_cover[row * (w + 1)]),
                    Column::Inside(col) => {
                        (d_winding[row * w + col], d_cover[row * (w + 1) + col + 1])
                    }
                };
                for (k, d) in d.iter_mut().enumerate() {
                    *d += area.d[k] * d_area + h.d[k] * d_h;
                }
            });
            d_poly[i] += Vec2::new(d[0], d[1]);
            d_poly[(i + 1) % n] += Vec2::new(d[2], d[3]);
        }
    }
    d_points
}
