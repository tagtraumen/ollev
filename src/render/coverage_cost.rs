//! The error of one solid path over a fixed backdrop, each pixel's a quadratic in coverage
//! (`a α² + b α + c`), from the outline's pixels and row prefix sums: equal to
//! [`crate::render::render_over`] and its backward pass, at a cost growing with the outline.

use glam::Vec2;

use crate::render::blur::Blurred;
use crate::render::dual::Dual;
use crate::render::raster::{Column, fill_coverage, line_contributions, lines, pixel_box};
use crate::render::scene::{FillRule, FlatPoint, Path, flatten};

pub struct CoverageCost {
    width: usize,
    height: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    /// Prefix sums of `a` and `b` along each row, `width + 1` per row.
    sum_a: Vec<f64>,
    sum_b: Vec<f64>,
    /// The error with nothing covered: the sum of `c`.
    constant: f64,
}

/// One line's contribution to a pixel, with its derivatives by the line's end points.
struct Contribution {
    line: u32,
    row: u32,
    col: i32,
    area: Dual<4>,
    cover: Dual<4>,
}

/// The buffers [`CoverageCost::evaluate`] fills, kept between calls on the thread.
#[derive(Default)]
struct Scratch {
    contributions: Vec<Contribution>,
    order: Vec<u32>,
    starts: Vec<u32>,
    cells: Vec<Cell>,
    cell_of: Vec<u32>,
    d_lines: Vec<[f32; 4]>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// A run of contributions to one pixel, or to everything left of the box (`col` -1).
#[derive(Clone, Copy, Default)]
struct Cell {
    row: u32,
    col: i32,
    area: f32,
    cover: f32,
    /// d error / d area and d error / d cover, filled in by the forward pass.
    d_area: f32,
    d_cover: f32,
}

impl CoverageCost {
    /// Costs `a[i] α² + b[i] α + c[i]` for the pixels of a `width` x `height` window, row-major.
    pub fn new(width: usize, height: usize, a: Vec<f32>, b: Vec<f32>, c: &[f32]) -> Self {
        let prefix = |v: &[f32]| {
            let mut out = Vec::with_capacity((width + 1) * height);
            for row in v.chunks(width.max(1)).take(height) {
                let mut acc = 0f64;
                out.push(0.);
                for &x in row {
                    acc += x as f64;
                    out.push(acc);
                }
            }
            out
        };
        Self {
            width,
            height,
            sum_a: prefix(&a),
            sum_b: prefix(&b),
            a,
            b,
            constant: c.iter().map(|&c| c as f64).sum(),
        }
    }

    /// The error with `path` drawn, its color aside.
    pub fn value(&self, path: &Path) -> f32 {
        if path.blur > 0. {
            return self.blurred(path, false).0;
        }
        let polys = flatten(path);
        let Some(bbox) = self.bbox(&polys) else {
            return self.constant as f32;
        };
        let [x0, x1, y0, y1] = bbox;
        let mut cells: Vec<Cell> = Vec::new();
        for (a, b) in lines(&polys) {
            line_contributions(
                a.pos.to_array(),
                b.pos.to_array(),
                x0,
                x1,
                y0,
                y1,
                |row, col, area, cover| cells.push(Cell::new(row, col, area, cover)),
            );
        }
        cells.sort_unstable_by_key(|c| (c.row, c.col));
        cells.dedup_by(|next, kept| {
            let same = next.row == kept.row && next.col == kept.col;
            if same {
                kept.area += next.area;
                kept.cover += next.cover;
            }
            same
        });
        self.sweep(&mut cells, bbox, path.fill_rule) as f32
    }

    /// The error with `path` drawn, its color aside, and its gradient with respect to the points
    /// of each subpath.
    pub fn evaluate(&self, path: &Path) -> (f32, Vec<Vec<Vec2>>) {
        if path.blur > 0. {
            let (value, d_points, _) = self.blurred(path, true);
            return (value, d_points);
        }
        let polys = flatten(path);
        let mut d_points = path.zero_points();
        let Some(bbox) = self.bbox(&polys) else {
            return (self.constant as f32, d_points);
        };
        let [x0, x1, y0, y1] = bbox;

        SCRATCH.with_borrow_mut(|scratch| {
            let Scratch {
                contributions,
                order,
                starts,
                cells,
                cell_of,
                d_lines,
            } = scratch;
            // Every line's contributions with their derivatives by the line's end points,
            // computed once, then bucketed by row and ordered by column within each row
            contributions.clear();
            let rows = (y1 - y0) as usize;
            starts.clear();
            starts.resize(rows + 1, 0);
            let mut line = 0u32;
            for poly in &polys {
                let n = poly.len();
                for i in 0..n {
                    let (a, b) = (poly[i].pos, poly[(i + 1) % n].pos);
                    let p0 = [Dual::<4>::var(a.x, 0), Dual::var(a.y, 1)];
                    let p1 = [Dual::var(b.x, 2), Dual::var(b.y, 3)];
                    line_contributions(p0, p1, x0, x1, y0, y1, |row, col, area, cover| {
                        let col = Cell::new(row, col, 0., 0.).col;
                        starts[row + 1] += 1;
                        contributions.push(Contribution {
                            line,
                            row: row as u32,
                            col,
                            area,
                            cover,
                        });
                    });
                    line += 1;
                }
            }
            for r in 0..rows {
                starts[r + 1] += starts[r];
            }
            order.clear();
            order.resize(contributions.len(), 0);
            {
                let mut next: Vec<u32> = starts[..rows].to_vec();
                for (k, c) in contributions.iter().enumerate() {
                    let at = &mut next[c.row as usize];
                    order[*at as usize] = k as u32;
                    *at += 1;
                }
            }
            for r in 0..rows {
                let slice = &mut order[starts[r] as usize..starts[r + 1] as usize];
                slice.sort_unstable_by_key(|&k| (contributions[k as usize].col, k));
            }

            // Merged per pixel, remembering which merged cell each contribution went to
            cells.clear();
            cell_of.clear();
            cell_of.resize(contributions.len(), 0);
            let mut last = (u32::MAX, i32::MIN);
            for &k in &*order {
                let c = &contributions[k as usize];
                if (c.row, c.col) != last {
                    cells.push(Cell {
                        row: c.row,
                        col: c.col,
                        ..Cell::default()
                    });
                    last = (c.row, c.col);
                }
                let cell = cells.last_mut().expect("just pushed");
                cell.area += c.area.v;
                cell.cover += c.cover.v;
                cell_of[k as usize] = (cells.len() - 1) as u32;
            }
            let error = self.sweep(cells, bbox, path.fill_rule);

            // Each contribution passes on its pixel's derivatives through its own
            d_lines.clear();
            d_lines.resize(line as usize, [0f32; 4]);
            for (c, &cell) in contributions.iter().zip(cell_of.iter()) {
                let cell = &cells[cell as usize];
                for (k, d) in d_lines[c.line as usize].iter_mut().enumerate() {
                    *d += c.area.d[k] * cell.d_area + c.cover.d[k] * cell.d_cover;
                }
            }
            let mut line = 0;
            for (poly, subpath) in polys.iter().zip(&mut d_points) {
                let n = poly.len();
                for i in 0..n {
                    let d = d_lines[line];
                    line += 1;
                    let (a, b) = (&poly[i], &poly[(i + 1) % n]);
                    for (point, grad) in [(a, Vec2::new(d[0], d[1])), (b, Vec2::new(d[2], d[3]))] {
                        for (&src, &weight) in point.src.iter().zip(&point.weights) {
                            subpath[src as usize] += weight * grad;
                        }
                    }
                }
            }
            (error as f32, d_points)
        })
    }

    /// The error with the blurred `path` drawn (see [`crate::render::blur`]), its color aside,
    /// over the box its blur reaches; with `gradient`, its gradient by the points of each
    /// subpath and by the blur's σ.
    pub fn blurred(&self, path: &Path, gradient: bool) -> (f32, Vec<Vec<Vec2>>, f32) {
        let polys = flatten(path);
        let blurred = Blurred::new(polys, path.fill_rule, path.blur, self.width, self.height);
        let mut error = self.constant;
        let mut d_values = Vec::with_capacity(if gradient { blurred.values.len() } else { 0 });
        for y in 0..blurred.h {
            let row = (blurred.y0 + y) * self.width + blurred.x0;
            let values = &blurred.values[y * blurred.w..(y + 1) * blurred.w];
            for (x, &alpha) in values.iter().enumerate() {
                let (a, b) = (self.a[row + x], self.b[row + x]);
                error += (alpha * (alpha * a + b)) as f64;
                if gradient {
                    d_values.push(2. * alpha * a + b);
                }
            }
        }
        if !gradient {
            return (error as f32, Vec::new(), 0.);
        }
        let (d_flat, d_sigma) = blurred.backward(&d_values);
        let d_points = path.points_gradient(&blurred.polys, &d_flat);
        (error as f32, d_points, d_sigma)
    }

    /// The box `[x0, x1, y0, y1]` the flattened path's pixels lie in, as the rasterizer clamps
    /// it, or `None` when nothing of it lies in the window.
    fn bbox(&self, polys: &[Vec<FlatPoint>]) -> Option<[i32; 4]> {
        pixel_box(polys, self.width as i32, self.height as i32)
    }

    /// The error with the merged contributions `cells` drawn, sorted by row and column, filling
    /// in their derivatives of the error by area and cover.
    fn sweep(&self, cells: &mut [Cell], [x0, x1, y0, _]: [i32; 4], rule: FillRule) -> f64 {
        // Forward along each row: runs between cells keep the winding they start with
        let width = (x1 - x0) as usize;
        let mut error = self.constant;
        let mut start = 0;
        let mut d_runs: Vec<f32> = Vec::new();
        while start < cells.len() {
            let row = cells[start].row;
            let end = start + cells[start..].partition_point(|c| c.row == row);
            let image_row = (y0 as usize + row as usize) * (self.width + 1);
            let pixel_row = (y0 as usize + row as usize) * self.width;
            let span = |from: usize, to: usize| {
                let (p, q) = (image_row + x0 as usize + from, image_row + x0 as usize + to);
                (self.sum_a[q] - self.sum_a[p], self.sum_b[q] - self.sum_b[p])
            };
            // d error / d winding of each cell and, before it, of the run left of it
            d_runs.clear();
            let mut acc = 0f32;
            let mut next = 0usize;
            for cell in &mut cells[start..end] {
                if cell.col < 0 {
                    acc += cell.cover;
                    d_runs.push(0.);
                    continue;
                }
                let col = cell.col as usize;
                let (sa, sb) = span(next, col);
                let (alpha, slope) = fill_coverage(acc, rule);
                let (alpha, slope) = (alpha as f64, slope as f64);
                error += alpha * alpha * sa + alpha * sb;
                d_runs.push((slope * (2. * alpha * sa + sb)) as f32);

                let i = pixel_row + x0 as usize + col;
                let winding = acc + cell.area;
                let (alpha, slope) = fill_coverage(winding, rule);
                let (a, b) = (self.a[i], self.b[i]);
                error += (alpha * alpha * a + alpha * b) as f64;
                cell.d_area = slope * (2. * alpha * a + b);
                acc += cell.cover;
                next = col + 1;
            }
            let (sa, sb) = span(next, width);
            let (alpha, slope) = fill_coverage(acc, rule);
            let (alpha, slope) = (alpha as f64, slope as f64);
            error += alpha * alpha * sa + alpha * sb;
            let mut suffix = (slope * (2. * alpha * sa + sb)) as f32;

            // Backward: a cell's cover reaches every pixel right of it
            for (cell, d_run) in cells[start..end].iter_mut().zip(&d_runs).rev() {
                cell.d_cover = suffix;
                if cell.col >= 0 {
                    suffix += cell.d_area + d_run;
                }
            }
            start = end;
        }
        error
    }
}

impl Cell {
    fn new(row: usize, col: Column, area: f32, cover: f32) -> Self {
        Self {
            row: row as u32,
            col: match col {
                Column::Left => -1,
                Column::Inside(col) => col as i32,
            },
            area,
            cover,
            d_area: 0.,
            d_cover: 0.,
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::render::scene::{Scene, Subpath};

    /// The quadratic error and its gradient must match rendering the path over the backdrop and
    /// backpropagating through every pixel.
    #[test]
    fn matches_dense_render() {
        let (w, h) = (37usize, 29usize);
        let mut seed = 7u32;
        let mut random = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let color = Vec3::new(0.8, 0.3, 0.1);
        let below: Vec<Vec3> = (0..w * h)
            .map(|_| Vec3::new(random(), random(), random()))
            .collect();
        let target: Vec<Vec3> = (0..w * h)
            .map(|_| Vec3::new(random(), random(), random()))
            .collect();
        let weight: Vec<f32> = (0..w * h).map(|_| random()).collect();
        // A star with a hole, partly outside the window
        let star: Vec<Vec2> = (0..14)
            .map(|k| {
                let angle = k as f32 * std::f32::consts::TAU / 14.;
                let radius = if k % 2 == 0 { 17. } else { 8. };
                Vec2::new(15., 13.) + radius * Vec2::from_angle(angle)
            })
            .collect();
        let hole: Vec<Vec2> = [(12., 10.), (12., 16.), (18., 16.), (18., 10.)]
            .map(Vec2::from)
            .to_vec();
        let path = Path {
            subpaths: vec![Subpath::polygon(&star), Subpath::polygon(&hole)],
            color: [color.x, color.y, color.z, 1.],
            gradient: None,
            fill_rule: FillRule::EvenOdd,
            blur: 0.,
        };
        let scene = Scene {
            width: w as u32,
            height: h as u32,
            paths: vec![path.clone()],
        };
        let (image, tape) = crate::render::backward::render_taped(&scene, below.clone());
        let mut dense = 0f64;
        let d_image: Vec<Vec3> = (0..w * h)
            .map(|i| {
                let d = image[i] - target[i];
                dense += (weight[i] * d.length_squared()) as f64;
                2. * weight[i] * d
            })
            .collect();
        let dense_grad = crate::render::backward::backward(&scene, &tape, &d_image);

        let a = (0..w * h)
            .map(|i| weight[i] * (color - below[i]).length_squared())
            .collect();
        let b = (0..w * h)
            .map(|i| 2. * weight[i] * (below[i] - target[i]).dot(color - below[i]))
            .collect();
        let c: Vec<f32> = (0..w * h)
            .map(|i| weight[i] * (below[i] - target[i]).length_squared())
            .collect();
        let cost = CoverageCost::new(w, h, a, b, &c);
        let (value, grad) = cost.evaluate(&path);
        assert_eq!(value, cost.value(&path));

        assert!(
            (value as f64 - dense).abs() < 1e-3 * dense,
            "{value} vs {dense}"
        );
        for (sparse, dense) in grad.iter().zip(&dense_grad.paths[0].subpaths) {
            for (s, d) in sparse.iter().zip(dense) {
                assert!((*s - *d).length() < 1e-3 * (1. + d.length()), "{s} vs {d}");
            }
        }
    }
}
