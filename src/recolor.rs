//! The final coloring: from the top down, every shape takes the fill of the pixels it shows
//! (see [`crate::fill::fit`]), or is removed where the shapes below show them about as well
//! for less than its segments and fill cost.

use glam::{Vec2, Vec3};

use crate::hierarchy::Stats;
use crate::{Context, Scene, Shape};

/// How much of a pixel a shape covers, over the shape's bounding box.
struct Coverage {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    values: Vec<f32>,
}

impl Coverage {
    /// The shape's raster, cropped to the image.
    fn of(shape: &Shape, width: usize, height: usize) -> Self {
        let r = shape.raster();
        let w = (r.x0 + r.w).min(width).saturating_sub(r.x0);
        let h = (r.y0 + r.h).min(height).saturating_sub(r.y0);
        Self {
            x0: r.x0,
            y0: r.y0,
            w,
            h,
            values: (0..h)
                .flat_map(|y| r.values[y * r.w..y * r.w + w].iter().copied())
                .collect(),
        }
    }

    /// Image indices and coverage of the pixels in the box.
    fn pixels(&self, width: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        (0..self.h).flat_map(move |y| {
            (0..self.w).map(move |x| {
                let i = (self.y0 + y) * width + self.x0 + x;
                (i, self.values[y * self.w + x])
            })
        })
    }

    fn at(&self, i: usize, width: usize) -> f32 {
        let (x, y) = (i % width, i / width);
        if x < self.x0 || y < self.y0 || x >= self.x0 + self.w || y >= self.y0 + self.h {
            return 0.;
        }
        self.values[(y - self.y0) * self.w + x - self.x0]
    }
}

pub fn recolor(ctx: &Context, scene: &mut Scene) {
    let (width, height) = (ctx.width, ctx.height);
    // What each shape covers once clipped, its clip coming before it
    let mut coverage: Vec<Coverage> = Vec::with_capacity(scene.shapes.len());
    for shape in &scene.shapes {
        let mut own = Coverage::of(shape, width, height);
        if let Some(clip) = shape.clip {
            let clip = &coverage[scene.position(clip)];
            let indices: Vec<usize> = own.pixels(width).map(|(i, _)| i).collect();
            for (v, i) in own.values.iter_mut().zip(indices) {
                *v *= clip.at(i, width);
            }
        }
        coverage.push(own);
    }
    let target = |i: usize| ctx.target[i].as_dvec3();
    // The coverage of the kept shapes above the one at hand, per pixel
    let mut above = vec![0f32; width * height];
    let mut keep = vec![true; scene.shapes.len()];
    for k in (0..scene.shapes.len()).rev() {
        if crate::progress::poll() {
            break;
        }
        // Pixels the shape shows, and those it shows without anti-aliasing, which give the color
        let mut shown = Vec::new();
        let (mut all, mut interior) = (Stats::default(), Stats::default());
        for (i, c) in coverage[k].pixels(width) {
            if c >= 0.5 && above[i] < 0.5 {
                shown.push(i);
                all = all.add(&Stats::pixel(target(i)));
                if c > 0.99 && above[i] < 0.01 {
                    interior = interior.add(&Stats::pixel(target(i)));
                }
            }
        }
        let stats = if interior.n > 0. { interior } else { all };
        // The fill is fitted away from the rim, see [`crate::fill::erosion`]: each shown
        // pixel's depth is how far around it everything is fully shown
        let sampled: Vec<(Vec2, Vec3, usize)> = if interior.n > 0. {
            let c = &coverage[k];
            let full = |x: isize, y: isize| {
                x < 0
                    || y < 0
                    || x >= width as isize
                    || y >= height as isize
                    || (x as usize >= c.x0
                        && y as usize >= c.y0
                        && (x as usize) < c.x0 + c.w
                        && (y as usize) < c.y0 + c.h
                        && c.values[(y as usize - c.y0) * c.w + x as usize - c.x0] > 0.99
                        && above[y as usize * width + x as usize] < 0.01)
            };
            let g = crate::fill::RIM + 1;
            let (gx0, gy0) = (c.x0.saturating_sub(g), c.y0.saturating_sub(g));
            let (gx1, gy1) = ((c.x0 + c.w + g).min(width), (c.y0 + c.h + g).min(height));
            let d = full_depths(&full, [gx0, gy0, gx1, gy1], crate::fill::RIM);
            let depths: Vec<usize> = shown
                .iter()
                .map(|&i| d[(i / width - gy0) * (gx1 - gx0) + i % width - gx0])
                .collect();
            let mut counts = [0usize; crate::fill::RIM + 2];
            for &d in &depths {
                counts[d] += 1;
            }
            let erosion = crate::fill::erosion(&counts);
            shown
                .iter()
                .zip(&depths)
                .filter(|&(_, &d)| d > erosion)
                .map(|(&i, _)| {
                    let p = Vec2::new((i % width) as f32 + 0.5, (i / width) as f32 + 0.5);
                    (p, ctx.target[i], i)
                })
                .collect()
        } else {
            Vec::new()
        };
        let samples: Vec<(Vec2, Vec3)> = sampled.iter().map(|&(p, c, _)| (p, c)).collect();
        let tolerance: Vec<f32> = if ctx.tolerance.is_some() {
            sampled
                .iter()
                .map(|&(_, _, i)| ctx.tolerance_at(i))
                .collect()
        } else {
            Vec::new()
        };
        let costs = crate::fill::Costs::at(ctx.segment_cost);
        let center = scene.shapes[k].primitive_center();
        let extent: Vec<Vec2> = shown
            .iter()
            .map(|&i| Vec2::new((i % width) as f32 + 0.5, (i / width) as f32 + 0.5))
            .collect();
        let (fill, _) = crate::fill::fit(
            &samples,
            &tolerance,
            stats.mean().as_vec3(),
            center,
            costs,
            &extent,
        );
        let color = fill.color;
        if k > 0 {
            // Without the shape, its pixels show the kept shape below
            let below = |i: usize| {
                let (x, y) = (i % width, i / width);
                (0..k)
                    .rev()
                    .find(|&j| keep[j] && coverage[j].at(i, width) >= 0.5)
                    .map_or(scene.shapes[0].fill.at_pixel(x, y), |j| {
                        scene.shapes[j].fill.at_pixel(x, y)
                    })
            };
            let error = |i: usize, c: Vec3| {
                let d2 = ctx.distance_squared(ctx.target[i], c);
                crate::uncertainty::excess(d2, ctx.tolerance_at(i)) as f64
            };
            let saving: f64 = shown
                .iter()
                .map(|&i| error(i, below(i)) - error(i, fill.at_pixel(i % width, i / width)))
                .sum();
            // A shape that kept shapes are clipped to stays, whatever it shows itself
            let id = scene.shapes[k].id;
            let clips =
                (k + 1..scene.shapes.len()).any(|j| keep[j] && scene.shapes[j].clip == Some(id));
            // Whether the shape stays is priced at the acceptance's price
            let own = scene.shapes[k].cost_segments() * ctx.accept_cost()
                + ctx.accept_path_cost()
                + crate::fill::cost_segments(&fill) * ctx.accept_cost();
            let pays = saving >= own;
            if !clips && (shown.is_empty() || !pays) && crate::diagnostics::debug() {
                eprintln!(
                    "pruned: shape {id} at {k}, {} pixels shown, saving {saving:.1} for {} segments, color {:?} below {:?}",
                    shown.len(),
                    scene.shapes[k].cost_segments(),
                    color * 255.,
                    shown.first().map(|&i| below(i) * 255.)
                );
            }
            if !clips && (shown.is_empty() || !pays) {
                keep[k] = false;
                continue;
            }
        }
        if all.n > 0. {
            // By place, since ids are not unique in every scene: the shapes without one share
            // a placeholder
            scene.set_fill_at(k, fill);
        }
        for (i, c) in coverage[k].pixels(width) {
            above[i] = 1. - (1. - above[i]) * (1. - c);
        }
    }
    let mut k = 0;
    scene.shapes.retain(|_| {
        k += 1;
        keep[k - 1]
    });
    // The whole-scene render is cached by the count of changes, and pruning is one
    if keep.iter().any(|&k| !k) {
        scene.change();
    }
}

/// Per pixel of the box `[x0, y0, x1, y1)`, how many squares about it of radius 0 to `rim`
/// hold only pixels `full` says are: its chessboard distance to the nearest pixel that is not,
/// at most `rim + 1`, by two passes (`full` must be true beyond the box's reach of the image).
fn full_depths(full: &impl Fn(isize, isize) -> bool, b: [usize; 4], rim: usize) -> Vec<usize> {
    let ([x0, y0, x1, y1], cap) = (b, rim + 1);
    let (w, h) = (x1 - x0, y1 - y0);
    let mut d: Vec<usize> = (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| (x, y)))
        .map(|(x, y)| if full(x as isize, y as isize) { cap } else { 0 })
        .collect();
    let pass = |d: &mut Vec<usize>, x: usize, y: usize, dy: isize| {
        let mut v = d[y * w + x];
        let ahead = if dy < 0 {
            x.checked_sub(1)
        } else {
            (x + 1 < w).then_some(x + 1)
        };
        if let Some(ax) = ahead {
            v = v.min(d[y * w + ax] + 1);
        }
        let row = y as isize + dy;
        if (0..h as isize).contains(&row) {
            for nx in x.saturating_sub(1)..(x + 2).min(w) {
                v = v.min(d[row as usize * w + nx] + 1);
            }
        }
        d[y * w + x] = v;
    };
    for y in 0..h {
        for x in 0..w {
            pass(&mut d, x, y, -1);
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            pass(&mut d, x, y, 1);
        }
    }
    d
}

#[cfg(test)]
mod tests {
    /// The two passes count the full squares about each pixel as checking each square does.
    #[test]
    fn full_depths_count_the_full_squares() {
        let mut seed = 5u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        for _ in 0..40 {
            let (w, h) = (3 + next() % 20, 3 + next() % 20);
            let holes: Vec<bool> = (0..w * h).map(|_| next() % 9 == 0).collect();
            let full = |x: isize, y: isize| {
                x < 0
                    || y < 0
                    || x >= w as isize
                    || y >= h as isize
                    || !holes[y as usize * w + x as usize]
            };
            let rim = 1 + next() % 4;
            let d = super::full_depths(&full, [0, 0, w, h], rim);
            for y in 0..h as isize {
                for x in 0..w as isize {
                    let brute = (0..=rim as isize)
                        .take_while(|&r| (-r..=r).all(|dy| (-r..=r).all(|dx| full(x + dx, y + dy))))
                        .count();
                    assert_eq!(d[y as usize * w + x as usize], brute, "{x} {y}");
                }
            }
        }
    }
}
