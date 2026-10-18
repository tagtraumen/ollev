//! The verify loop: the emitted bytes rendered and compared with the model's own render; an
//! object whose miss costs more than a segment steps to a finer encoding, kept where it
//! lowers the miss.

use std::time::Duration;

use crate::model::{Fidelity, Model};
use crate::translucency::raster::NOISE;
use glam::Vec3;
use resvg::{tiny_skia, usvg};

use crate::emit::Emitter;

/// What the loop found: objects missing at the first encoding, those a finer one helped,
/// those still missing, and the miss left in units of the segment's price.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub objects: usize,
    pub checked: bool,
    pub missed: usize,
    pub stepped: usize,
    pub left: usize,
    pub excess: f64,
    /// The largest miss of an object, in units of the segment's price.
    pub worst: f64,
    pub renders: usize,
    pub time: Duration,
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if !self.checked {
            return write!(f, "{} objects, not verified", self.objects);
        }
        write!(
            f,
            "{} objects, {} missed, {} finer, {} left, excess {:.2} segments, worst {:.3}, {} renders, {:.1?}",
            self.objects,
            self.missed,
            self.stepped,
            self.left,
            self.excess,
            self.worst,
            self.renders,
            self.time
        )
    }
}

/// An SVG rendered by resvg over white at `width` by `height`, as the bench renders it.
pub fn render_svg(svg: &str, width: usize, height: usize) -> Option<Vec<Vec3>> {
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).ok()?;
    let mut pixmap = tiny_skia::Pixmap::new(width as u32, height as u32)?;
    pixmap.fill(tiny_skia::Color::WHITE);
    let size = tree.size();
    let transform = tiny_skia::Transform::from_scale(
        width as f32 / size.width(),
        height as f32 / size.height(),
    );
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    Some(
        pixmap
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                let a = p[3] as u32;
                let blend = |c: u8| ((c as u32 * a + 255 * (255 - a)) / 255) as f32 / 255.;
                Vec3::new(blend(p[0]), blend(p[1]), blend(p[2]))
            })
            .collect(),
    )
}

/// Per pixel, the squared colour distance of an emitted render beyond the model's: beyond the
/// range the model takes over the pixel's 3 x 3 neighbourhood (an anti-aliasing pixel of
/// position) by more than the picture's noise there.
struct Budget {
    lo: Vec<Vec3>,
    hi: Vec<Vec3>,
    tolerance: Vec<f32>,
}

impl Budget {
    fn new(reference: &[Vec3], width: usize, height: usize, fidelity: &Fidelity) -> Self {
        let n = width * height;
        let (mut lo, mut hi) = (vec![Vec3::INFINITY; n], vec![Vec3::NEG_INFINITY; n]);
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                for yy in y.saturating_sub(1)..(y + 2).min(height) {
                    for xx in x.saturating_sub(1)..(x + 2).min(width) {
                        let v = reference[yy * width + xx];
                        lo[i] = lo[i].min(v);
                        hi[i] = hi[i].max(v);
                    }
                }
            }
        }
        let floor = NOISE as f32;
        let tolerance = (0..n)
            .map(|i| {
                fidelity
                    .tolerance
                    .as_ref()
                    .and_then(|t| t.get(i))
                    .map_or(floor, |&t| t.max(floor))
            })
            .collect();
        Budget { lo, hi, tolerance }
    }

    /// The miss of each of `n` objects: the excess of `emitted` summed over the pixels it
    /// owns (see [`Model::render`]).
    fn misses(&self, emitted: &[Vec3], owner: &[u32], n: usize) -> Vec<f64> {
        let mut miss = vec![0f64; n];
        for (i, &k) in owner.iter().enumerate() {
            if k == u32::MAX {
                continue;
            }
            let d = (emitted[i].clamp(self.lo[i], self.hi[i]) - emitted[i]).length();
            let over = (d - self.tolerance[i]).max(0.) as f64;
            miss[k as usize] += over * over;
        }
        miss
    }
}

/// The emitted bytes and the loop's report.
pub struct Output {
    pub bytes: Vec<u8>,
    pub levels: Vec<u8>,
    pub report: Report,
}

/// Compiles `model` to the emitter's bytes, each object at the coarsest encoding whose render
/// misses the model's by less than a segment's price, or at the level that missed least.
pub fn compile(model: &Model, emitter: &dyn Emitter) -> Output {
    let start = crate::timing::Clock::now();
    let objects = model.objects();
    let mut levels = vec![0u8; objects.len()];
    let mut bytes = emitter.encode(model, &levels);
    let mut report = Report {
        objects: objects.len(),
        ..Report::default()
    };
    let (w, h) = (model.width, model.height);
    let render = |bytes: &[u8]| emitter.render(bytes, w, h);
    let (Some(fidelity), Some(emitted)) = (&model.fidelity, render(&bytes)) else {
        report.time = start.elapsed();
        return Output {
            bytes,
            levels,
            report,
        };
    };
    report.checked = true;
    report.renders = 1;
    let (reference, owner) = model.render();
    let budget = Budget::new(&reference, w, h, fidelity);
    let price = fidelity.segment_price;
    let mut miss = budget.misses(&emitted, &owner, objects.len());
    report.missed = miss.iter().filter(|&&m| m > price).count();
    let finest: Vec<u8> = objects.iter().map(|o| emitter.finest(o)).collect();
    let mut trying: Vec<usize> = (0..objects.len())
        .filter(|&k| miss[k] > price && levels[k] < finest[k])
        .collect();
    // Each missing object steps through every finer level (a level need not miss less than
    // the one before it), keeping the one that missed least
    let mut best: Vec<(u8, f64)> = (0..objects.len()).map(|k| (0, miss[k])).collect();
    let mut rendered = levels.clone();
    while !trying.is_empty() {
        trying.iter().for_each(|&k| levels[k] += 1);
        rendered = levels.clone();
        let stepped = emitter.encode(model, &levels);
        let Some(emitted) = render(&stepped) else {
            break;
        };
        report.renders += 1;
        let next = budget.misses(&emitted, &owner, objects.len());
        let mut more = Vec::new();
        for &k in &trying {
            if next[k] < best[k].1 {
                best[k] = (levels[k], next[k]);
            }
            if best[k].1 > price && levels[k] < finest[k] {
                more.push(k);
            }
        }
        trying = more;
    }
    for (k, &(level, m)) in best.iter().enumerate() {
        (levels[k], miss[k]) = (level, m);
    }
    if levels.iter().any(|&l| l > 0) {
        bytes = emitter.encode(model, &levels);
    }
    // The misses of the levels kept, where a step was taken back
    if rendered != levels
        && let Some(emitted) = render(&bytes)
    {
        report.renders += 1;
        miss = budget.misses(&emitted, &owner, objects.len());
    }
    report.stepped = levels.iter().filter(|&&l| l > 0).count();
    report.left = miss.iter().filter(|&&m| m > price).count();
    let unit = price.max(f64::MIN_POSITIVE);
    report.excess = miss.iter().filter(|&&m| m > price).sum::<f64>() / unit + 0.;
    report.worst = miss.iter().fold(0f64, |a, &m| a.max(m)) / unit;
    report.time = start.elapsed();
    Output {
        bytes,
        levels,
        report,
    }
}
