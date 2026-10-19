//! Soft regions: a run of flat bands with no edge between them, the picture shading smoothly
//! where the trace posterized it, read as one region and drawn as one shape with a gradient
//! where its error plus price beats the bands'.

use glam::{Vec2, Vec3};

use crate::render::distance::Contour;

use crate::fill::{Costs, along, fit};
use crate::translucency::NOISE;
use crate::translucency::raster::unit_of;
use crate::{Context, HashMap, HashSet, Scene, Shape, Window};

/// The least width, in drawing units ([`unit_of`]) and never under as many pixels, over which
/// the step between two bands has to spread for their shared boundary to be shading and not an
/// edge: an edge is anti-aliased within about one pixel, and drawn sharp within one unit.
pub const SOFT_EDGE: f64 = 2.;

/// Fewest bands a run has: two flat regions meeting at a blurred boundary are a blurred edge
/// (see [`crate::blur`]), not shading; a run shades through a band between its neighbours,
/// or is two bands the picture steps across as it does inside them.
pub const LEAST_RUN: usize = 3;

/// No shape owns a pixel: under a translucent shape, or covered by none by half.
pub(crate) const NONE: u32 = u32::MAX;

/// Whether a shape may be a band of a soft region: an opaque, unclipped, unblurred fill of the
/// opaque scene.
pub(crate) fn is_band(s: &Shape) -> bool {
    s.layer.is_none()
        && s.stroke.is_none()
        && s.clip.is_none()
        && s.blur == 0.
        && s.fill.opacity >= 1. - 0.5 / 255.
        && s.fill.gradient.is_none_or(|g| g.is_opaque())
}

/// What a shape costs at the acceptance price: its segments, its path and its paint.
pub(crate) fn price(ctx: &Context, s: &Shape, costs: Costs) -> f64 {
    s.cost_segments() * ctx.accept_cost() + ctx.accept_path_cost() + costs.of(&s.fill) as f64
}

/// What the picture does from pixel to pixel over a set of neighbouring pairs: how many, and
/// their summed colour step.
#[derive(Default, Clone, Copy)]
struct Steps {
    pairs: usize,
    step: f64,
}

impl Steps {
    fn mean(&self) -> f64 {
        self.step / self.pairs.max(1) as f64
    }
}

/// The scene as the bands show it: per pixel the place of the topmost shape covering it by half
/// ([`NONE`] under a translucent shape), each band's mean colour, and the bands linked by soft
/// seams.
pub(crate) struct Bands {
    pub(crate) owner: Vec<u32>,
    pub(crate) mean: Vec<Vec3>,
    pub(crate) linked: HashMap<usize, Vec<usize>>,
    /// The linked pairs whose seam steps no more than their bands do inside, by the noise.
    seamless: HashSet<(usize, usize)>,
    /// The pixels each shape owns.
    pub(crate) pixels: Vec<Vec<usize>>,
}

impl Bands {
    pub(crate) fn of(ctx: &Context, scene: &Scene) -> Self {
        let (w, h) = (ctx.width, ctx.height);
        let n = scene.shapes.len();
        let mut owner = vec![NONE; w * h];
        for (k, s) in scene.shapes.iter().enumerate() {
            let r = s.painted();
            let opaque = s.fill.opacity >= 1. - 0.5 / 255.
                && s.fill.gradient.is_none_or(|g| g.is_opaque())
                && s.clip.is_none();
            for y in r.y0..(r.y0 + r.h).min(h) {
                for x in r.x0..(r.x0 + r.w).min(w) {
                    let v = r.values[(y - r.y0) * r.w + x - r.x0];
                    if !opaque && v > 0. {
                        owner[y * w + x] = NONE;
                    } else if v >= 0.5 {
                        owner[y * w + x] = k as u32;
                    }
                }
            }
        }
        let (mut sum, mut count) = (vec![Vec3::ZERO; n], vec![0usize; n]);
        let mut pixels = vec![Vec::new(); n];
        for (i, &o) in owner.iter().enumerate() {
            if o != NONE {
                sum[o as usize] += ctx.target[i];
                count[o as usize] += 1;
                pixels[o as usize].push(i);
            }
        }
        let mean: Vec<Vec3> = (0..n).map(|k| sum[k] / count[k].max(1) as f32).collect();
        let band: Vec<bool> = scene.shapes.iter().map(is_band).collect();
        // From each pixel to its right and lower neighbour: the steps within each band (its
        // noise and its slope), and across each seam between two bands, and how sharp it is
        let mut seams: HashMap<(usize, usize), (Steps, Steps)> = HashMap::default();
        let mut inner = vec![Steps::default(); n];
        let t = |i: usize| ctx.target[i];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                for (j, d, room) in [(1, x + 4 <= w && x >= 2), (w, y + 4 <= h && y >= 2)]
                    .into_iter()
                    .filter(|&(d, _)| if d == 1 { x + 1 < w } else { y + 1 < h })
                    .map(|(d, room)| (i + d, d, room))
                {
                    let (a, b) = (owner[i], owner[j]);
                    if a == NONE || b == NONE || !band[a as usize] || !band[b as usize] {
                        continue;
                    }
                    let step = t(i).distance(t(j)) as f64;
                    if a == b {
                        inner[a as usize].pairs += 1;
                        inner[a as usize].step += step;
                        continue;
                    }
                    let (s, sharp) = seams
                        .entry((a.min(b) as usize, a.max(b) as usize))
                        .or_default();
                    s.pairs += 1;
                    s.step += step;
                    // Sharp: the change over the anti-aliased reach about the seam exceeds what
                    // the steeper slope just outside predicts by twice the noise (a ramp's end:
                    // not), counted per crossing so the seam's share of them is its mean
                    if room {
                        let across = t(i - d).distance(t(j + d)) as f64;
                        let beyond = t(i - 2 * d)
                            .distance(t(i - d))
                            .max(t(j + d).distance(t(j + 2 * d)))
                            as f64;
                        let step = across - 3. * beyond > 2. * NOISE as f64;
                        sharp.pairs += 1;
                        sharp.step += f64::from(u8::from(step));
                    }
                }
            }
        }
        // Soft: the picture's step from pixel to pixel across is no more than the bands' step
        // spread over the soft width plus what it steps within them, along a seam that long,
        // and along most of it no step within the anti-aliased reach beyond the slope by twice
        // the noise (a sharp edge over a glow is no glow band)
        let width = (SOFT_EDGE * unit_of(w)).max(SOFT_EDGE);
        let mut linked: HashMap<usize, Vec<usize>> = HashMap::default();
        let mut seamless: HashSet<(usize, usize)> = HashSet::default();
        for (&(a, b), (s, sharp)) in &seams {
            let bands = mean[a].distance(mean[b]) as f64;
            let within = 0.5 * (inner[a].mean() + inner[b].mean());
            if s.pairs as f64 >= width && s.mean() <= bands / width + within && sharp.mean() <= 0.5
            {
                linked.entry(a).or_default().push(b);
                linked.entry(b).or_default().push(a);
                if s.mean() <= within + NOISE as f64 {
                    seamless.insert((a, b));
                }
            }
        }
        Bands {
            owner,
            mean,
            linked,
            seamless,
            pixels,
        }
    }

    /// Whether band `m` lies between two of its neighbours in `run`, or has fewer than two
    /// there: else it is a ridge or a trough, which no steady progression of colour holds.
    fn steady(&self, m: usize, run: &HashSet<usize>) -> bool {
        let near: Vec<usize> = self.linked[&m]
            .iter()
            .copied()
            .filter(|k| run.contains(k))
            .collect();
        let between = |a: Vec3, b: Vec3| {
            let d = b - a;
            let t = (self.mean[m] - a).dot(d) / d.length_squared().max(f32::MIN_POSITIVE);
            t > 0. && t < 1.
        };
        near.len() < 2
            || near.iter().enumerate().any(|(p, &a)| {
                near[p + 1..]
                    .iter()
                    .any(|&b| between(self.mean[a], self.mean[b]))
            })
    }

    /// The run of band `lowest`: the bands above it within its outline it reaches by soft
    /// seams through such bands, those whose colours do not progress steadily left out, in
    /// stack order; `None` below [`LEAST_RUN`] bands but for two seamless ones.
    fn run(&self, scene: &Scene, lowest: usize) -> Option<Vec<usize>> {
        if !self.linked.contains_key(&lowest) {
            return None;
        }
        let (w, h) = (scene.width, scene.height);
        let unit = unit_of(w);
        let base = &scene.shapes[lowest];
        let mut inside: HashMap<usize, bool> = HashMap::default();
        // Within: what of the band lies outside the lowest's outline, its holes filled (bands
        // nested as rings, each over the hole of the one below), is no more than a unit along
        // its rim
        let mut outline: Option<crate::scene::Raster> = None;
        let mut within = |k: usize| {
            *inside.entry(k).or_insert_with(|| {
                let s = &scene.shapes[k];
                let (r, b) = (s.raster(), &*outline.get_or_insert_with(|| filled(base)));
                // The coverage's box first, which handles flung wide do not widen
                let grown = unit.ceil() as usize;
                if r.x0 + grown < b.x0
                    || r.y0 + grown < b.y0
                    || r.x0 + r.w > b.x0 + b.w + grown
                    || r.y0 + r.h > b.y0 + b.h + grown
                {
                    return false;
                }
                let (mut outside, mut rim) = (0f64, 0f64);
                for y in r.y0..(r.y0 + r.h).min(h) {
                    for x in r.x0..(r.x0 + r.w).min(w) {
                        let v = r.values[(y - r.y0) * r.w + x - r.x0] as f64;
                        outside += v * (1. - b.at(y * w + x, w) as f64);
                        rim += f64::from(u8::from(v > 0.02 && v < 0.98));
                    }
                }
                outside <= unit * rim
            })
        };
        let mut run: HashSet<usize> = HashSet::from_iter([lowest]);
        let mut stack = vec![lowest];
        while let Some(k) = stack.pop() {
            for &m in &self.linked[&k] {
                if m > lowest && !run.contains(&m) && within(m) {
                    run.insert(m);
                    stack.push(m);
                }
            }
        }
        // Leave out the ridges and troughs, and what they alone linked, until none is left
        loop {
            let steady: HashSet<usize> = run
                .iter()
                .copied()
                .filter(|&m| m == lowest || self.steady(m, &run))
                .collect();
            let mut reached = HashSet::from_iter([lowest]);
            let mut stack = vec![lowest];
            while let Some(k) = stack.pop() {
                for &m in &self.linked[&k] {
                    if steady.contains(&m) && reached.insert(m) {
                        stack.push(m);
                    }
                }
            }
            if reached.len() == run.len() {
                break;
            }
            run = reached;
        }
        let mut run: Vec<usize> = run.into_iter().collect();
        run.sort_unstable();
        // Two bands make a run where the picture steps across their seam as it does inside
        // them (a ramp cut once), not as at a blurred edge
        let seamless = run.len() == 2 && self.seamless.contains(&(run[0], run[1]));
        (run.len() >= LEAST_RUN || seamless).then_some(run)
    }
}

/// The coverage of each loop of `s` alone.
fn loop_rasters(s: &Shape) -> Vec<crate::scene::Raster> {
    (0..s.loops.len())
        .map(|l| {
            crate::scene::coverage_raster(
                &crate::scene::loops_path(&s.loops[l..=l]),
                crate::render::FillRule::EvenOdd,
            )
        })
        .collect()
}

/// The coverage of `s` with its holes filled: the most any one of its loops covers alone.
fn filled(s: &Shape) -> crate::scene::Raster {
    if s.loops.len() < 2 {
        return s.raster().clone();
    }
    let parts = loop_rasters(s);
    let x0 = parts.iter().map(|p| p.x0).min().unwrap_or(0);
    let y0 = parts.iter().map(|p| p.y0).min().unwrap_or(0);
    let x1 = parts.iter().map(|p| p.x0 + p.w).max().unwrap_or(0);
    let y1 = parts.iter().map(|p| p.y0 + p.h).max().unwrap_or(0);
    let (w, h) = (x1 - x0, y1 - y0);
    let mut values = vec![0f32; w * h];
    for p in &parts {
        for y in 0..p.h {
            for x in 0..p.w {
                let v = &mut values[(p.y0 + y - y0) * w + p.x0 + x - x0];
                *v = v.max(p.values[y * p.w + x]);
            }
        }
    }
    crate::scene::Raster {
        x0,
        y0,
        w,
        h,
        values,
    }
}

/// The outline of `run` drawn as one shape where it is not its lowest band's: the lowest's
/// outer loops with the holes of every band of the run that hold no band of it (the bands
/// nested as rings over a core below them keep only the innermost hole), or None as it is.
fn run_outline(
    scene: &Scene,
    owner: &[u32],
    run: &[usize],
) -> Option<Vec<Vec<crate::tracer::Cubic>>> {
    if run.iter().all(|&k| scene.shapes[k].loops.len() < 2) {
        return None;
    }
    let w = scene.width;
    let unit = unit_of(w);
    let of_run = |i: usize| owner[i] != NONE && run.binary_search(&(owner[i] as usize)).is_ok();
    let mut loops = Vec::new();
    for (place, &k) in run.iter().enumerate() {
        let s = &scene.shapes[k];
        let rasters = loop_rasters(s);
        for (l, outline) in s.loops.iter().enumerate() {
            // A hole lies within an odd number of the shape's other loops
            let Some(p) = outline.first().map(|c| c[0]) else {
                continue;
            };
            let at = p.y.max(0.) as usize * w + (p.x.max(0.) as usize).min(w - 1);
            let depth = (0..s.loops.len())
                .filter(|&m| m != l && rasters[m].at(at, w) > 0.5)
                .count();
            if depth % 2 == 0 {
                if place == 0 {
                    loops.push(outline.clone());
                }
                continue;
            }
            // A hole holding a band of the run beyond its anti-aliased rim is filled by it
            let (mut held, mut rim) = (0f64, 0f64);
            for (i, v) in rasters[l].pixels(w) {
                if v >= 0.98 && i < owner.len() && of_run(i) {
                    held += 1.;
                }
                rim += f64::from(u8::from(v > 0.02 && v < 0.98));
            }
            let kept = held <= unit * rim;
            if kept {
                loops.push(outline.clone());
            }
        }
    }
    (loops != scene.shapes[run[0]].loops).then_some(loops)
}

/// Most samples a run's path gradients are compared on, each curve's: enough for the ramp's
/// profile at a few samples per pixel along a frame-wide run.
const PATH_SAMPLES: usize = 4096;

/// Most band boundaries a path gradient is tried along.
const BOUNDARIES: usize = 4;

/// The band boundaries of `run` a path gradient is tried along: the outer loops of its bands
/// above the lowest, at most [`BOUNDARIES`] of them evenly by area.
fn boundaries(scene: &Scene, run: &[usize]) -> Vec<Contour> {
    let mut loops: Vec<(f32, Contour)> = run[1..]
        .iter()
        .filter_map(|&k| {
            let s = &scene.shapes[k];
            let contour = Contour::new(s.loops.first()?)?;
            Some((s.raster().values.iter().sum::<f32>(), contour))
        })
        .collect();
    loops.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = loops.len();
    if n <= BOUNDARIES {
        return loops.into_iter().map(|l| l.1).collect();
    }
    (0..BOUNDARIES)
        .map(|i| loops[(2 * i + 1) * n / (2 * BOUNDARIES)].1)
        .collect()
}

/// The seams between the bands of `run` as curves, the longest [`BOUNDARIES`] of them: the
/// points midway between the pixels of two bands that meet, walked into a chain from an end
/// (or anywhere, a loop) and fitted by cubics to a pixel; open ones measured in `window`.
fn seams(owner: &[u32], width: usize, run: &[usize], window: Window) -> Vec<Contour> {
    use glam::DVec2;
    let of_run = |o: u32| o != NONE && run.binary_search(&(o as usize)).is_ok();
    let mut points: HashMap<(u32, u32), Vec<DVec2>> = HashMap::default();
    for i in window.pixels(width) {
        let (x, y) = (i % width, i / width);
        let right = (x + 1 < window.x + window.width).then_some(i + 1);
        let down = (y + 1 < window.y + window.height).then_some(i + width);
        for j in [right, down].into_iter().flatten() {
            let (a, b) = (owner[i], owner[j]);
            if a != b && of_run(a) && of_run(b) {
                let p = DVec2::new(x as f64 + 0.5, y as f64 + 0.5);
                let q = DVec2::new((j % width) as f64 + 0.5, (j / width) as f64 + 0.5);
                points
                    .entry((a.min(b), a.max(b)))
                    .or_default()
                    .push(0.5 * (p + q));
            }
        }
    }
    let mut seams: Vec<Vec<DVec2>> = points.into_values().collect();
    seams.sort_by_key(|s| std::cmp::Reverse(s.len()));
    let (min, max) = (
        window.origin(),
        window.origin() + Vec2::new(window.width as f32, window.height as f32),
    );
    seams
        .into_iter()
        .take(BOUNDARIES)
        .filter_map(|mut left| {
            // The chain from an end: a point with one neighbour, else the first
            let near = |p: DVec2, q: DVec2| p.distance_squared(q) <= 1.01;
            let degree =
                |p: DVec2, all: &[DVec2]| all.iter().filter(|q| **q != p && near(p, **q)).count();
            let start = (0..left.len())
                .find(|&k| degree(left[k], &left) <= 1)
                .unwrap_or(0);
            let mut chain = vec![left.swap_remove(start)];
            while let Some(k) = (0..left.len())
                .filter(|&k| left[k].distance_squared(chain[chain.len() - 1]) <= 2.26)
                .min_by(|&a, &b| {
                    let l = chain[chain.len() - 1];
                    left[a]
                        .distance_squared(l)
                        .total_cmp(&left[b].distance_squared(l))
                })
            {
                chain.push(left.swap_remove(k));
            }
            if chain.len() < 4 {
                return None;
            }
            let closed = chain.len() > 8 && chain[0].distance(chain[chain.len() - 1]) <= 1.5;
            if closed {
                chain.push(chain[0]);
            }
            let cubics: Vec<[Vec2; 4]> = crate::translucency::lines::fit_cubics(&chain, 1.0)
                .into_iter()
                .map(|c| c.map(glam::DVec2::as_vec2))
                .collect();
            if closed {
                Contour::new(&cubics)
            } else {
                Contour::open(&cubics, min, max)
            }
        })
        .collect()
}

/// Draws `run` (places in the stack, its lowest first) as its lowest band painted with the
/// gradient fitted to the run's pixels, the others removed, where that lowers the error plus
/// price over the run's box; else leaves the scene as it was. A checkpoint between its steps.
async fn replace(
    ctx: &Context,
    scene: &mut Scene,
    owner: &[u32],
    run: &[usize],
    paths: bool,
) -> bool {
    let (w, h) = (ctx.width, ctx.height);
    let lowest = run[0];
    // No shape clipped to a band that would go
    let ids: Vec<usize> = run[1..].iter().map(|&k| scene.shapes[k].id).collect();
    if scene
        .shapes
        .iter()
        .any(|s| s.clip.is_some_and(|c| ids.contains(&c)))
    {
        return false;
    }
    let of_run = |i: usize| owner[i] != NONE && run.binary_search(&(owner[i] as usize)).is_ok();
    let (mut min, mut max) = (Vec2::INFINITY, Vec2::NEG_INFINITY);
    for &k in run {
        min = min.min(scene.shapes[k].paint_min());
        max = max.max(scene.shapes[k].paint_max());
    }
    let window = Window::around(min, max, 1., w, h);
    // The run's pixels, its rim against what is not of it left out of the fit
    let (mut samples, mut extent) = (Vec::new(), Vec::new());
    for i in window.pixels(w).filter(|&i| of_run(i)) {
        let (x, y) = (i % w, i / w);
        let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
        extent.push(p);
        if crate::context::neighbours(i, w, h).all(of_run) {
            samples.push((p, ctx.target[i]));
        }
    }
    if samples.is_empty() {
        return false;
    }
    // Fitted on the plain error (under the tolerance a flat colour passes for the shading), and
    // priced as the planner prices whether a shape is drawn at all, its stops included
    let solid = samples.iter().map(|s| s.1).sum::<Vec3>() / samples.len() as f32;
    let costs = Costs::at(ctx.accept_cost());
    let center = scene.shapes[lowest].primitive_center();
    let (mut fill, sse) = fit(&samples, &[], solid, center, costs, &extent);
    crate::progress::poll().await;
    // A path gradient along each band boundary of the run (its iso-contours start the curve
    // well) against the linear or radial, on error plus price; the curves compared on at most
    // [`PATH_SAMPLES`] of the samples, the best refitted on all; none without `paths`
    let stride = samples.len().div_ceil(PATH_SAMPLES);
    let few: Vec<(Vec2, Vec3)> = samples.iter().step_by(stride).copied().collect();
    let share = few.len() as f32 / samples.len() as f32;
    let scaled = Costs {
        stop: costs.stop * share,
        ..costs
    };
    let mut curve: Option<(Contour, f32)> = None;
    let contours = if paths {
        let mut contours = boundaries(scene, run);
        contours.extend(seams(owner, w, run, window));
        contours
    } else {
        Vec::new()
    };
    for contour in contours {
        crate::progress::poll().await;
        if let Some((f, e)) = along(&few, contour, scaled) {
            let total = e / share + costs.of(&f);
            if curve.is_none_or(|c| total < c.1) {
                curve = Some((contour, total));
            }
        }
    }
    if let Some((contour, _)) = curve
        && let Some((f, e)) = along(&samples, contour, costs)
        && e + costs.of(&f) < sse + costs.of(&fill)
    {
        fill = f;
    }
    if fill.gradient.is_none() {
        return false;
    }
    crate::progress::poll().await;
    let render = |scene: &Scene| scene.render(window, 0..scene.shapes.len(), [0.; 3]);
    let before = ctx.error(window, &render(scene)) as f64
        + run.iter().map(|&k| price(ctx, &scene.shapes[k], costs)).sum::<f64>();
    let old = scene.shapes[lowest].clone();
    crate::progress::poll().await;
    let outline = run_outline(scene, owner, run);
    crate::progress::poll().await;
    let removed: Vec<(usize, Shape)> = run[1..]
        .iter()
        .rev()
        .map(|&k| (k, scene.shapes.remove(k)))
        .collect();
    if let Some(loops) = outline {
        let change = scene.change();
        let s = &mut scene.shapes[lowest];
        (s.loops, s.primitive, s.changed) = (loops, None, change);
        s.update_bounds();
    }
    scene.set_fill_at(lowest, fill);
    scene.change();
    let after = ctx.error(window, &render(scene)) as f64 + price(ctx, &scene.shapes[lowest], costs);
    if crate::diagnostics::log() {
        eprintln!(
            "soft: run of {} bands at {min:.0?}..{max:.0?}, {} px, {}: {before:.2} to {after:.2}",
            run.len(),
            extent.len(),
            match fill.gradient {
                Some(crate::render::Gradient::Linear { .. }) => "linear",
                Some(crate::render::Gradient::Path { contour, .. }) if contour.is_closed() =>
                    "path",
                Some(crate::render::Gradient::Path { .. }) => "open path",
                _ => "radial",
            }
        );
    }
    if after < before {
        return true;
    }
    scene.shapes[lowest] = old;
    scene.shapes[lowest].changed = scene.change();
    for (k, s) in removed.into_iter().rev() {
        scene.shapes.insert(k, s);
    }
    scene.change();
    false
}

/// Replaces runs of soft bands of the decided `scene`, each by its lowest band with a gradient
/// where that lowers the error plus price (see [`replace`]). A run is the bands above
/// one, within its outline, linked to it by seams the picture shades across (no step within
/// [`SOFT_EDGE`]) and progressing steadily in colour. The stack is walked from the bottom, so
/// the largest run that pays is taken. Returns how many runs were replaced.
pub async fn soften_scene(ctx: &Context, scene: &mut Scene, paths: bool) -> usize {
    let mut bands = Bands::of(ctx, scene);
    let mut replaced = 0;
    let mut lowest = 0;
    while lowest < scene.shapes.len() {
        crate::progress::poll().await;
        if !is_band(&scene.shapes[lowest]) {
            lowest += 1;
            continue;
        }
        let Some(run) = bands.run(scene, lowest) else {
            lowest += 1;
            continue;
        };
        if replace(ctx, scene, &bands.owner, &run, paths).await {
            replaced += 1;
            crate::progress::poll().await;
            bands = Bands::of(ctx, scene);
        }
        lowest += 1;
    }
    replaced
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::*;
    use crate::scene::tests::disc;

    /// A radial ramp the trace posterized into four nested discs is drawn as the outer disc
    /// with a gradient, nearer the picture than the discs.
    #[test]
    fn posterized_discs_become_one_gradient() {
        let (w, c) = (64usize, Vec2::splat(32.));
        let colour = |d: f32| Vec3::new(1., 0.9 - 0.5 * d, 0.2) * (1. - 0.4 * d);
        let rgb: Vec<u8> = (0..w * w)
            .flat_map(|i| {
                let p = Vec2::new((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                let d = p.distance(c) / 28.;
                let v = if d < 1. { colour(d) } else { Vec3::splat(0.2) };
                v.to_array().map(|x| (x * 255.).round() as u8)
            })
            .collect();
        let ctx = Context::new(&rgb, w, w, 1.);
        let mut scene = Scene::with_background(w, w, Vec3::splat(0.2));
        for k in 0..4 {
            let r = 28. * (1. - k as f32 / 4.);
            scene.push(Shape::new(disc(c, r), colour(1. - (k as f32 + 0.5) / 4.)));
        }
        let full = Window::full(w, w);
        let before = ctx.error(full, &scene.render(full, 0..scene.shapes.len(), [0.; 3]));
        let replaced = crate::progress::finish(soften_scene(&ctx, &mut scene, true));
        assert_eq!(replaced, 1);
        assert_eq!(scene.shapes.len(), 2);
        assert!(scene.shapes[1].fill.gradient.is_some());
        let after = ctx.error(full, &scene.render(full, 0..scene.shapes.len(), [0.; 3]));
        assert!(after < before, "{after} {before}");
    }

    /// The same ramp posterized into rings nested over a flat core drawn below them: one ring
    /// with a gradient replaces them, keeping the hole the core shows through.
    #[test]
    fn rings_over_a_core_become_one_gradient() {
        let (w, c) = (64usize, Vec2::splat(32.));
        let colour = |d: f32| Vec3::new(1., 0.9 - 0.5 * d, 0.2) * (1. - 0.4 * d);
        let core = Vec3::new(0.95, 0.95, 1.);
        let rgb: Vec<u8> = (0..w * w)
            .flat_map(|i| {
                let p = Vec2::new((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                let d = p.distance(c) / 28.;
                let v = match d {
                    d if d < 0.25 => core,
                    d if d < 1. => colour(d),
                    _ => Vec3::splat(0.2),
                };
                v.to_array().map(|x| (x * 255.).round() as u8)
            })
            .collect();
        let ctx = Context::new(&rgb, w, w, 1.);
        let mut scene = Scene::with_background(w, w, Vec3::splat(0.2));
        scene.push(Shape::new(disc(c, 7.), core));
        for k in 0..3 {
            let (outer, inner) = (28. * (1. - k as f32 / 4.), 28. * (0.75 - k as f32 / 4.));
            let mut loops = disc(c, outer);
            loops.extend(disc(c, inner));
            scene.push(Shape::new(loops, colour(1. - (k as f32 + 0.5) / 4.)));
        }
        let full = Window::full(w, w);
        let before = ctx.error(full, &scene.render(full, 0..scene.shapes.len(), [0.; 3]));
        assert_eq!(crate::progress::finish(soften_scene(&ctx, &mut scene, true)), 1);
        assert_eq!(scene.shapes.len(), 3);
        let ring = &scene.shapes[2];
        assert!(ring.fill.gradient.is_some());
        assert_eq!(ring.loops.len(), 2, "the outline and the core's hole");
        let after = ctx.error(full, &scene.render(full, 0..scene.shapes.len(), [0.; 3]));
        assert!(after < before, "{after} {before}");
    }
}
