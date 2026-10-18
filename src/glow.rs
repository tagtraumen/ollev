//! Glows: a flat ground with a radial gradient fading to transparent over it (lamp halos,
//! spotlight pools, bokeh). The trace draws one as the ground with the glow cut out, the glow
//! shredded into flat bands or folded into a frame-sized gradient; this pass asks of each ground
//! whether its flat colour plus glows fitted from its residual lowers the error plus price.

use glam::{Vec2, Vec3};

use crate::fill::{Costs, Over, concentric, ellipse_transform, stretch, with_alpha};
use crate::soft::{Bands, NONE, is_band, price};
use crate::tracer::primitives::Primitive;
use crate::translucency::NOISE;
use crate::context::neighbours;
use crate::{Context, HashSet, Scene, Shape, Window};

/// How far a pixel's colour stands from the ground's for the ground's residual to hold a glow
/// there: beyond the noise of each of the two colours.
const GLOW_STEP: f32 = 2. * NOISE;

/// The plain squared error of pixel `i` showing `rendered`: glows are judged as the economy
/// judges objects, the tolerance aside (under it smooth shading passes for a flat ground).
fn plain(ctx: &Context, i: usize, rendered: Vec3) -> f32 {
    ctx.target[i].distance_squared(rendered)
}

/// [`plain`] summed over `window`, `rendered` its pixels row by row.
fn plain_error(ctx: &Context, window: Window, rendered: &[Vec3]) -> f32 {
    window
        .pixels(ctx.width)
        .zip(rendered)
        .map(|(i, r)| plain(ctx, i, *r))
        .sum()
}

/// The ground's colour: the most common colour of its pixels, to 8 bits, then the mean of the
/// pixels within the noise of it, so a glow folded into the ground does not tint it.
fn ground_colour(ctx: &Context, pixels: &[usize]) -> Option<Vec3> {
    let mut counts: crate::HashMap<[u8; 3], usize> = crate::HashMap::default();
    let key = |c: Vec3| c.to_array().map(|v| (v.clamp(0., 1.) * 255.).round() as u8);
    for &i in pixels {
        *counts.entry(key(ctx.target[i])).or_default() += 1;
    }
    let mode = counts.into_iter().max_by_key(|&(k, n)| (n, k))?.0;
    let mode = Vec3::from(mode.map(|v| v as f32 / 255.));
    let near: Vec<Vec3> = pixels
        .iter()
        .map(|&i| ctx.target[i])
        .filter(|c| (*c - mode).abs().max_element() <= NOISE)
        .collect();
    Some(near.iter().sum::<Vec3>() / near.len().max(1) as f32)
}

/// The bands the glow of ground `k` may have been drawn as: those reached from it over soft
/// seams, each farther in colour from the ground than the band it is reached from (a glow's
/// colour moves steadily from its core to the ground), within the ground's box (the glow lies
/// over it), neither the frame nor with a shape clipped to it.
fn members(scene: &Scene, bands: &Bands, k: usize, ground: Vec3) -> Vec<usize> {
    let clips: HashSet<usize> = scene.shapes.iter().filter_map(|s| s.clip).collect();
    let far = |m: usize| bands.mean[m].distance(ground);
    let g = &scene.shapes[k];
    let within = |s: &Shape| {
        s.paint_min().cmpge(g.paint_min()).all() && s.paint_max().cmple(g.paint_max()).all()
    };
    let mut found: HashSet<usize> = HashSet::default();
    let mut stack = vec![(k, 0f32)];
    while let Some((j, d)) = stack.pop() {
        for &m in bands.linked.get(&j).into_iter().flatten() {
            let s = &scene.shapes[m];
            if m != k
                && m != 0
                && is_band(s)
                && within(s)
                && !clips.contains(&s.id)
                && far(m) > d
                && found.insert(m)
            {
                stack.push((m, far(m)));
            }
        }
    }
    let mut out: Vec<usize> = found.into_iter().collect();
    out.sort_unstable();
    out
}

/// The 4-connected pieces of `mask` (indices into the image), each as its pixels.
fn pieces(mask: &HashSet<usize>, width: usize, height: usize) -> Vec<Vec<usize>> {
    let mut seen: HashSet<usize> = HashSet::default();
    let mut out = Vec::new();
    let mut sorted: Vec<usize> = mask.iter().copied().collect();
    sorted.sort_unstable();
    for &start in &sorted {
        if !seen.insert(start) {
            continue;
        }
        let (mut piece, mut stack) = (Vec::new(), vec![start]);
        while let Some(i) = stack.pop() {
            piece.push(i);
            for j in neighbours(i, width, height) {
                if mask.contains(&j) && seen.insert(j) {
                    stack.push(j);
                }
            }
        }
        out.push(piece);
    }
    out
}

/// A glow fitted to the pixels around `piece` of the ground's residual: a radial gradient
/// over `under` (what lies below the glow, per image pixel) fading to transparent at its
/// radius, drawn as the circle of that radius; or a dot, a flat disc out to its half contrast,
/// where it shows a flat inside. `shown` says which pixels the glow may explain, `ground_of`
/// which are the ground's or its bands', `apart` which the glow does not reach (see `propose`).
fn fit_glow(
    ctx: &Context,
    piece: &[usize],
    shown: &dyn Fn(usize) -> bool,
    ground_of: &dyn Fn(usize) -> bool,
    apart: &dyn Fn(usize) -> bool,
    under: &[Vec3],
    costs: Costs,
) -> Option<Shape> {
    let w = ctx.width;
    let at = |i: usize| Vec2::new((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
    let (mut min, mut max) = (Vec2::INFINITY, Vec2::NEG_INFINITY);
    let (mut centre, mut mass) = (Vec2::ZERO, 0f32);
    for &i in piece {
        let p = at(i);
        (min, max) = (min.min(p), max.max(p));
        let r = (ctx.target[i] - under[i]).length();
        (centre, mass) = (centre + p * r, mass + r);
    }
    let centre = centre / mass.max(1e-9);
    // The ground around the piece as far again as the piece reaches, so the fit sees its end
    let reach = 0.5 * (max - min).max_element();
    let window = Window::around(min, max, reach, w, ctx.height);
    let samples: Vec<Over> = window
        .pixels(w)
        .filter(|&i| shown(i))
        .map(|i| Over {
            p: at(i),
            target: ctx.target[i],
            below: under[i],
        })
        .collect();
    if samples.len() < 4 {
        return None;
    }
    let plain: Vec<(Vec2, Vec3)> = samples.iter().map(|s| (s.p, s.target)).collect();
    let extent: Vec<Vec2> = plain.iter().map(|s| s.0).collect();
    let opaque = concentric(&plain, centre, &extent)?;
    let Some(crate::render::Gradient::Radial { center, radius, .. }) = opaque.gradient else {
        return None;
    };
    // The circle stretched into the ellipse that fits best (a spotlight's pool on the floor)
    let (radius, angle, squash) = stretch(&plain, center, radius);
    let transform = ellipse_transform(angle, squash);
    let mut opaque = opaque;
    if let Some(crate::render::Gradient::Radial {
        radius: r,
        transform: t,
        ..
    }) = &mut opaque.gradient
    {
        (*r, *t) = (radius, transform);
    }
    // A glow fades out over its ground: of its rim it reaches, most lies on the ground (or its
    // bands); a gradient whose rim runs onto what shows through the ground is no glow over it
    let rim: Vec<usize> = (0..64)
        .map(|k| {
            center + transform * (Vec2::from_angle(k as f32 * std::f32::consts::TAU / 64.) * radius)
        })
        .filter(|p| p.cmpge(Vec2::ZERO).all() && p.x < w as f32 && p.y < ctx.height as f32)
        .map(|p| p.y as usize * w + p.x as usize)
        .collect();
    let on = rim.iter().filter(|&&i| ground_of(i)).count();
    let off = rim.iter().filter(|&&i| !ground_of(i) && !apart(i)).count();
    if on == 0 || on < off {
        return None;
    }
    let fill = with_alpha(&samples, &opaque, costs.stop, true)?;
    // A glow fades: from half its contrast with what lies below to nothing over more than an
    // anti-aliased edge's two pixels; one that falls off within them is a crisp shape, not a
    // glow. Contrast as the knots paint it over the ground
    let gradient = fill.gradient?;
    let below = under[(center.y.clamp(0., ctx.height as f32 - 1.) as usize) * w
        + center.x.clamp(0., w as f32 - 1.) as usize];
    let contrast = |c: [f32; 4]| ((glam::Vec4::from(c).truncate() - below) * c[3]).length();
    let knots: Vec<(f32, f32)> =
        std::iter::once((0., contrast([fill.color.x, fill.color.y, fill.color.z, 1.])))
            .chain(gradient.stops().iter().map(|(t, c)| (t, contrast(c))))
            .chain(std::iter::once((1., contrast(gradient.to()))))
            .collect();
    let top = knots.iter().map(|k| k.1).fold(0f32, f32::max);
    // Where the contrast falls through half its top, between the knots it falls between
    let half = knots
        .windows(2)
        .rev()
        .find(|k| k[0].1 >= 0.5 * top)
        .map_or(0., |k| {
            let fall = k[0].1 - k[1].1;
            let u = if fall > 0. {
                (k[0].1 - 0.5 * top) / fall
            } else {
                1.
            };
            k[0].0 + (k[1].0 - k[0].0) * u.min(1.)
        });
    // Along its narrowest axis, so a sliver of an ellipse is no glow
    let minor = radius * squash.min(1.);
    // and seen fading so on the pixels it explains, not under the shapes over it (a lit object
    // over its halo has no glow of its own)
    let fade = samples
        .iter()
        .map(|s| gradient.t(s.p))
        .filter(|&t| t >= half && t <= 1.)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), t| {
            (a.min(t), b.max(t))
        });
    let crisp = minor * (1. - half) <= 2. * crate::edges::SIGMA as f32
        || minor * (fade.1 - fade.0) <= 2. * crate::edges::SIGMA as f32;
    // A dot's inside is flat: the pixels it shows clear of its anti-aliased rim (at half the
    // contrast) within the noise of one colour; small discs are drawn so, not as glows
    let aa = 2. * crate::edges::SIGMA as f32;
    // Clear: farther from the edge along its own ray from the centre than an edge's AA
    let clear = |p: Vec2| {
        let t = gradient.t(p);
        t < half && (t <= 0. || (p - center).length() * (half / t - 1.) >= aa)
    };
    let inside: Vec<Vec3> = samples
        .iter()
        .filter(|s| clear(s.p))
        .map(|s| s.target)
        .collect();
    let mean = inside.iter().sum::<Vec3>() / inside.len().max(1) as f32;
    let flat = !inside.is_empty()
        && inside
            .iter()
            .all(|c| (*c - mean).abs().max_element() <= NOISE);
    if crisp && !flat {
        return None;
    }
    let (radius, fill) = if flat {
        (radius * half, crate::Fill::solid(mean))
    } else {
        (radius, fill)
    };
    let outline = if squash < 1. {
        Primitive::Ellipse {
            center,
            radii: Vec2::new(radius, radius * squash),
            angle,
        }
    } else {
        Primitive::Circle { center, radius }
    };
    let mut shape = Shape::new(vec![outline.cubics()], fill);
    shape.primitive = Some((0, outline));
    shape.smooth = true;
    shape.antialiased = true;
    Some(shape)
}

/// The pixels inside loop `l` of `s` alone, wholly covered.
fn inside_loop(s: &Shape, l: usize, w: usize, h: usize) -> Vec<usize> {
    let r = crate::scene::coverage_raster(
        &crate::scene::loops_path(&s.loops[l..=l]),
        crate::render::FillRule::EvenOdd,
    );
    r.pixels(w)
        .filter(|&(i, v)| v >= 1. && i % w >= r.x0 && i < w * h)
        .map(|(i, _)| i)
        .collect()
}

/// The loop of `s` enclosing the most, its outline; the others are its holes.
fn outer_loop(s: &Shape, w: usize, h: usize) -> Option<usize> {
    (0..s.loops.len()).max_by_key(|&l| inside_loop(s, l, w, h).len())
}

/// The holes of band `m` through which what lies below it shows (mostly), each as its loop and
/// pixels: the objects a glow drawn as `m` must keep, as an object lying in a lamp's light.
fn shown_through(
    scene: &Scene,
    bands: &Bands,
    m: usize,
) -> Vec<(Vec<crate::tracer::Cubic>, Vec<usize>)> {
    let (w, h) = (scene.width, scene.height);
    let s = &scene.shapes[m];
    let Some(outer) = outer_loop(s, w, h) else {
        return Vec::new();
    };
    (0..s.loops.len())
        .filter(|&l| l != outer)
        .filter_map(|l| {
            let pixels = inside_loop(s, l, w, h);
            let below = pixels
                .iter()
                .filter(|&&i| (bands.owner[i] as usize) < m)
                .count();
            (2 * below > pixels.len()).then(|| (s.loops[l].clone(), pixels))
        })
        .collect()
}

/// Ground `k` drawn flat in its own colour, with glows fitted from its residual over it, the
/// bands its glow was drawn as removed and the holes they filled closed, where that lowers the
/// error plus price; the place of the topmost glow, or None with the scene as it was.
async fn propose(
    ctx: &Context,
    scene: &mut Scene,
    bands: &Bands,
    now: &[Vec3],
    k: usize,
) -> Option<usize> {
    let (w, h) = (ctx.width, ctx.height);
    let costs = Costs::at(ctx.accept_cost());
    // A glow pays for its circle and gradient at least; a piece of residual removing less
    // than that is no glow
    let least = (4. * crate::scene::PRIMITIVE_SHARE + crate::fill::GRADIENT_SEGMENTS)
        * ctx.accept_cost()
        + ctx.accept_path_cost();
    // A solid ground within the noise of its own colour keeps it, so nothing changes
    // outside the glows and only they are judged
    let fill = scene.shapes[k].fill;
    let ground = ground_colour(ctx, &bands.pixels[k])?;
    let kept = fill.gradient.is_none() && (fill.color - ground).abs().max_element() <= NOISE;
    let ground = if kept { fill.color } else { ground };
    let members = members(scene, bands, k, ground);
    // No glow lowers the error below nothing: where the scene explains the ground and its
    // bands to within a glow's price already, there is nothing to gain
    let room: f64 = std::iter::once(k)
        .chain(members.iter().copied())
        .flat_map(|m| bands.pixels[m].iter())
        .map(|&i| plain(ctx, i, now[i]) as f64)
        .sum::<f64>()
        + members
            .iter()
            .map(|&m| price(ctx, &scene.shapes[m], costs))
            .sum::<f64>();
    if room <= least {
        return None;
    }
    // What shows through the bands' holes is an object over the glow, not part of it
    let through: Vec<(usize, Vec<crate::tracer::Cubic>, Vec<usize>)> = members
        .iter()
        .flat_map(|&m| {
            shown_through(scene, bands, m)
                .into_iter()
                .map(move |(l, p)| (m, l, p))
        })
        .collect();
    let hidden: HashSet<usize> = through.iter().flat_map(|t| t.2.iter().copied()).collect();
    let of = |i: usize| {
        let o = bands.owner[i];
        o != NONE
            && !hidden.contains(&i)
            && (o as usize == k || members.binary_search(&(o as usize)).is_ok())
    };
    let modes: std::cell::RefCell<crate::HashMap<u32, Option<Vec3>>> =
        std::cell::RefCell::default();
    // The pixels the ground and its glow show, the rim against anything else left out
    let shown = |i: usize| of(i) && neighbours(i, w, h).all(of);
    let residual: HashSet<usize> = std::iter::once(k)
        .chain(members.iter().copied())
        .flat_map(|m| bands.pixels[m].iter().copied())
        .filter(|&i| shown(i) && (ctx.target[i] - ground).abs().max_element() > GLOW_STEP)
        .collect();
    let mut found: Vec<(f64, Vec<usize>)> = pieces(&residual, w, h)
        .into_iter()
        .map(|p| {
            let mass: f64 = p.iter().map(|&i| plain(ctx, i, ground) as f64).sum();
            (mass, p)
        })
        // A glow explains a piece the flat ground leaves, and pays where the scene misses it
        // now or draws it with bands of its own
        .filter(|(m, p)| {
            let missed: f64 = p.iter().map(|&i| plain(ctx, i, now[i]) as f64).sum();
            *m > least && (missed > least || p.iter().any(|&i| bands.owner[i] as usize != k))
        })
        .collect();
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    // Glows fitted one after another, each over the ground and the ones before it
    let mut under = vec![ground; w * h];
    let mut glows: Vec<Shape> = Vec::new();
    for (_, piece) in &found {
        crate::progress::poll().await;
        // Apart from a glow clipped to the ground: under shapes above it, or on a shape below
        // the ground showing its own flat colour there, untinted (a tinted one is the glow's
        // ground); made past the checkpoint, so no borrow of `modes` is held across it
        let apart = |i: usize| {
            let o = bands.owner[i];
            if o == NONE || hidden.contains(&i) || of(i) {
                return false;
            }
            if o as usize > k {
                return true;
            }
            let mode = *modes
                .borrow_mut()
                .entry(o)
                .or_insert_with(|| ground_colour(ctx, &bands.pixels[o as usize]));
            mode.is_some_and(|m| (ctx.target[i] - m).abs().max_element() <= GLOW_STEP)
        };
        let Some(mut glow) = fit_glow(ctx, piece, &shown, &of, &apart, &under, costs) else {
            continue;
        };
        // A glow lies over its ground alone where it would reach a shape below the ground that
        // shows untinted: clipped to it
        if glow.fill.gradient.is_some()
            && glow
                .raster()
                .pixels(w)
                .any(|(i, v)| v > 0. && i < w * h && (bands.owner[i] as usize) < k && apart(i))
        {
            glow.clip = Some(scene.shapes[k].id);
        }
        let r = glow.raster();
        for y in r.y0..(r.y0 + r.h).min(h) {
            for x in r.x0..(r.x0 + r.w).min(w) {
                let v = r.values[(y - r.y0) * r.w + x - r.x0];
                let (c, a) = glow.fill.paint_pixel(x, y);
                let u = under[y * w + x];
                under[y * w + x] = u + v * a * (c - u);
            }
        }
        glows.push(glow);
    }
    if glows.is_empty() {
        return None;
    }
    crate::progress::poll().await;
    // Dots alone lie over the ground as it is drawn: they explain no band and no reason to
    // redraw the ground flat
    let dots = glows.iter().all(|g| g.fill.gradient.is_none());
    // A band goes where the glows explain its pixels better than it does, by more than it costs
    let removed: Vec<usize> = members
        .iter()
        .copied()
        .filter(|&m| {
            let change: f64 = bands.pixels[m]
                .iter()
                .map(|&i| (plain(ctx, i, under[i]) - plain(ctx, i, now[i])) as f64)
                .sum();
            !dots && change < price(ctx, &scene.shapes[m], costs)
        })
        .collect();
    let gone = |o: u32| removed.binary_search(&(o as usize)).is_ok();
    // The ground's holes the removed bands or shapes above fill, closed
    let base = &scene.shapes[k];
    let outer = outer_loop(base, w, h)?;
    let closed: Vec<usize> = (0..base.loops.len())
        .filter(|&l| l != outer)
        .filter(|&l| {
            // Closed where a removed band shows through it and nothing else does but shapes
            // above covering it wholly
            let inside = inside_loop(base, l, w, h);
            inside.iter().any(|&i| gone(bands.owner[i]))
                && inside.iter().all(|&i| {
                    let o = bands.owner[i];
                    gone(o)
                        || (o != NONE
                            && o as usize > k
                            && !members.contains(&(o as usize))
                            && scene.shapes[o as usize].raster().at(i, w) >= 1.)
                })
        })
        .collect();
    // What showed through a removed band's holes is an object over the glow: each such hole is
    // lifted over the glows, painted as what showed through it most
    let lifted: Vec<Shape> = through
        .iter()
        .filter(|t| gone(t.0 as u32))
        .filter_map(|(_, l, pixels)| {
            let mut owners: crate::HashMap<u32, usize> = crate::HashMap::default();
            for &i in pixels {
                *owners.entry(bands.owner[i]).or_default() += 1;
            }
            let (o, _) = owners.into_iter().max_by_key(|&(o, n)| (n, o))?;
            // Painted as what showed through, or flat in its own colour where that went too
            let fill = if o != NONE && !gone(o) {
                scene.shapes[o as usize].fill
            } else {
                let sum = pixels.iter().map(|&i| ctx.target[i]).sum::<Vec3>();
                crate::Fill::solid(sum / pixels.len() as f32)
            };
            let mut s = Shape::new(vec![l.clone()], fill);
            s.smooth = base.smooth;
            s.antialiased = base.antialiased;
            Some(s)
        })
        .collect();
    let mut flat = base.clone();
    if !dots {
        flat.fill = crate::Fill::solid(ground);
    }
    if !closed.is_empty() {
        flat.loops = (0..base.loops.len())
            .filter(|l| !closed.contains(l))
            .map(|l| base.loops[l].clone())
            .collect();
        flat.primitive = None;
        flat.update_bounds();
    }
    // Shapes below the ground seen whole (what it hides of them no more than a unit along their
    // rim) through its holes where the glows lie are objects over the glow, raised in order
    let unit = crate::translucency::raster::unit_of(w) as f32;
    let raised: Vec<usize> = (1..k)
        .filter(|&j| !dots && !gone(j as u32) && !members.contains(&j))
        .filter(|&j| {
            let r = scene.shapes[j].raster();
            let area: f32 = r.values.iter().sum();
            let rim = r.values.iter().filter(|&&v| v > 0.02 && v < 0.98).count() as f32;
            let seen = &bands.pixels[j];
            area - seen.len() as f32 <= unit * rim
                && glows.iter().any(|g| {
                    let c = g.raster();
                    seen.iter().any(|&i| c.at(i, w) >= 0.5)
                })
        })
        .collect();
    // Judged over everything that changes, as the planner prices a shape at all
    let (mut min, mut max) = if (kept || dots) && closed.is_empty() {
        (Vec2::INFINITY, Vec2::NEG_INFINITY)
    } else {
        (base.paint_min(), base.paint_max())
    };
    let moving = removed.iter().chain(&raised).map(|&m| &scene.shapes[m]);
    for s in moving.chain(&glows).chain(&lifted) {
        (min, max) = (min.min(s.paint_min()), max.max(s.paint_max()));
    }
    let window = Window::around(min, max, 1., w, h);
    let render = |scene: &Scene| scene.render(window, 0..scene.shapes.len(), [0.; 3]);
    let before = plain_error(ctx, window, &render(scene)) as f64
        + std::iter::once(k)
            .chain(removed.iter().copied())
            .map(|m| price(ctx, &scene.shapes[m], costs))
            .sum::<f64>();
    crate::progress::poll().await;
    let saved = scene.shapes.clone();
    let mut taken: Vec<usize> = removed.iter().chain(&raised).copied().collect();
    taken.sort_unstable();
    let moved: Vec<Shape> = raised.iter().map(|&j| scene.shapes[j].clone()).collect();
    for &j in taken.iter().rev() {
        scene.shapes.remove(j);
    }
    let at = k - taken.iter().filter(|&&m| m < k).count();
    let change = scene.change();
    scene.shapes[at] = Shape {
        changed: change,
        ..flat
    };
    // New shapes, the glows and then the lifted ones, priced; the raised ones only move
    let (mut count, mut lifts) = (glows.len(), lifted.len());
    let raised_count = moved.len();
    for (n, s) in glows.into_iter().chain(lifted).chain(moved).enumerate() {
        let change = scene.change();
        scene.shapes.insert(
            at + 1 + n,
            Shape {
                changed: change,
                ..s
            },
        );
    }
    let cost = |scene: &Scene, new: usize| {
        plain_error(ctx, window, &render(scene)) as f64
            + (at..=at + new)
                .map(|j| price(ctx, &scene.shapes[j], costs))
                .sum::<f64>()
    };
    // A new shape that does not pay for itself goes, a glow or a lifted hole
    let mut after = cost(scene, count + lifts);
    let mut j = at + 1;
    while j <= at + count + lifts {
        crate::progress::poll().await;
        let shape = scene.shapes.remove(j);
        scene.change();
        let without = cost(scene, count + lifts - 1);
        if without < after {
            after = without;
            if j <= at + count {
                count -= 1;
            } else {
                lifts -= 1;
            }
        } else {
            scene.shapes.insert(j, shape);
            scene.change();
            j += 1;
        }
    }
    if crate::diagnostics::log() {
        eprintln!(
            "glow: ground {k} at {min:.0?}..{max:.0?}, {} of {} bands, {count} glows, {} holes closed, {lifts} lifted: {before:.2} to {after:.2}",
            removed.len(),
            members.len(),
            closed.len(),
        );
    }
    if count > 0 && after < before {
        return Some(at + count + lifts + raised_count);
    }
    scene.shapes = saved;
    scene.change();
    None
}

/// Replaces grounds of the decided `scene` by their flat colour and glows over it where that
/// lowers the error plus price (see [`propose`]), walked from the bottom. Returns how many.
pub async fn glow_scene(ctx: &Context, scene: &mut Scene) -> usize {
    let mut bands = Bands::of(ctx, scene);
    let full = Window::full(ctx.width, ctx.height);
    let mut now = scene.render(full, 0..scene.shapes.len(), [1.; 3]);
    let (mut done, mut k) = (0, 0);
    while k < scene.shapes.len() {
        crate::progress::poll().await;
        if is_band(&scene.shapes[k])
            && let Some(top) = propose(ctx, scene, &bands, &now, k).await
        {
            done += 1;
            crate::progress::poll().await;
            bands = Bands::of(ctx, scene);
            crate::progress::poll().await;
            now = scene.render(full, 0..scene.shapes.len(), [1.; 3]);
            k = top;
        }
        k += 1;
    }
    done
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::*;

    /// A lamp's halo over a dark ground, which the trace posterized into nested discs over the
    /// ground, is drawn as the flat ground and one glow fading to transparent, nearer the
    /// picture and with fewer shapes.
    #[test]
    fn a_posterized_halo_becomes_one_glow() {
        let (w, c) = (96usize, Vec2::splat(48.));
        let (ground, light) = (Vec3::new(0.16, 0.13, 0.2), Vec3::new(0.95, 0.8, 0.3));
        let at = |p: Vec2| ground.lerp(light, (1. - p.distance(c) / 30.).clamp(0., 1.));
        let rgb: Vec<u8> = (0..w * w)
            .flat_map(|i| {
                let p = Vec2::new((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                at(p).to_array().map(|x| (x * 255.).round() as u8)
            })
            .collect();
        let ctx = Context::new(&rgb, w, w, 1.);
        let mut scene = Scene::with_background(w, w, ground);
        for k in 0..4 {
            let r = 30. * (1. - k as f32 / 4.);
            let circle = crate::tracer::primitives::Primitive::Circle {
                center: c,
                radius: r,
            };
            let p = c + Vec2::X * (r - 30. / 8.);
            scene.push(Shape::new(vec![circle.cubics()], at(p)));
        }
        let full = Window::full(w, w);
        let before = ctx.error(full, &scene.render(full, 0..scene.shapes.len(), [0.; 3]));
        assert_eq!(crate::progress::finish(glow_scene(&ctx, &mut scene)), 1);
        let after = ctx.error(full, &scene.render(full, 0..scene.shapes.len(), [0.; 3]));
        assert!(scene.shapes.len() < 5, "{} shapes", scene.shapes.len());
        let glow = scene
            .shapes
            .iter()
            .find(|s| !s.fill.is_opaque())
            .expect("a glow");
        assert_eq!(glow.fill.gradient.unwrap().to()[3], 0.);
        assert!(after < before, "{after} {before}");
    }

    /// A small flat dot the trace left in its ground is drawn as a flat disc, not a glow.
    #[test]
    fn a_flat_dot_is_a_disc() {
        let (w, c, r) = (48usize, Vec2::new(24.3, 23.8), 4.5);
        let (ground, dot) = (Vec3::new(0.8, 0.78, 0.65), Vec3::new(0.55, 0.62, 0.43));
        let circle = crate::tracer::primitives::Primitive::Circle {
            center: c,
            radius: r,
        };
        let disc = Shape::new(vec![circle.cubics()], dot);
        let mut truth = Scene::with_background(w, w, ground);
        truth.push(disc);
        let full = Window::full(w, w);
        let rgb: Vec<u8> = truth
            .render(full, 0..2, [0.; 3])
            .iter()
            .flat_map(|v| v.to_array().map(|x| (x * 255.).round() as u8))
            .collect();
        let ctx = Context::new(&rgb, w, w, 1.);
        let mut scene = Scene::with_background(w, w, ground);
        assert_eq!(crate::progress::finish(glow_scene(&ctx, &mut scene)), 1);
        assert_eq!(scene.shapes.len(), 2);
        let s = &scene.shapes[1];
        assert!(s.fill.gradient.is_none(), "{:?}", s.fill);
        assert!(
            (s.fill.color - dot).abs().max_element() < 0.03,
            "{:?}",
            s.fill
        );
        let size = s.max - s.min;
        assert!((size.x - 2. * r).abs() < 1.5, "{size}");
    }
}
