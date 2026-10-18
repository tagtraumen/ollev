//! Structure in the instances' transforms: equal steps along a line (a linear repeat), a full
//! lattice of two steps (a grid repeat), or equal turns about a point (a circular repeat), each
//! kept where moving the instances onto it costs less error than the transforms it saves.

use glam::{DVec2, Vec2};

use super::snap::{Site, stand_in};
use super::{ALIGN, Repeat, Structure, Transform, overlaps};
use crate::{Context, Scene};

/// Translations on a lattice: `origin + i * a + j * b` for each point's `(i, j)`, with `b`
/// absent for a row, for the points lying on it (`inliers`, in order).
pub(super) struct Lattice {
    pub origin: Vec2,
    pub a: Vec2,
    pub b: Option<Vec2>,
    pub index: Vec<(i32, i32)>,
    pub inliers: Vec<usize>,
}

/// The vector most of `vectors` lie within [`ALIGN`] of, averaged over those.
fn most_common(vectors: &[Vec2]) -> Option<Vec2> {
    let near = |v: Vec2| vectors.iter().filter(move |w| w.distance(v) <= ALIGN);
    let best = vectors.iter().max_by_key(|&&v| near(v).count())?;
    let close: Vec<Vec2> = near(*best).copied().collect();
    Some(close.iter().sum::<Vec2>() / close.len() as f32)
}

/// `v` turned into the half plane of positive x (or positive y on the axis), so a step and its
/// opposite are one.
fn canonical(v: Vec2) -> Vec2 {
    if v.x < 0. || (v.x == 0. && v.y < 0.) {
        -v
    } else {
        v
    }
}

/// The lattice the points lie on, each within [`ALIGN`] of its node, if any: the most common
/// nearest-neighbour step, and the most common step off its line.
pub(super) fn lattice(points: &[Vec2]) -> Option<Lattice> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    let nearest_step = |i: usize, off: Option<Vec2>| -> Option<Vec2> {
        (0..n)
            .filter(|&j| j != i)
            .map(|j| points[j] - points[i])
            .filter(|v| off.is_none_or(|a| a.normalize().perp_dot(*v).abs() > 2. * ALIGN))
            .min_by(|u, v| u.length_squared().total_cmp(&v.length_squared()))
            .map(canonical)
    };
    let steps: Vec<Vec2> = (0..n).filter_map(|i| nearest_step(i, None)).collect();
    let a = most_common(&steps)?;
    if a.length() <= 2. * ALIGN {
        return None;
    }
    let off: Vec<Vec2> = (0..n).filter_map(|i| nearest_step(i, Some(a))).collect();
    let b = most_common(&off).filter(|b| {
        let share = off.iter().filter(|v| v.distance(*b) <= ALIGN).count();
        // a second direction most points step along, not the odd point beside a row
        2 * share >= n
    });
    // Each point's indices on the lattice through the first
    let basis = |v: Vec2| -> (f32, f32) {
        match b {
            Some(b) => {
                let det = a.perp_dot(b);
                (v.perp_dot(b) / det, a.perp_dot(v) / det)
            }
            None => (v.dot(a) / a.length_squared(), 0.),
        }
    };
    let index: Vec<(i32, i32)> = points
        .iter()
        .map(|p| {
            let (i, j) = basis(*p - points[0]);
            (i.round() as i32, j.round() as i32)
        })
        .collect();
    let mut seen = index.clone();
    seen.sort();
    seen.dedup();
    if seen.len() != n {
        return None;
    }
    // Fitted to all, then to the points within reach of their nodes
    let mut inliers: Vec<usize> = (0..n).collect();
    for _ in 0..2 {
        let (p, k): (Vec<Vec2>, Vec<(i32, i32)>) =
            inliers.iter().map(|&i| (points[i], index[i])).unzip();
        let (origin, a, b) = fit(&p, &k, b.is_some())?;
        let at =
            |(i, j): (i32, i32)| origin + i as f32 * a + b.map_or(Vec2::ZERO, |b| j as f32 * b);
        inliers = (0..n)
            .filter(|&i| points[i].distance(at(index[i])) <= ALIGN)
            .collect();
        if inliers.len() < 3 {
            return None;
        }
    }
    let (p, k): (Vec<Vec2>, Vec<(i32, i32)>) =
        inliers.iter().map(|&i| (points[i], index[i])).unzip();
    let (i0, j0) = (k.iter().map(|k| k.0).min()?, k.iter().map(|k| k.1).min()?);
    let k: Vec<(i32, i32)> = k.iter().map(|&(i, j)| (i - i0, j - j0)).collect();
    let (origin, a, b) = fit(&p, &k, b.is_some())?;
    let lattice = Lattice {
        origin,
        a,
        b,
        index: k,
        inliers,
    };
    let fits = p
        .iter()
        .zip(&lattice.index)
        .all(|(p, &k)| p.distance(lattice.at(k)) <= ALIGN);
    fits.then_some(lattice)
}

impl Lattice {
    pub fn at(&self, (i, j): (i32, i32)) -> Vec2 {
        self.origin + i as f32 * self.a + self.b.map_or(Vec2::ZERO, |b| j as f32 * b)
    }
}

/// The origin and steps that put the points nearest their indices, least squares.
fn fit(points: &[Vec2], index: &[(i32, i32)], two: bool) -> Option<(Vec2, Vec2, Option<Vec2>)> {
    // Normal equations of p = o + i a + j b, the same for x and y
    let m = if two { 3 } else { 2 };
    let mut ata = [[0f64; 3]; 3];
    let mut atp = [DVec2::ZERO; 3];
    for (p, &(i, j)) in points.iter().zip(index) {
        let row = [1., i as f64, j as f64];
        for r in 0..m {
            for c in 0..m {
                ata[r][c] += row[r] * row[c];
            }
            atp[r] += row[r] * p.as_dvec2();
        }
    }
    let x = solve(&ata, &atp, m)?;
    Some((x[0].as_vec2(), x[1].as_vec2(), two.then(|| x[2].as_vec2())))
}

/// Solves the `m` by `m` system by Gaussian elimination with partial pivoting.
fn solve(a: &[[f64; 3]; 3], b: &[DVec2; 3], m: usize) -> Option<[DVec2; 3]> {
    let (mut a, mut b) = (*a, *b);
    for c in 0..m {
        let p = (c..m).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-9 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in 0..m {
            if r != c {
                let f = a[r][c] / a[c][c];
                let pivot = a[c];
                for (v, p) in a[r].iter_mut().zip(pivot).take(m) {
                    *v -= f * p;
                }
                let pb = b[c];
                b[r] -= f * pb;
            }
        }
    }
    let mut x = [DVec2::ZERO; 3];
    for c in 0..m {
        x[c] = b[c] / a[c][c];
    }
    Some(x)
}

/// Whether every transform has the first's rotation and scale, as far as the outline of
/// `reach` shows.
fn same_frame(transforms: &[Transform], reach: f32) -> bool {
    let t0 = transforms[0];
    transforms.iter().all(|t| {
        (t.angle - t0.angle).abs() * reach <= ALIGN / 2.
            && (t.scale - t0.scale).abs() * reach <= ALIGN / 2.
    })
}

/// The members in the structure's order with the transforms it gives, for a row or a full grid
/// of translations.
fn on_lattice(repeat: &Repeat, reach: f32) -> Option<(Structure, Vec<(usize, Transform)>)> {
    let transforms: Vec<Transform> = repeat.members.iter().map(|m| m.1).collect();
    if !same_frame(&transforms, reach) {
        return None;
    }
    let offsets: Vec<Vec2> = transforms.iter().map(|t| t.offset).collect();
    let l = lattice(&offsets).filter(|l| l.inliers.len() == offsets.len())?;
    let n = offsets.len() as i32;
    let columns = l.index.iter().map(|k| k.0).max()? + 1;
    let rows = l.index.iter().map(|k| k.1).max()? + 1;
    // full: every node of the box holds a member
    if columns * rows != n {
        return None;
    }
    let mut members: Vec<(usize, Transform, i32)> = repeat
        .members
        .iter()
        .zip(&l.index)
        .map(|(&(id, _), &k)| {
            let placed = Transform {
                offset: l.at(k),
                ..transforms[0]
            };
            (id, placed, k.1 * columns + k.0)
        })
        .collect();
    members.sort_by_key(|m| m.2);
    let structure = match l.b {
        Some(b) if rows > 1 => Structure::Grid {
            a: l.a,
            b,
            columns: columns as usize,
        },
        _ => Structure::Linear { step: l.a },
    };
    Some((structure, members.into_iter().map(|m| (m.0, m.1)).collect()))
}

/// The members in turn order with the transforms of equal turns about one centre, if the
/// transforms turn by multiples of one step and their offsets go round that centre.
fn on_circle(repeat: &Repeat, reach: f32) -> Option<(Structure, Vec<(usize, Transform)>)> {
    let mut members = repeat.members.clone();
    let t0 = members[0].1;
    if members
        .iter()
        .any(|m| (m.1.scale - t0.scale).abs() * reach > ALIGN / 2.)
    {
        return None;
    }
    let tau = std::f32::consts::TAU;
    members.sort_by(|a, b| {
        a.1.angle
            .rem_euclid(tau)
            .total_cmp(&b.1.angle.rem_euclid(tau))
    });
    let first = members[0].1;
    let turns: Vec<f32> = members
        .iter()
        .map(|m| (m.1.angle - first.angle).rem_euclid(tau))
        .collect();
    let step = turns
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f32::INFINITY, f32::min);
    if step * reach <= ALIGN {
        return None;
    }
    let ks: Vec<i32> = turns.iter().map(|t| (t / step).round() as i32).collect();
    if ks.iter().enumerate().any(|(i, &k)| k != i as i32) {
        return None;
    }
    let n = members.len() as f32;
    let step = turns
        .iter()
        .zip(&ks)
        .skip(1)
        .map(|(t, &k)| t / k as f32)
        .sum::<f32>()
        / (n - 1.);
    if step * (n - 1.) > tau - step / 2. + ALIGN / reach {
        return None;
    }
    // The centre c with o_k - R_k o_0 = (I - R_k) c, least squares
    let (mut m, mut v) = ([[0f64; 2]; 2], DVec2::ZERO);
    for (k, member) in members.iter().enumerate().skip(1) {
        let r = Vec2::from_angle(k as f32 * step);
        let rhs = (member.1.offset - r.rotate(first.offset)).as_dvec2();
        // I - R as rows
        let (c, s) = (r.x as f64, r.y as f64);
        let rows = [[1. - c, s], [-s, 1. - c]];
        let rhs = [rhs.x, rhs.y];
        for r_i in 0..2 {
            for a in 0..2 {
                for b in 0..2 {
                    m[a][b] += rows[r_i][a] * rows[r_i][b];
                }
                v[a] += rows[r_i][a] * rhs[r_i];
            }
        }
    }
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-9 {
        return None;
    }
    let center = DVec2::new(
        (v.x * m[1][1] - v.y * m[0][1]) / det,
        (m[0][0] * v.y - m[1][0] * v.x) / det,
    )
    .as_vec2();
    let placed: Vec<(usize, Transform)> = members
        .iter()
        .enumerate()
        .map(|(k, m)| {
            let r = Vec2::from_angle(k as f32 * step);
            (
                m.0,
                Transform {
                    offset: center + r.rotate(first.offset - center),
                    angle: first.angle + k as f32 * step,
                    scale: first.scale,
                },
            )
        })
        .collect();
    let fits = members
        .iter()
        .zip(&placed)
        .all(|(m, p)| m.1.offset.distance(p.1.offset) <= ALIGN);
    fits.then_some((Structure::Circular { center, step }, placed))
}

/// Moves the members (in their structure's order) next to one another in the stack at the
/// topmost one's place, if no other shape between them overlaps one, and members that overlap
/// are stacked in that order; whether it did.
fn gather(scene: &mut Scene, members: &[(usize, Transform)]) -> bool {
    let width = scene.width;
    let ids: crate::HashSet<usize> = members.iter().map(|m| m.0).collect();
    let positions: Vec<usize> = members.iter().map(|m| scene.position(m.0)).collect();
    let top = *positions.iter().max().expect("members");
    for (k, &p) in positions.iter().enumerate() {
        let shape = &scene.shapes[p];
        let blocked = (p + 1..top).any(|q| {
            !ids.contains(&scene.shapes[q].id) && overlaps(shape, &scene.shapes[q], width)
        });
        let misordered = positions[k + 1..]
            .iter()
            .any(|&q| q < p && overlaps(shape, &scene.shapes[q], width));
        if blocked || misordered {
            return false;
        }
    }
    let mut moved: Vec<crate::Shape> = members
        .iter()
        .map(|m| scene.shapes[scene.position(m.0)].clone())
        .collect();
    let mut shapes = Vec::with_capacity(scene.shapes.len());
    for (q, shape) in std::mem::take(&mut scene.shapes).into_iter().enumerate() {
        if q == top {
            shapes.append(&mut moved);
        } else if !ids.contains(&shape.id) {
            shapes.push(shape);
        }
    }
    scene.shapes = shapes;
    scene.change();
    true
}

/// The repeat drawn together in the stack: with the structure its transforms show where moving
/// the instances onto it pays, else as a list; none where the stack keeps the instances apart.
pub(super) fn recognize(ctx: &Context, scene: &mut Scene, repeat: Repeat) -> Option<Repeat> {
    if let Some(structured) = structured(ctx, scene, &repeat) {
        return Some(structured);
    }
    let mut members = repeat.members.clone();
    members.sort_by_key(|m| scene.position(m.0));
    gather(scene, &members).then_some(Repeat { members, ..repeat })
}

/// The repeat on the structure its transforms show, its instances moved onto it and gathered
/// in the stack, where that pays and the stack lets them.
fn structured(ctx: &Context, scene: &mut Scene, repeat: &Repeat) -> Option<Repeat> {
    if repeat.members.len() < 3 {
        return None;
    }
    let reach = repeat
        .prototype
        .loops
        .iter()
        .flatten()
        .flatten()
        .map(|p| p.length())
        .fold(1., f32::max);
    let (structure, placed) = on_lattice(repeat, reach).or_else(|| on_circle(repeat, reach))?;
    let candidate = Repeat {
        structure,
        members: placed,
        prototype: repeat.prototype.clone(),
    };
    let saved = (repeat.segments() - candidate.segments()) * ctx.accept_cost();
    // The error the instances gain moved onto the structure, each against the scene as it is
    let current: crate::HashMap<usize, Transform> = repeat.members.iter().copied().collect();
    let mut rise = 0.;
    let mut stand_ins = Vec::new();
    for &(id, t) in &candidate.members {
        let position = scene.position(id);
        let member = scene.shapes[position].clone();
        if t == current[&id] {
            continue;
        }
        let placed = stand_in(&repeat.prototype, &member, t);
        let site = Site::new(scene, position, placed.min, placed.max);
        rise += site.change(&placed);
        stand_ins.push((position, placed));
        if rise >= saved {
            return None;
        }
    }
    let kept = scene.shapes.clone();
    for (position, placed) in stand_ins {
        scene.shapes[position] = placed;
    }
    scene.change();
    if !gather(scene, &candidate.members) {
        scene.shapes = kept;
        scene.change();
        return None;
    }
    Some(candidate)
}
