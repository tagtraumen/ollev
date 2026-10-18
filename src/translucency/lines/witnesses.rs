//! The colours either side of a link and the relations they solve: the witnesses.

use super::*;

// ----------------------------------------------------------------------------- colors

/// The colors of a chain's part near end `e`, oriented along the path leaving the chain by
/// `e`: medians over the points `skip` to `skip + window` px from the end, as rows (left,
/// right, and the centre of a ridge), with their median deviation and the points used.
pub fn part_colours(ch: &Chain, e: usize, window: f64, skip: f64) -> (Vec<DVec3>, f64, Vec<bool>) {
    let last = ch.length;
    let d: Vec<f64> = if e == 1 {
        ch.arc.iter().map(|a| last - a).collect()
    } else {
        ch.arc.iter().map(|a| last - (last - a)).collect()
    };
    let mut sel: Vec<bool> = d.iter().map(|&v| v >= skip && v <= skip + window).collect();
    if sel.iter().filter(|s| **s).count() < 3 {
        sel = d.iter().map(|&v| v <= skip + window).collect();
    }
    let pick = |v: &[DVec3]| {
        v.iter()
            .zip(&sel)
            .filter(|(_, s)| **s)
            .map(|(c, _)| *c)
            .collect::<Vec<_>>()
    };
    let (mut left, mut right) = (pick(&ch.left), pick(&ch.right));
    // leaving by end 1 the path runs along the chain, by end 0 against it: left is right
    if e == 0 {
        std::mem::swap(&mut left, &mut right);
    }
    let dev_of = |v: &[DVec3], r: DVec3| {
        median_rows(&v.iter().map(|c| (*c - r).abs()).collect::<Vec<_>>()).max_element()
    };
    let mut rows = vec![median_rows(&left), median_rows(&right)];
    let mut dev = vec![dev_of(&left, rows[0]), dev_of(&right, rows[1])];
    if let Some(c) = &ch.centre {
        let cc = pick(c);
        rows.push(median_rows(&cc));
        dev.push(dev_of(&cc, rows[2]));
    }
    (rows, max_of(&dev), sel)
}

/// [`part_colours`] for the part a path enters by end `e`: oriented along the path, which
/// runs into the chain by `e` (the opposite of leaving by it).
pub fn entry_colours(ch: &Chain, e: usize, window: f64, skip: f64) -> (Vec<DVec3>, f64, Vec<bool>) {
    let (mut rows, dev, sel) = part_colours(ch, e, window, skip);
    rows.swap(0, 1);
    (rows, dev, sel)
}

/// The relation between two parts of one line, or why it failed.
#[derive(Clone, Copy, Debug)]
pub struct Solved {
    pub k: Option<f64>,
    pub m: Option<DVec3>,
    /// 1 where the second part is inside, 0 where the first is
    pub inside: usize,
    pub why: Option<&'static str>,
}

/// The relation between two parts' color rows (left, right, a ridge's centre): the part with
/// the smaller steps is inside, k by least squares over the steps. Both sides must change: a
/// change on one side alone is another region beginning (a T), which any k would read.
pub fn solve_relation(a: &[DVec3], b: &[DVec3], slack: f64) -> Solved {
    let center = |v: &[DVec3]| {
        let mu = rows_mean(v);
        v.iter().map(|r| *r - mu).collect::<Vec<_>>()
    };
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    let (mut ac, mut bc) = (center(&a), center(&b));
    let sq = |v: &[DVec3]| sum_all3(&v.iter().map(|r| *r * *r).collect::<Vec<_>>());
    let mut inside = 1;
    if sq(&ac) < sq(&bc) {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut ac, &mut bc);
        inside = 0;
    }
    // now a outside, b inside
    let den = sq(&ac);
    if den <= 0.0 {
        return Solved {
            k: None,
            m: None,
            inside,
            why: Some("flat"),
        };
    }
    let k = sum_all3(&ac.iter().zip(&bc).map(|(x, y)| *x * *y).collect::<Vec<_>>()) / den;
    let m = rows_mean(
        &b.iter()
            .zip(&a)
            .map(|(y, x)| *y - k * *x)
            .collect::<Vec<_>>(),
    );
    let res = bc
        .iter()
        .zip(&ac)
        .map(|(y, x)| (*y - k * *x).abs().max_element())
        .fold(f64::NEG_INFINITY, f64::max);
    let out = |why: Option<&'static str>| Solved {
        k: Some(k),
        m: Some(m),
        inside,
        why,
    };
    let one_side = (0..2)
        .map(|r| cheb(b[r], a[r]))
        .fold(f64::INFINITY, f64::min);
    if one_side <= TOL {
        return out(Some("one side"));
    }
    let lo = ac
        .iter()
        .copied()
        .fold(DVec3::splat(f64::INFINITY), DVec3::min);
    let hi = ac
        .iter()
        .copied()
        .fold(DVec3::splat(f64::NEG_INFINITY), DVec3::max);
    let spread = hi - lo;
    if (0..3).filter(|&c| spread[c] >= 4.0 * NOISE).count() < 2 {
        return out(Some("one channel"));
    }
    for (x, y) in ac.iter().zip(&bc) {
        for c in 0..3 {
            if x[c].abs() > 2.0 * NOISE
                && y[c].abs() > 2.0 * NOISE
                && x[c].signum() != y[c].signum()
            {
                return out(Some("polarity"));
            }
        }
    }
    if !(0.05..=0.95).contains(&k) {
        return out(Some("k range"));
    }
    if res > TOL + slack {
        return out(Some("channels"));
    }
    let c = m / (1.0 - k);
    let margin = NOISE / (1.0 - k);
    if c.min_element() < -margin || c.max_element() > 1.0 + margin {
        return out(Some("gamut"));
    }
    out(None)
}

/// The Hermite cubic from end pa (outward heading ha) to end pb (outward heading hb),
/// handles a third of the gap along the headings: the line's course across the gap. Its
/// points and normals at t from 0.25 to 0.75.
pub(super) fn bridge(
    pa: DVec2,
    ha: DVec2,
    pb: DVec2,
    hb: DVec2,
    n: usize,
) -> (Vec<DVec2>, Vec<DVec2>) {
    let g = hyp(pb - pa);
    let c = [pa, pa + g / 3.0 * ha, pb + g / 3.0 * hb, pb];
    let ts = linspace(0.25, 0.75, n);
    let p = ts.iter().map(|&t| bez(&c, t)).collect();
    let nrm = ts
        .iter()
        .map(|&t| {
            let d = bez_d1(&c, t);
            let d = d / hyp(d).max(1e-9);
            DVec2::new(-d.y, d.x)
        })
        .collect();
    (p, nrm)
}

/// One witness, of three kinds: two linked parts whose sides both change, a line alike across a
/// gap whose course over it shows both sides changed, one chain whose sides both step. Kept:
/// the reason it failed, if it did, whether it solved, and the inside and outside sample points.
#[derive(Clone, Debug)]
pub struct Witness {
    pub why: Option<&'static str>,
    /// Whether the relation was solved (its k and m found).
    pub solved: bool,
    pub inside: Vec<DVec2>,
    pub outside: Vec<DVec2>,
}

pub(super) fn side_points(ch: &Chain, sel: &[bool]) -> Vec<DVec2> {
    let idx: Vec<usize> = (0..sel.len()).filter(|&i| sel[i]).collect();
    let d = ch.side_d;
    let mut out: Vec<DVec2> = idx.iter().map(|&i| ch.p[i] + d * ch.normal[i]).collect();
    out.extend(idx.iter().map(|&i| ch.p[i] - d * ch.normal[i]));
    out
}

/// The witnesses of an image's chains and links: per link the two parts' colors near the
/// link, solved; links whose parts agree give a bridge witness when the bridge's sides
/// differ; and steps along single chains. Failed candidates are kept with `why`.
pub fn line_witnesses(img: &Image, chains: &[Chain], links: &[Link]) -> Vec<Witness> {
    let u = unit_of(img.width);
    let s = img.width as f64 / 330.0;
    // the arms of a crossing read at 330 run 3 to 12 px; in proportion above
    let window = 10.0 * s;
    let mut out = Vec::new();
    for l in links {
        let ((i, ea), (j, eb)) = (l.a, l.b);
        let (a, b) = (&chains[i], &chains[j]);
        let (mut ra, da, sa) = part_colours(a, ea, window, u);
        let (mut rb, db, sb) = entry_colours(b, eb, window, u);
        // a ridge's centre is a pair only with another ridge's
        let rows = ra.len().min(rb.len());
        ra.truncate(rows);
        rb.truncate(rows);
        let same = ra
            .iter()
            .zip(&rb)
            .map(|(x, y)| cheb(*x, *y))
            .fold(f64::NEG_INFINITY, f64::max);
        if same <= TOL {
            // the same line on both sides of the gap: a bridge, read along its course
            let (x, y) = (a.ends[ea], b.ends[eb]);
            let g = hyp(y.p - x.p);
            if g <= JUNCTION_GAP * u {
                continue;
            }
            let (p, nrm) = bridge(x.p, x.heading, y.p, y.heading, 24);
            let d = a.side_d.max(b.side_d);
            let lb_: Vec<DVec3> = p
                .iter()
                .zip(&nrm)
                .map(|(q, n)| img.sample(q.x + d * n.x, q.y + d * n.y))
                .collect();
            let rb_: Vec<DVec3> = p
                .iter()
                .zip(&nrm)
                .map(|(q, n)| img.sample(q.x - d * n.x, q.y - d * n.y))
                .collect();
            let mut bm = vec![median_rows(&lb_), median_rows(&rb_)];
            let mut am = vec![0.5 * (ra[0] + rb[0]), 0.5 * (ra[1] + rb[1])];
            if a.centre.is_some() && b.centre.is_some() {
                let cb: Vec<DVec3> = p.iter().map(|q| img.sample(q.x, q.y)).collect();
                bm.push(median_rows(&cb));
                am.push(0.5 * (ra[2] + rb[2]));
            }
            let mut sv = solve_relation(&am, &bm, da.max(db));
            if sv.why.is_none() && sv.inside != 1 {
                sv.why = Some("bridge outside");
            }
            let mut inside: Vec<DVec2> = p.iter().zip(&nrm).map(|(q, n)| *q + d * *n).collect();
            inside.extend(p.iter().zip(&nrm).map(|(q, n)| *q - d * *n));
            let mut outside = side_points(a, &sa);
            outside.extend(side_points(b, &sb));
            out.push(Witness {
                why: sv.why,
                solved: sv.k.is_some() && sv.m.is_some(),
                inside,
                outside,
            });
            continue;
        }
        let sv = solve_relation(&ra, &rb, da.max(db));
        let (pin, pout) = if sv.inside == 1 {
            (side_points(b, &sb), side_points(a, &sa))
        } else {
            (side_points(a, &sa), side_points(b, &sb))
        };
        out.push(Witness {
            why: sv.why,
            solved: sv.k.is_some() && sv.m.is_some(),
            inside: pin,
            outside: pout,
        });
    }
    for ch in chains {
        out.extend(step_witnesses(ch, window, 2.0 * u));
    }
    out
}

/// Places along one chain where both sides step: the medians of windows before and after,
/// `skip` clear of the place, at a local maximum of the smaller side's change above twice
/// the windows' spread (a gradual shading spreads them too). Solved as a self link.
pub fn step_witnesses(ch: &Chain, window: f64, skip: f64) -> Vec<Witness> {
    let a = &ch.arc;
    if ch.length < 2.0 * (window + skip) + 2.0 * ch.unit {
        return Vec::new();
    }
    let ts = arange(window + skip, ch.length - window - skip, 1.0);
    let windows = |t: f64| {
        let before: Vec<bool> = a
            .iter()
            .map(|&v| v >= t - skip - window && v <= t - skip)
            .collect();
        let after: Vec<bool> = a
            .iter()
            .map(|&v| v >= t + skip && v <= t + skip + window)
            .collect();
        (before, after)
    };
    let pick = |v: &[DVec3], sel: &[bool]| {
        v.iter()
            .zip(sel)
            .filter(|(_, s)| **s)
            .map(|(c, _)| *c)
            .collect::<Vec<_>>()
    };
    let count = |sel: &[bool]| sel.iter().filter(|s| **s).count();
    let cand: Vec<f64> = ts
        .iter()
        .map(|&t| {
            let (before, after) = windows(t);
            if count(&before) < 3 || count(&after) < 3 {
                return 0.0;
            }
            let dl = cheb(
                median_rows(&pick(&ch.left, &before)),
                median_rows(&pick(&ch.left, &after)),
            );
            let dr = cheb(
                median_rows(&pick(&ch.right, &before)),
                median_rows(&pick(&ch.right, &after)),
            );
            dl.min(dr)
        })
        .collect();
    let mut out = Vec::new();
    let wi = window as usize;
    for i in 0..ts.len() {
        if cand[i] <= TOL {
            continue;
        }
        let (lo, hi) = (i.saturating_sub(wi), ts.len().min(i + wi + 1));
        if cand[i] < max_of(&cand[lo..hi]) || cand[lo..i].contains(&cand[i]) {
            continue;
        }
        let (before, after) = windows(ts[i]);
        let mut rows = Vec::new();
        let mut devs = Vec::new();
        for sel in [&before, &after] {
            let (l, r) = (pick(&ch.left, sel), pick(&ch.right, sel));
            let mut rr = vec![median_rows(&l), median_rows(&r)];
            let dev_of = |v: &[DVec3], m: DVec3| {
                median_rows(&v.iter().map(|c| (*c - m).abs()).collect::<Vec<_>>()).max_element()
            };
            let mut dv = vec![dev_of(&l, rr[0]), dev_of(&r, rr[1])];
            if let Some(c) = &ch.centre {
                let cc = pick(c, sel);
                rr.push(median_rows(&cc));
                dv.push(dev_of(&cc, rr[2]));
            }
            rows.push(rr);
            devs.push(max_of(&dv));
        }
        let dmax = devs[0].max(devs[1]);
        if cand[i] <= 2.0 * dmax {
            continue;
        }
        let sv = solve_relation(&rows[0], &rows[1], dmax);
        let (pin, pout) = if sv.inside == 1 {
            (side_points(ch, &after), side_points(ch, &before))
        } else {
            (side_points(ch, &before), side_points(ch, &after))
        };
        out.push(Witness {
            why: sv.why,
            solved: sv.k.is_some() && sv.m.is_some(),
            inside: pin,
            outside: pout,
        });
    }
    out
}
