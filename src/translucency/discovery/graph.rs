//! The equation graph: relations from the contacts of planar regions, and the best cut.

use super::*;

/// A directed pair of leaves supporting a relation, as the cuts see it: the inside `a`, the
/// outside `b`, the contact, the two colors compared and how far the inside's plane carries
/// its color across it as far as the background's does too (a gradient beneath the shape).
#[derive(Clone, Copy, Debug)]
pub struct Support {
    pub a: u32,
    pub b: u32,
    pub len: f32,
    pub bg: Vec3,
    pub img: Vec3,
    pub span: f32,
}

/// The leaf graph for the cuts: per leaf its neighbors with the contact length.
pub struct Graph {
    pub reach: Vec<Vec<(u32, f32)>>,
}

impl Graph {
    pub fn of(lv: &Leaves) -> Self {
        let mut reach = vec![Vec::new(); lv.n];
        for p in &lv.pairs {
            reach[p.x as usize].push((p.y, p.len as f32));
        }
        Graph { reach }
    }
}

/// The best connected group of `under` leaves: the contact of the pairs leaving it for leaves
/// not under the shape, and those pairs. A group counts only if its pairs span two backgrounds
/// and two images (show-through, not one opaque color), or one of them is graded.
pub(super) fn cut_score(
    supporting: &[Support],
    under: &HashSet<u32>,
    graph: &Graph,
) -> (f32, Vec<Support>) {
    let mut comp: HashMap<u32, u32> = HashMap::default();
    for p in supporting {
        if !under.contains(&p.a) || under.contains(&p.b) || comp.contains_key(&p.a) {
            continue;
        }
        comp.insert(p.a, p.a);
        let mut stack = vec![p.a];
        while let Some(x) = stack.pop() {
            for &(nb, _) in &graph.reach[x as usize] {
                if under.contains(&nb) && !comp.contains_key(&nb) {
                    comp.insert(nb, p.a);
                    stack.push(nb);
                }
            }
        }
    }
    // The groups in the order their first pair appears
    let mut order: Vec<u32> = Vec::new();
    let mut groups: HashMap<u32, Vec<Support>> = HashMap::default();
    for p in supporting {
        if under.contains(&p.a) && !under.contains(&p.b) {
            let g = comp[&p.a];
            groups
                .entry(g)
                .or_insert_with(|| {
                    order.push(g);
                    Vec::new()
                })
                .push(*p);
        }
    }
    let (mut best, mut best_members) = (0f32, Vec::new());
    for g in order {
        let members = &groups[&g];
        let bs: HashSet<IVec3> = members.iter().map(|p| bin(p.bg, 16.)).collect();
        let os: HashSet<IVec3> = members.iter().map(|p| bin(p.img, 16.)).collect();
        // Or one pair whose inside follows a gradient beneath it, a range of backgrounds
        // under one leaf (as saturn's ring, in pieces of one pair each)
        let graded = members.iter().any(|p| p.span > 2. * NOISE);
        if (bs.len() < 2 || os.len() < 2) && !graded {
            continue;
        }
        let total: f32 = members.iter().map(|p| p.len).sum();
        // Count only what an opaque shape could not explain as well: the largest class of one
        // flat inside color counts only as far as the second matches it, graded insides always
        // (so glassware's grout network does not lead on its contact alone)
        let mut classes: Vec<(Vec3, f32)> = Vec::new();
        for p in members {
            if p.span > 2. * NOISE {
                continue;
            }
            match classes
                .iter_mut()
                .find(|c| (c.0 - p.img).abs().max_element() <= 3. * NOISE)
            {
                Some(c) => c.1 += p.len,
                None => classes.push((p.img, p.len)),
            }
        }
        let mut sizes: Vec<f32> = classes.iter().map(|c| c.1).collect();
        sizes.sort_by(|a, b| b.total_cmp(a));
        sizes.extend([0., 0.]);
        let mut evidence = total - (sizes[0] - sizes[1]);
        if evidence <= 0. {
            continue;
        }
        // A translucent shape is a cut whose edges all carry its relation: the group weighs
        // by the share of its boundary its supporting pairs explain
        let boundary: f32 = comp
            .iter()
            .filter(|&(_, &r)| r == g)
            .map(|(&a, _)| {
                graph.reach[a as usize]
                    .iter()
                    .filter(|&&(b, _)| comp.get(&b) != Some(&g))
                    .map(|&(_, l)| l)
                    .sum::<f32>()
            })
            .sum();
        evidence *= (total / boundary.max(1.)).min(1.);
        if evidence > best {
            best = evidence;
            best_members = members.clone();
        }
    }
    (best, best_members)
}

/// Most leaves on both sides of a relation whose assignments are tried exhaustively.
pub(super) const EXHAUSTIVE: usize = 10;

/// A relation's shape as a cut: which leaves are under it. A leaf inside one pair and outside
/// another (bands of a gradient) goes to the side giving the best connected group the most
/// contact, exhaustively when there are few such leaves, greedily otherwise.
pub fn best_cut(supporting: &[Support], graph: &Graph) -> (f32, Vec<Support>) {
    let insides: HashSet<u32> = supporting.iter().map(|p| p.a).collect();
    let outsides: HashSet<u32> = supporting.iter().map(|p| p.b).collect();
    let fixed: HashSet<u32> = insides.difference(&outsides).copied().collect();
    let mut ambiguous: Vec<u32> = insides.intersection(&outsides).copied().collect();
    ambiguous.sort_unstable();
    if ambiguous.is_empty() {
        return cut_score(supporting, &fixed, graph);
    }
    if ambiguous.len() <= EXHAUSTIVE {
        let mut best = (0f32, Vec::new());
        for mask in 0u32..1 << ambiguous.len() {
            let mut under = fixed.clone();
            for (i, &a) in ambiguous.iter().enumerate() {
                if mask >> i & 1 == 1 {
                    under.insert(a);
                }
            }
            let r = cut_score(supporting, &under, graph);
            if r.0 > best.0 {
                best = r;
            }
        }
        return best;
    }
    let mut under = insides.clone();
    let mut best = cut_score(supporting, &under, graph);
    let mut improved = true;
    while improved {
        improved = false;
        for &a in &ambiguous {
            let mut trial = under.clone();
            if !trial.remove(&a) {
                trial.insert(a);
            }
            let r = cut_score(supporting, &trial, graph);
            if r.0 > best.0 {
                under = trial;
                best = r;
                improved = true;
            }
        }
    }
    best
}

/// In gamut up to the noise as it reaches `c`: `c = m / (1 - k)`, so an error of a level in
/// the colors is `1 / (1 - k)` levels in `c` (a black shape sits on the gamut's edge).
pub(super) fn gamut_k(c: Vec3, k: f32) -> bool {
    let margin = NOISE / (1. - k).max(0.05);
    c.min_element() >= -margin && c.max_element() <= 1. + margin
}

/// Least contact for a pair of leaves to give a relation, and least inside leaf, in pixels.
pub(super) const MIN_CONTACT: u32 = 8;
pub(super) const MIN_INSIDE: u32 = 64;

/// Relations between neighboring leaves' equations: for inside A over a graded B, `k` from the
/// slopes and `m` with B's plane at A's centroid; a flat B gives a line of (k, m), fixed by
/// two pairs crossing. Refitted on their best cuts, pairs claimed in score order.
pub fn equation_graph(lv: &Leaves) -> Vec<Hypothesis> {
    let graph = Graph::of(lv);
    let width = lv.width;
    let at = |i: u32| glam::Vec2::new((i as usize % width) as f32, (i as usize / width) as f32);
    // How far a leaf's plane carries a color across a leaf's pixels, over the channels
    let span_over = |plane: &Plane, leaf: usize| -> f32 {
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for &i in lv.pixels_of(leaf) {
            let v = plane.step(at(i));
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (hi - lo).max_element().max(0.)
    };
    let span: Vec<f32> = (0..lv.n)
        .map(|l| {
            if lv.plane[l].slope() > 0.2 * SLOPE_MIN {
                span_over(&lv.plane[l], l)
            } else {
                0.
            }
        })
        .collect();
    struct Raw {
        a: u32,
        b: u32,
        len: f32,
        bg: Vec3,
        img: Vec3,
        ga: Plane,
        gb: Plane,
        span: f32,
    }
    let mut raw: Vec<(Hypothesis, bool)> = Vec::new();
    let mut pairs: Vec<Raw> = Vec::new();
    for (a, &span_a) in span.iter().enumerate() {
        if lv.size[a] < MIN_INSIDE || lv.thin[a] {
            continue;
        }
        let pa = lv.plane[a];
        let ca = lv.centroid[a];
        for &pi in &lv.reach[a] {
            let p = &lv.pairs[pi as usize];
            let b = p.y as usize;
            if p.len < MIN_CONTACT {
                continue;
            }
            let pb = lv.plane[b];
            // A shape's boundary is a jump in color: planes that agree where they meet are
            // one surface cut at a kink (as the jellyfish's water)
            if (pa.at(p.mid) - pb.at(p.mid)).abs().max_element() <= 2. * NOISE {
                continue;
            }
            // B's plane extrapolated to A's centroid: the background beneath A
            let bg = pb.at(ca);
            let img = pa.at(ca);
            if (img - bg).abs().max_element() <= 2. * NOISE {
                continue;
            }
            // The inside's variation counts as show-through only as far as the background's
            // plane, a real gradient, varies under it too (anti-aliased dashes vary over a
            // flat sky)
            let span_b = if pb.slope() > SLOPE_MIN {
                span_over(&pb, a)
            } else {
                0.
            };
            pairs.push(Raw {
                a: a as u32,
                b: b as u32,
                len: p.len as f32,
                bg,
                img,
                ga: pa,
                gb: pb,
                span: span_a.min(span_b),
            });
            let (sa, sb) = (pa.slope(), pb.slope());
            if sb > SLOPE_MIN && sa > 0.2 * SLOPE_MIN {
                // Slopes: A's is k times B's, in the same direction
                let num = pa.gx.dot(pb.gx) + pa.gy.dot(pb.gy);
                let den = pb.gx.length_squared() + pb.gy.length_squared();
                let k = num / den;
                if !(0.05..=0.95).contains(&k) {
                    continue;
                }
                let off = ((pa.gx - k * pb.gx).length_squared()
                    + (pa.gy - k * pb.gy).length_squared())
                .sqrt();
                if off > 0.3 * sa + SLOPE_MIN {
                    continue;
                }
                let c = (img - k * bg) / (1. - k);
                if gamut_k(c, k) {
                    raw.push((
                        Hypothesis {
                            k,
                            c,
                            score: p.len as f32 * k * 2.,
                        },
                        true,
                    ));
                }
            }
        }
    }
    // Crossings: two pairs at one inside leaf, or at two inside leaves that touch
    let mut by_inside: HashMap<u32, Vec<usize>> = HashMap::default();
    let mut insides: Vec<u32> = Vec::new();
    for (i, p) in pairs.iter().enumerate() {
        by_inside
            .entry(p.a)
            .or_insert_with(|| {
                insides.push(p.a);
                Vec::new()
            })
            .push(i);
    }
    for &a in &insides {
        let first = &by_inside[&a];
        let others = std::iter::once(a).chain(
            graph.reach[a as usize]
                .iter()
                .map(|&(n, _)| n)
                .filter(|n| by_inside.contains_key(n)),
        );
        for a2 in others {
            let second = &by_inside[&a2];
            for &i in first {
                for &j in second {
                    let (p1, p2) = (&pairs[i], &pairs[j]);
                    if p1.b == p2.b || (a2 == a && p2.b <= p1.b) {
                        continue;
                    }
                    let (db, d_o) = (p1.bg - p2.bg, p1.img - p2.img);
                    let usable: Vec<usize> = (0..3).filter(|&c| db[c].abs() > 8. * NOISE).collect();
                    if usable.len() < 2 {
                        continue;
                    }
                    let ks: Vec<f32> = usable.iter().map(|&c| d_o[c] / db[c]).collect();
                    let k = ks.iter().sum::<f32>() / ks.len() as f32;
                    if !(0.05..=0.95).contains(&k)
                        || ks
                            .iter()
                            .zip(&usable)
                            .any(|(&kc, &c)| (kc - k).abs() * db[c].abs() > 2. * NOISE)
                    {
                        continue;
                    }
                    let m = (p1.img - k * p1.bg + p2.img - k * p2.bg) / 2.;
                    let c = m / (1. - k);
                    if gamut_k(c, k) {
                        raw.push((
                            Hypothesis {
                                k,
                                c,
                                score: p1.len.min(p2.len) * k,
                            },
                            false,
                        ));
                    }
                }
            }
        }
    }
    // Raw hypotheses merge only when nearly identical, and each relation is then refitted
    // over its own support: a loose merge would decide it by its first member
    let mut sorted: Vec<Hypothesis> = raw.iter().map(|r| r.0).collect();
    by_score(&mut sorted);
    let mut merged: Vec<Hypothesis> = Vec::new();
    for h in sorted {
        match merged
            .iter_mut()
            .find(|e| (e.k - h.k).abs() <= 0.015 && (e.c - h.c).abs().max_element() <= 4. / 255.)
        {
            Some(e) => e.score += h.score,
            None => merged.push(h),
        }
    }
    let tol = 3. * NOISE;
    let support = |k: f32, m: Vec3| -> Vec<Support> {
        let c = m / (1. - k);
        pairs
            .iter()
            .filter(|p| {
                (p.img - (k * p.bg + m)).abs().max_element() <= tol
                    // an inside of the relation's own color shows nothing through: an opaque
                    // shape of that color explains it as well, and it fits any k
                    && (p.img - c).abs().max_element() > tol
                    && slope_fits(&p.ga, &p.gb, k)
            })
            .map(|p| Support {
                a: p.a,
                b: p.b,
                len: p.len,
                bg: p.bg,
                img: p.img,
                span: p.span,
            })
            .collect()
    };
    // o = k b + m over the supporting pairs, one k for all channels, weighted by contact; the
    // given relation where the backgrounds do not vary
    let refit = |found: &[Support], k: f32, m: Vec3| -> (f32, Vec3) {
        let w: f32 = found.iter().map(|p| p.len).sum();
        let bm = found.iter().map(|p| p.len * p.bg).sum::<Vec3>() / w;
        let om = found.iter().map(|p| p.len * p.img).sum::<Vec3>() / w;
        let den: f32 = found
            .iter()
            .map(|p| p.len * (p.bg - bm).length_squared())
            .sum();
        if den < (4. * NOISE).powi(2) * w {
            return (k, m);
        }
        let k2 = found
            .iter()
            .map(|p| p.len * (p.bg - bm).dot(p.img - om))
            .sum::<f32>()
            / den;
        (k2, om - k2 * bm)
    };
    struct Scored {
        k: f32,
        c: Vec3,
        score: f32,
        supporting: Vec<Support>,
        members: Vec<Support>,
    }
    let mut out: Vec<Scored> = Vec::new();
    for h in &merged {
        let (mut k, mut m) = (h.k, (1. - h.k) * h.c);
        let mut supporting = support(k, m);
        let (mut best, mut members) = if supporting.is_empty() {
            (0., Vec::new())
        } else {
            best_cut(&supporting, &graph)
        };
        // The refit uses the pairs of the relation's best connected group, not every pair in
        // the image that fits, which would drift it; kept while the score holds
        for _ in 0..3 {
            if members.is_empty() {
                break;
            }
            let (k2, m2) = refit(&members, k, m);
            if !(0.05..=0.95).contains(&k2) || !gamut_k(m2 / (1. - k2), k2) {
                break;
            }
            let again = support(k2, m2);
            let (b2, m2s) = if again.is_empty() {
                (0., Vec::new())
            } else {
                best_cut(&again, &graph)
            };
            if b2 < best {
                break;
            }
            let key =
                |s: &[Support]| -> HashSet<(u32, u32)> { s.iter().map(|p| (p.a, p.b)).collect() };
            let same = key(&m2s) == key(&members);
            (k, m, supporting, best, members) = (k2, m2, again, b2, m2s);
            if same {
                break;
            }
        }
        if supporting.is_empty() || best <= 0. {
            continue;
        }
        // Times k, the share of the background that shows through: a near-opaque relation
        // fits any nearly flat region beside anything
        out.push(Scored {
            k,
            c: m / (1. - k),
            score: best * k,
            supporting,
            members,
        });
    }
    // Each edge carries one relation: taken in order of score, each relation claims its
    // shape's pairs and the rest are scored again without them
    let mut claimed: HashSet<(u32, u32)> = HashSet::default();
    let mut final_: Vec<Hypothesis> = Vec::new();
    let mut pending = out;
    while !pending.is_empty() {
        pending.sort_by(|a, b| b.score.total_cmp(&a.score));
        let h = pending.remove(0);
        if h.score <= 0. {
            break;
        }
        final_.push(Hypothesis {
            k: h.k,
            c: h.c,
            score: h.score,
        });
        let mine: HashSet<(u32, u32)> = h.members.iter().map(|p| (p.a, p.b)).collect();
        claimed.extend(mine.iter().copied());
        for o in &mut pending {
            if o.members.iter().any(|p| mine.contains(&(p.a, p.b))) {
                let left: Vec<Support> = o
                    .supporting
                    .iter()
                    .filter(|p| !claimed.contains(&(p.a, p.b)))
                    .copied()
                    .collect();
                let (b2, m2) = if left.is_empty() {
                    (0., Vec::new())
                } else {
                    best_cut(&left, &graph)
                };
                o.score = b2 * o.k;
                o.supporting = left;
                o.members = m2;
            }
        }
    }
    // Refitting brings raw relations that differed onto one: the best of each
    let kept = merge_near(&final_, 0.02, 6. / 255.);
    fold_seen_through(kept, 4. * NOISE)
}

/// Where a lower shape L runs under an upper one U, the overlap is L's relation seen through
/// U (L's `k`, U over L's color): evidence for L, not a shape of its own, so its score goes
/// to L.
pub(super) fn fold_seen_through(mut hyps: Vec<Hypothesis>, tol: f32) -> Vec<Hypothesis> {
    let mut gone = vec![false; hyps.len()];
    for i in 0..hyps.len() {
        let (k, c, s) = (hyps[i].k, hyps[i].c, hyps[i].score);
        for j in 0..hyps.len() {
            if j == i || gone[j] || (hyps[j].k - k).abs() > 0.03 {
                continue;
            }
            let cl = hyps[j].c;
            // The view through U differs from L's color and from U's: over a near-white U any
            // light color looks near-white
            if (c - cl).abs().max_element() <= tol {
                continue;
            }
            let through = (0..hyps.len()).any(|u| {
                u != i
                    && u != j
                    && !gone[u]
                    && (c - hyps[u].c).abs().max_element() > tol
                    && (hyps[u].k * cl + (1. - hyps[u].k) * hyps[u].c - c)
                        .abs()
                        .max_element()
                        <= tol
            });
            if through {
                hyps[j].score += s;
                gone[i] = true;
                break;
            }
        }
    }
    let mut out: Vec<Hypothesis> = hyps
        .into_iter()
        .zip(gone)
        .filter(|(_, g)| !g)
        .map(|(h, _)| h)
        .collect();
    by_score(&mut out);
    out
}

/// Hypotheses per list taken into the merged list.
pub(super) const PER_LIST: usize = 30;

/// Hypothesis lists merged: each list's scores scaled to its best, hypotheses within 0.05 in
/// `k` and 12 levels in `c` merged by their scaled scores summed.
pub fn merge_lists(lists: &[Vec<Hypothesis>]) -> Vec<Hypothesis> {
    let mut merged: Vec<Hypothesis> = Vec::new();
    for hyps in lists {
        let Some(first) = hyps.first() else {
            continue;
        };
        let best = if first.score != 0. { first.score } else { 1. };
        for h in hyps.iter().take(PER_LIST) {
            let s = h.score / best;
            match merged.iter_mut().find(|e| {
                (e.k - h.k).abs() <= 0.05 && (e.c - h.c).abs().max_element() <= 12. / 255.
            }) {
                Some(e) => e.score += s,
                None => merged.push(Hypothesis { score: s, ..*h }),
            }
        }
    }
    by_score(&mut merged);
    merged
}
