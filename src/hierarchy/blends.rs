//! Thin blends between two regions dissolved into them before the tree is built.

use super::*;

/// Most pixels per pixel of boundary a blend region has: about one pixel thick.
const BLEND_THICKNESS: f64 = 1.;
/// Farthest a blend region's mean color lies from the line between its two neighbors' colors.
const BLEND_DISTANCE: f64 = 0.06;
/// Least color distance between two neighbors for a region between them to be their blend.
const BLEND_CONTRAST: f64 = 0.1;

/// Dissolves the thin regions of a segmentation that are blends of two neighbors, as
/// anti-aliasing leaves along edges, into those neighbors: each pixel goes to the one nearer its
/// color. Returns the new number of regions; `labels` are renumbered.
pub fn dissolve_blends(
    labels: &mut [u32],
    count: usize,
    colors: &[DVec3],
    width: usize,
    height: usize,
) -> usize {
    let mut stats = vec![Stats::default(); count];
    let mut boundary = vec![0u32; count];
    let mut shared: Vec<HashMap<u32, u32>> = vec![HashMap::default(); count];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let l = labels[i];
            stats[l as usize] = stats[l as usize].add(&Stats::pixel(colors[i]));
            boundary[l as usize] += u32::from(x == 0)
                + u32::from(y == 0)
                + u32::from(x + 1 == width)
                + u32::from(y + 1 == height);
            let mut link = |j: usize| {
                let m = labels[j];
                if m != l {
                    boundary[l as usize] += 1;
                    boundary[m as usize] += 1;
                    *shared[l as usize].entry(m).or_default() += 1;
                    *shared[m as usize].entry(l).or_default() += 1;
                }
            };
            if x + 1 < width {
                link(i + 1);
            }
            if y + 1 < height {
                link(i + width);
            }
        }
    }

    // The two neighbors each thin region is a blend of, if any
    let between: Vec<Option<(u32, u32)>> = (0..count)
        .map(|l| {
            if stats[l].n > BLEND_THICKNESS * boundary[l] as f64 {
                return None;
            }
            let m = stats[l].mean();
            let neighbors: Vec<u32> = shared[l].keys().copied().collect();
            let mut best: Option<(f64, (u32, u32))> = None;
            for (k, &a) in neighbors.iter().enumerate() {
                for &b in &neighbors[k + 1..] {
                    let (ca, cb) = (stats[a as usize].mean(), stats[b as usize].mean());
                    let axis = cb - ca;
                    if axis.length() < BLEND_CONTRAST {
                        continue;
                    }
                    let t = (m - ca).dot(axis) / axis.length_squared();
                    let distance = (m - (ca + t * axis)).length();
                    if (0. ..=1.).contains(&t)
                        && distance < BLEND_DISTANCE
                        && best.is_none_or(|(d, _)| distance < d)
                    {
                        best = Some((distance, (a, b)));
                    }
                }
            }
            best.map(|(_, pair)| pair)
        })
        .collect();
    // Only into regions that stay
    let dissolved = |l: u32| {
        between[l as usize]
            .is_some_and(|(a, b)| between[a as usize].is_none() && between[b as usize].is_none())
    };
    for (i, label) in labels.iter_mut().enumerate() {
        if dissolved(*label) {
            let (a, b) = between[*label as usize].expect("dissolved regions have neighbors");
            let (ca, cb) = (stats[a as usize].mean(), stats[b as usize].mean());
            *label = if colors[i].distance_squared(ca) <= colors[i].distance_squared(cb) {
                a
            } else {
                b
            };
        }
    }

    // Number the remaining regions consecutively
    let mut index = vec![u32::MAX; count];
    let mut next = 0;
    for label in labels.iter_mut() {
        if index[*label as usize] == u32::MAX {
            index[*label as usize] = next;
            next += 1;
        }
        *label = index[*label as usize];
    }
    next as usize
}

/// The width in pixels of the color transition at the edge between pixels `i` and `j` (`step`
/// apart): the difference two pixels further in each side over the jump at the edge, 1 to 8.
/// Edges with too little difference to measure count as clean.
pub(super) fn transition_width(
    labels: &[u32],
    colors: &[DVec3],
    i: usize,
    j: usize,
    step: usize,
) -> f64 {
    let same = |k: usize, l: u32| k < labels.len() && labels[k] == l;
    let (a, b) = (labels[i], labels[j]);
    let inside = (i >= 2 * step && same(i - 2 * step, a))
        .then(|| colors[i - 2 * step])
        .or_else(|| (i >= step && same(i - step, a)).then(|| colors[i - step]))
        .unwrap_or(colors[i]);
    let outside = same(j + 2 * step, b)
        .then(|| colors[j + 2 * step])
        .or_else(|| same(j + step, b).then(|| colors[j + step]))
        .unwrap_or(colors[j]);
    let local = inside.distance(outside);
    if local < 0.02 {
        return 1.;
    }
    (local / colors[i].distance(colors[j]).max(1e-6)).clamp(1., 8.)
}

#[cfg(test)]
mod linear_fill_tests {
    use super::*;

    fn stats_of(pixels: &[(DVec2, DVec3)]) -> Stats {
        pixels.iter().fold(Stats::default(), |s, (p, c)| {
            s.add(&Stats::pixel_at(*c, *p))
        })
    }

    #[test]
    fn moments_match_the_sums() {
        // A diagonal ramp with a little texture
        let pixels: Vec<(DVec2, DVec3)> = (0..20)
            .flat_map(|y| (0..30).map(move |x| DVec2::new(x as f64 + 0.5, y as f64 + 0.5)))
            .map(|p| {
                let wobble = ((p.x * 1.7).sin() * (p.y * 0.9).cos()) * 0.02;
                let c = DVec3::new(0.2, 0.5, 0.7)
                    + DVec3::new(0.01, 0.005, -0.004) * (p.x + 2. * p.y)
                    + DVec3::splat(wobble);
                (p, c)
            })
            .collect();
        let stats = stats_of(&pixels);
        let fill = stats.linear_fill().expect("a ramp");
        let direct: f64 = pixels
            .iter()
            .map(|(p, c)| (*c - fill.at(*p)).length_squared())
            .sum();
        let by_moments = stats.sse_about_linear(&fill);
        assert!(
            (direct - by_moments).abs() < 1e-6 * direct.max(1.),
            "{direct} vs {by_moments}"
        );
        assert!(
            by_moments < 0.3 * stats.sse(),
            "{by_moments} of {}",
            stats.sse()
        );
        // The axis is the ramp's direction, up to sign
        let expected = DVec2::new(1., 2.).normalize();
        assert!(
            fill.axis.dot(expected).abs() > 0.999,
            "axis {:?}",
            fill.axis
        );
    }

    #[test]
    fn radial_moments_match_the_sums() {
        // A disc shaded from a point off its center, far from the origin
        let light = DVec2::new(331.3, 207.8);
        let pixels: Vec<(DVec2, DVec3)> = (180..240)
            .flat_map(|y| (300..380).map(move |x| DVec2::new(x as f64 + 0.5, y as f64 + 0.5)))
            .map(|p| {
                let d = p.distance(light);
                (
                    p,
                    DVec3::new(0.9, 0.8, 0.2) - DVec3::new(0.01, 0.006, -0.003) * d,
                )
            })
            .collect();
        let stats = stats_of(&pixels);
        let fill = stats.radial_fill().expect("a radial fill");
        let direct: f64 = pixels
            .iter()
            .map(|(p, c)| (*c - fill.at(*p)).length_squared())
            .sum();
        let by_moments = stats.sse_about_radial(&fill);
        assert!(
            (direct - by_moments).abs() < 1e-6 * direct.max(1.),
            "{direct} vs {by_moments}"
        );
        assert!(fill.center.distance(light) < 3., "center {:?}", fill.center);
        let linear = stats.linear_fill().expect("a ramp");
        assert!(by_moments < 0.5 * stats.sse_about_linear(&linear));
    }

    #[test]
    fn flat_pixels_have_no_fill() {
        let pixels: Vec<(DVec2, DVec3)> = (0..10)
            .flat_map(|y| (0..10).map(move |x| DVec2::new(x as f64, y as f64)))
            .map(|p| (p, DVec3::splat(0.5)))
            .collect();
        assert!(stats_of(&pixels).linear_fill().is_none());
    }
}
