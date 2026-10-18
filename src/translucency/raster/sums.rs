//! Sums, means, medians and ranges in numpy's order of operations.

use super::*;

// ----------------------------------------------------------------------------- numpy sums

/// numpy's pairwise sum of a contiguous run (`pairwise_sum_DOUBLE`).
pub fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// numpy's sum of a 1-d array (`np.sum`, `.sum()`): the identity plus the pairwise sum.
pub fn np_sum(a: &[f64]) -> f64 {
    0.0 + pairwise(a)
}

/// numpy's `np.add.reduceat` over one segment: its first element plus the pairwise sum of
/// the rest.
pub fn reduceat_sum(a: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    a[0] + pairwise(&a[1..])
}

/// numpy's sum along the rows of a 2-d array (`X.sum(0)`): row after row.
pub fn rows_sum(rows: impl IntoIterator<Item = DVec3>) -> DVec3 {
    let mut s = DVec3::ZERO;
    for r in rows {
        s += r;
    }
    s
}

/// numpy's `X.mean(0)` of rows: their sum in order over their count.
pub fn rows_mean(rows: &[DVec3]) -> DVec3 {
    rows_sum(rows.iter().copied()) / rows.len() as f64
}

/// The sum of the elements of a (rows, 3) array (`(A * B).sum()`): pairwise over the
/// array's elements in row-major order.
pub fn sum_all3(rows: &[DVec3]) -> f64 {
    let flat: Vec<f64> = rows.iter().flat_map(|r| [r.x, r.y, r.z]).collect();
    np_sum(&flat)
}

/// numpy's median: the middle value, or the mean of the two middle ones; infinite for no
/// values. Sorts.
pub fn median(v: &mut [f64]) -> f64 {
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return f64::INFINITY;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// numpy's `np.percentile(v, q)` with linear interpolation: the virtual index `(n - 1) q /
/// 100` between its two ranks, numpy's `_lerp` (from the upper value when the fraction is
/// half or more). Sorts; 0 for no values.
pub fn percentile(v: &mut [f64], q: f64) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    v.sort_unstable_by(f64::total_cmp);
    let at = (n - 1) as f64 * (q / 100.0);
    if at >= (n - 1) as f64 {
        return v[n - 1];
    }
    let lo = at.floor();
    let gamma = at - lo;
    let (a, b) = (v[lo as usize], v[lo as usize + 1]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// numpy's `np.median(rows, axis=0)` per channel.
pub fn median_rows(rows: &[DVec3]) -> DVec3 {
    let mut out = DVec3::ZERO;
    let mut v = Vec::with_capacity(rows.len());
    for c in 0..3 {
        v.clear();
        v.extend(rows.iter().map(|r| r[c]));
        out[c] = median(&mut v);
    }
    out
}

/// numpy's `np.arange(start, stop, step)` for floats: the length by the ceiling of the
/// span over the step, the values `start + i * ((start + step) - start)` past the second.
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil();
    if n <= 0.0 {
        return Vec::new();
    }
    let n = n as usize;
    let second = start + step;
    let delta = second - start;
    (0..n)
        .map(|i| match i {
            0 => start,
            1 => second,
            _ => start + i as f64 * delta,
        })
        .collect()
}

/// numpy's `np.linspace(start, stop, num)`: `i * step + start`, the last value pinned.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut out: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    out[num - 1] = stop;
    out
}

/// numpy's `np.radians`.
pub fn radians(deg: f64) -> f64 {
    deg * (std::f64::consts::PI / 180.0)
}
