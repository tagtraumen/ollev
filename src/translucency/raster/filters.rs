//! Filters as scipy computes them: correlation, Gaussian, Sobel, interpolation.

use super::*;

// ----------------------------------------------------------------------------- filters

/// scipy's reflect extension ("d c b a | a b c d") of an index into a line of `n`.
pub(super) fn reflect(mut i: isize, n: isize) -> usize {
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n {
            i = 2 * n - i - 1;
        } else {
            return i as usize;
        }
    }
}

/// scipy's `correlate1d` in reflect mode along `axis` (0 the rows' direction, y; 1 x) of
/// a `width` x `height` plane, with the kernel's symmetry folded as `NI_Correlate1D`.
pub fn correlate1d(
    src: &[f64],
    width: usize,
    height: usize,
    axis: usize,
    weights: &[f64],
) -> Vec<f64> {
    let size = weights.len();
    let size1 = size / 2;
    let mut symmetric = 0;
    if size % 2 == 1 {
        symmetric = 1;
        for i in 1..=size / 2 {
            if (weights[i + size1] - weights[size1 - i]).abs() > f64::EPSILON {
                symmetric = 0;
                break;
            }
        }
        if symmetric == 0 {
            symmetric = -1;
            for i in 1..=size / 2 {
                if (weights[size1 + i] + weights[size1 - i]).abs() > f64::EPSILON {
                    symmetric = 0;
                    break;
                }
            }
        }
    }
    let size2 = (size - size1 - 1) as isize;
    // One output line from `at(j)`, the input line shifted by j: tap by tap over the whole
    // line, each pixel summed in the order `NI_Correlate1D` sums it
    fn line<'a>(
        o: &mut [f64],
        at: &dyn Fn(isize) -> &'a [f64],
        weights: &[f64],
        symmetric: i32,
        size2: isize,
    ) {
        let size1 = weights.len() / 2;
        let fw = |j: isize| weights[(size1 as isize + j) as usize];
        let first = if symmetric != 0 { 0 } else { size2 };
        for (o, &v) in o.iter_mut().zip(at(first)) {
            *o = v * fw(first);
        }
        if symmetric != 0 {
            for j in -(size1 as isize)..0 {
                let (a, b, f) = (at(j), at(-j), fw(j));
                let pairs = o.iter_mut().zip(a).zip(b);
                if symmetric > 0 {
                    pairs.for_each(|((o, &a), &b)| *o += (a + b) * f);
                } else {
                    pairs.for_each(|((o, &a), &b)| *o += (a - b) * f);
                }
            }
        } else {
            for j in -(size1 as isize)..size2 {
                let (a, f) = (at(j), fw(j));
                for (o, &a) in o.iter_mut().zip(a) {
                    *o += a * f;
                }
            }
        }
    }
    let mut out = vec![0.0; src.len()];
    if axis == 0 {
        // whole rows at once
        let row = |y: isize| {
            let y = reflect(y, height as isize);
            &src[y * width..(y + 1) * width]
        };
        for (ll, o) in out.chunks_exact_mut(width.max(1)).enumerate() {
            line(o, &|j| row(ll as isize + j), weights, symmetric, size2);
        }
        return out;
    }
    // each row extended by size1 on either side, read shifted
    let mut buf = vec![0.0; width + size];
    for (y, o) in out.chunks_exact_mut(width.max(1)).enumerate() {
        let src = &src[y * width..(y + 1) * width];
        for (bi, b) in buf.iter_mut().enumerate() {
            *b = src[reflect(bi as isize - size1 as isize, width as isize)];
        }
        let buf = &buf;
        let at = |j: isize| {
            let at = (size1 as isize + j) as usize;
            &buf[at..at + width]
        };
        line(o, &at, weights, symmetric, size2);
    }
    out
}

/// scipy's `_gaussian_kernel1d(sigma, order, radius)` reversed, as `gaussian_filter1d`
/// correlates with it; orders 0 and 1.
pub fn gaussian_weights(sigma: f64, order: usize) -> Vec<f64> {
    let radius = (4.0 * sigma + 0.5) as isize;
    let sigma2 = sigma * sigma;
    let phi: Vec<f64> = (-radius..=radius)
        .map(|x| ((-0.5 / sigma2) * ((x * x) as f64)).exp())
        .collect();
    let total = np_sum(&phi);
    let phi: Vec<f64> = phi.iter().map(|p| p / total).collect();
    let kernel: Vec<f64> = match order {
        0 => phi,
        1 => (-radius..=radius)
            .zip(&phi)
            .map(|(x, p)| (0.0 + x as f64 * (-1.0 / sigma2)) * p)
            .collect(),
        _ => unimplemented!("Gaussian derivatives above the first"),
    };
    kernel.into_iter().rev().collect()
}

/// scipy's `gaussian_filter(plane, sigma, order=(order_y, order_x))`: along y, then x.
pub fn gaussian_filter(
    src: &[f64],
    width: usize,
    height: usize,
    sigma: f64,
    order_y: usize,
    order_x: usize,
) -> Vec<f64> {
    let a = correlate1d(src, width, height, 0, &gaussian_weights(sigma, order_y));
    correlate1d(&a, width, height, 1, &gaussian_weights(sigma, order_x))
}

/// scipy's `sobel(plane, axis)`.
pub fn sobel(src: &[f64], width: usize, height: usize, axis: usize) -> Vec<f64> {
    let a = correlate1d(src, width, height, axis, &[-1.0, 0.0, 1.0]);
    correlate1d(&a, width, height, 1 - axis, &[1.0, 2.0, 1.0])
}

/// The gradient magnitude: per channel the Sobel gradient's length, the largest
/// over the channels, over 8 (a unit step gives 8).
pub fn gradient_magnitude(img: &Image) -> Vec<f64> {
    let (w, h) = (img.width, img.height);
    let mut g = vec![0.0f64; img.len()];
    for c in 0..3 {
        let ch = img.channel(c);
        let gx = sobel(&ch, w, h, 1);
        let gy = sobel(&ch, w, h, 0);
        for i in 0..g.len() {
            g[i] = g[i].max(gx[i].hypot(gy[i]));
        }
    }
    g.iter().map(|v| v / 8.0).collect()
}

/// Di Zenzo's color gradient at `sigma`: the magnitude and the direction across the edge
/// in radians.
pub fn di_zenzo(img: &Image, sigma: f64) -> (Vec<f64>, Vec<f64>) {
    let key = (sigma.to_bits(), img.width, img.height, digest(&img.px));
    (*remembered(&ZENZO, key, || di_zenzo_of(img, sigma))).clone()
}

/// The value computed for `key` if `known` still holds it, else `compute`'s, remembered: the
/// one found moves last, so the least recently asked is forgotten first past [`KEPT`].
pub fn remembered<K: PartialEq, V>(
    known: &'static std::thread::LocalKey<Remembered<K, V>>,
    key: K,
    compute: impl FnOnce() -> V,
) -> std::rc::Rc<V> {
    let found = known.with_borrow_mut(|known| {
        let at = known.iter().position(|z| z.0 == key)?;
        let z = known.remove(at);
        let out = z.1.clone();
        known.push(z);
        Some(out)
    });
    if let Some(found) = found {
        return found;
    }
    let out = std::rc::Rc::new(compute());
    known.with_borrow_mut(|known| {
        if known.len() >= KEPT {
            known.remove(0);
        }
        known.push((key, out.clone()));
    });
    out
}

/// What [`remembered`] keeps: per key its value, the least recently asked first.
pub type Remembered<K, V> = std::cell::RefCell<Vec<(K, std::rc::Rc<V>)>>;

/// How many images each remembered computation keeps its result of.
const KEPT: usize = 3;

/// Two 64-bit digests of the pixels' bits, by different multipliers: images whose digests
/// agree are taken as one (a chance agreement is about one in 2^128).
pub fn digest(px: &[DVec3]) -> (u64, u64) {
    let (mut a, mut b) = (0x243f_6a88_85a3_08d3u64, 0x1319_8a2e_0370_7344u64);
    for v in px.iter().flat_map(|p| p.to_array()) {
        let x = v.to_bits();
        a = (a ^ x).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29);
        b = (b ^ x.rotate_left(17))
            .wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
            .rotate_left(31);
    }
    (a, b)
}

/// An image's gradient by [`di_zenzo`] by σ's bits, its size and digest.
type Zenzo = (
    (u64, usize, usize, (u64, u64)),
    std::rc::Rc<(Vec<f64>, Vec<f64>)>,
);

thread_local! {
    /// The gradients computed last: the uncertainty, the edges, the linework and the edge map
    /// read the same picture's at the same scales.
    static ZENZO: std::cell::RefCell<Vec<Zenzo>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn di_zenzo_of(img: &Image, sigma: f64) -> (Vec<f64>, Vec<f64>) {
    let (w, h) = (img.width, img.height);
    let n = img.len();
    let (mut j0, mut j1, mut j2) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
    for c in 0..3 {
        let ch = img.channel(c);
        let gx = gaussian_filter(&ch, w, h, sigma, 0, 1);
        let gy = gaussian_filter(&ch, w, h, sigma, 1, 0);
        for i in 0..n {
            j0[i] += gx[i] * gx[i];
            j1[i] += gy[i] * gy[i];
            j2[i] += gx[i] * gy[i];
        }
    }
    let mut mag = vec![0.0; n];
    let mut theta = vec![0.0; n];
    for i in 0..n {
        let (tr, df) = (j0[i] + j1[i], j0[i] - j1[i]);
        let lam = 0.5 * (tr + (df * df + 4.0 * (j2[i] * j2[i])).sqrt());
        mag[i] = lam.max(0.0).sqrt();
        theta[i] = 0.5 * (2.0 * j2[i]).atan2(df);
    }
    (mag, theta)
}

/// The two indices and weights of order-1 interpolation along one axis, mode nearest.
pub(super) fn axis_weights(cc: f64, len: usize) -> ([usize; 2], [f64; 2]) {
    let fl = cc.floor();
    let start = fl as isize;
    let t = cc - fl;
    let w0 = 1.0 - t;
    let w1 = 1.0 - w0;
    let clamp = |i: isize| i.clamp(0, len as isize - 1) as usize;
    ([clamp(start), clamp(start + 1)], [w0, w1])
}

/// scipy's `map_coordinates(plane, [[y], [x]], order=1, mode="nearest")`.
pub fn map1(plane: &[f64], width: usize, height: usize, x: f64, y: f64) -> f64 {
    let (iy, wy) = axis_weights(y, height);
    let (ix, wx) = axis_weights(x, width);
    let mut t = 0.0;
    for a in 0..2 {
        for b in 0..2 {
            t += plane[iy[a] * width + ix[b]] * wy[a] * wx[b];
        }
    }
    t
}

// ----------------------------------------------------------------------------- morphology
