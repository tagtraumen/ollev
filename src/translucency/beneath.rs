//! The image beneath a layer: the raster peeled by the layer and its inner layers
//! ([`super::inner`]), and how far a color lies outside what the layer can produce, which
//! marks the pixels drawn over the layer (strands, outlines) whose peel is no beneath.

#[cfg(feature = "diagnostics")]
use super::inner::{Class, InnerFit};
#[cfg(feature = "diagnostics")]
use super::raster::Image;
use super::raster::DVec3;

/// How far a color lies outside the colors the layer `o = k b + m` can produce (b in the unit
/// cube), the largest excess over the channels; zero inside.
pub fn outside_image(c: DVec3, k: f64, m: DVec3) -> f64 {
    let lo = m;
    let hi = m + DVec3::splat(k);
    (lo - c).max(c - hi).max(DVec3::ZERO).max_element()
}

/// For the lab: the raster peeled inside the region by the inner layers (the white by its
/// opacity, a tint by its relation), then by the layer's relation; not clamped, as a peel
/// outside the unit cube is a color the layer cannot have been over.
#[cfg(feature = "diagnostics")]
pub fn peel(img: &Image, region: &[bool], fit: &InnerFit) -> Vec<DVec3> {
    (0..img.len())
        .map(|i| {
            let mut o = img.px[i];
            if !region[i] {
                return o;
            }
            let t = fit.white.get(i).copied().unwrap_or(0.0);
            if t > 0.0 && t < 1.0 {
                o = (o - DVec3::splat(t)) / (1.0 - t);
            }
            if let Class::Tint(ti) = fit.px[i].class {
                let tint = fit.tints[ti];
                o = (o - tint.m) / tint.k;
            }
            (o - fit.m) / fit.k
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layer's colors span the box from m to m + k; a color past either face is outside.
    #[test]
    fn image_box() {
        let (k, m) = (0.6, DVec3::splat(0.2));
        assert_eq!(outside_image(DVec3::splat(0.5), k, m), 0.0);
        assert!((outside_image(DVec3::new(0.9, 0.5, 0.5), k, m) - 0.1).abs() < 1e-12);
        assert!((outside_image(DVec3::new(0.5, 0.0, 0.5), k, m) - 0.2).abs() < 1e-12);
    }
}
