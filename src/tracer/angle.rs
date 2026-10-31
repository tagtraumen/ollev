// Panic ratchet: the lints this module has yet to meet, each removed as it is cleared
// clean of these, which the parent module still expects
#![cfg_attr(not(test), deny(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::disallowed_macros,
    clippy::indexing_slicing,
))]

use glam::Vec2;

/// The signed angle from `a` to `b` (glam's `angle_to`), or None where either has no length and
/// so no direction; glam's own call asserts there under `glam-assert` (Graphite's debug builds).
#[must_use]
pub fn turn(a: Vec2, b: Vec2) -> Option<f32> {
    (a.length_squared() > 0. && b.length_squared() > 0.).then(|| a.angle_to(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_vector_has_no_turn() {
        assert_eq!(turn(Vec2::ZERO, Vec2::X), None);
        assert_eq!(turn(Vec2::Y, Vec2::ZERO), None);
        assert_eq!(turn(Vec2::NAN, Vec2::X), None);
        assert_eq!(turn(Vec2::X, Vec2::Y), Some(Vec2::X.angle_to(Vec2::Y)));
    }
}
