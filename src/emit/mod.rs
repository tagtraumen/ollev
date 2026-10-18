//! Emitters: the tracer's scene model ([`crate::model::Model`]) compiled to a target's
//! bytes, each output checked against the model's own render (see [`verify`]).

use crate::Scene;
use crate::model::{Model, Object};
use glam::Vec3;

mod bands;
pub mod dump;
pub mod svg;
pub mod verify;

pub use dump::ModelDump;
pub use svg::Svg;
pub use verify::{Output, Report, compile};

/// A target: the model's bytes with object `k` of [`Model::objects`] encoded at level
/// `levels[k]` (finer as it grows), and the bytes rendered where this target can be.
pub trait Emitter {
    fn encode(&self, model: &Model, levels: &[u8]) -> Vec<u8>;
    /// The finest level an object may step to.
    fn finest(&self, object: &Object) -> u8;
    fn render(&self, bytes: &[u8], width: usize, height: usize) -> Option<Vec<Vec3>>;
}

/// The scene as SVG at the first encoding, unverified (for snapshots and dumps).
pub fn svg_of(scene: &Scene) -> String {
    svg::encode(&scene.model(), &[])
}

#[cfg(test)]
mod tests;
