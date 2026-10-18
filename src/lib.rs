//! Raster images traced into layered vector scenes: each region a shape drawn over the shape it
//! lies in, strokes and dashed lines where the picture draws lines, gradients, blurs, repeated
//! objects and translucent layers where it shows them. One objective decides everything: the
//! squared colour error plus a price per cubic segment, set by a detail slider.
//!
//! [`trace`] takes an RGBA [`Image`] and [`TraceOptions`] and returns the scene as a
//! [`model::Model`], lowered to what the host draws ([`Capabilities`]); [`trace_with`] also
//! reports [`Progress`] and can be cancelled.
//!
//! Features: `svg` adds the module `emit`, the SVG writer with its verify loop (rendering with
//! resvg); `diagnostics` adds the developers' logs and dumps (`Diagnostics`, in
//! [`TraceOptions`]), and implies `svg`.
//!
//! ```
//! let pixels = [255u8, 0, 0, 255].repeat(16 * 16);
//! let image = ollev::Image {
//!     pixels: ollev::Pixels::Rgba8(&pixels),
//!     width: 16,
//!     height: 16,
//! };
//! let model = ollev::trace(&image, &ollev::TraceOptions::default()).unwrap();
//! assert!(!model.objects().is_empty());
//! ```

// The lab's modules link their private parts in their docs
#![cfg_attr(feature = "diagnostics", allow(rustdoc::private_intra_doc_links))]

/// The tracer's own modules: private, and with the `diagnostics` feature public but left out
/// of the documentation, for the bench's lab tools.
macro_rules! internal {
    ($($m:ident),* $(,)?) => {$(
        #[cfg(feature = "diagnostics")]
        #[doc(hidden)]
        pub mod $m;
        #[cfg(not(feature = "diagnostics"))]
        mod $m;
    )*};
}

internal!(
    blur,
    cleanup,
    complete,
    context,
    dashes,
    diagnostics,
    economy,
    edges,
    fill,
    glow,
    hierarchy,
    input,
    instancing,
    linework,
    lower,
    outline,
    plan,
    polish,
    progress,
    recolor,
    refine,
    render,
    scene,
    soft,
    stroke,
    timing,
    tracer,
    translucency,
    uncertainty,
);

#[cfg(feature = "svg")]
pub mod emit;
pub mod model;

pub(crate) use context::{Context, Weights, Window};
#[cfg(feature = "diagnostics")]
pub use diagnostics::Diagnostics;
/// The vector types of the model, for its callers.
pub use glam;
pub use input::{Detail, Image, Pixels, Stages, TraceError, TraceOptions, trace, trace_with};
pub use lower::Capabilities;
pub use progress::{Answer, Progress};
pub use scene::Scene;
pub(crate) use scene::{Fill, Raster, Shape};

/// A hash map with a fixed seed, so the traced output is the same on every run.
pub(crate) type HashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<std::hash::DefaultHasher>>;
/// A hash set with a fixed seed, for the same reason.
pub(crate) type HashSet<K> =
    std::collections::HashSet<K, std::hash::BuildHasherDefault<std::hash::DefaultHasher>>;
