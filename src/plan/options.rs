//! What a level of the plan traces with.

/// What varies between the picture's level and the levels beneath its peeled layers.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// The families are proposed, gated and decided whole after the opaque trace of the first
    /// level (see [`super::accepted_families`], [`super::decide_whole`]), not beneath.
    pub acceptance: bool,
    /// The detail slider: whether a shape, a stroke or a mark is drawn is decided by the
    /// residual beyond this multiple of the picture's uncertainty that it removes (see
    /// [`crate::uncertainty`]).
    pub tolerance: f64,
    /// The stages run (see [`crate::Stages`]).
    pub stages: crate::Stages,
}

impl Options {
    /// A trace at the detail slider's `tolerance`.
    pub fn new(tolerance: f64) -> Self {
        Self {
            acceptance: true,
            tolerance,
            stages: crate::Stages::ALL,
        }
    }
}

impl Default for Options {
    fn default() -> Self {
        Self::new(TOLERANCE)
    }
}

/// The detail slider's default: the residual beyond the picture's uncertainty itself.
pub const TOLERANCE: f64 = 1.0;

/// The edge term the plan runs with: the error at the target's edges weighted at lambda 1, so
/// lines the eye reads the drawing by (hems, laces, contours) pay.
pub const EDGE_TERM: crate::edges::EdgeTerm = crate::edges::EdgeTerm { lambda: 1.0 };

/// Least materials whose images meet in a family's region for it to be tried at all. Fitted
/// families are not exempt, since a shading of one material passes on one cluster; a second
/// material is what tells a layer.
pub(super) const GATE: usize = 2;
