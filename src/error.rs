//! Why a trace fails ([`TraceError`]), and the tracer's checked forms of what would otherwise
//! panic: an invariant that may not hold ([`Invariant`]) and an allocation that may not fit
//! ([`with_capacity`], [`filled`], [`area`]).

use std::collections::TryReserveError;
use std::panic::Location;

/// Why an image is not traced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TraceError {
    /// A side of zero pixels.
    Empty,
    /// The pixels do not hold four channels for each pixel of the size.
    Size {
        /// The channel values the size asks for.
        expected: usize,
        /// The channel values given.
        actual: usize,
    },
    /// The size's pixels do not fit in the address space (four channels each).
    TooLarge {
        /// Pixels per row.
        width: usize,
        /// Rows.
        height: usize,
    },
    /// An allocation the trace needed was refused.
    OutOfMemory,
    /// The progress callback asked to stop ([`crate::trace_with`]).
    Cancelled,
    /// The trace was stepped again after it was done or had failed ([`crate::Tracing::step`]).
    Finished,
    /// An invariant of the tracer did not hold: a bug in ollev, worth reporting with the image.
    Internal(Internal),
}

/// Where the tracer found one of its invariants broken (see [`TraceError::Internal`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Internal {
    what: &'static str,
    location: &'static Location<'static>,
}

impl Internal {
    /// What was expected to hold.
    #[must_use]
    pub fn what(&self) -> &'static str {
        self.what
    }

    /// Where in ollev's source it did not.
    #[must_use]
    pub fn location(&self) -> &'static Location<'static> {
        self.location
    }
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraceError::Empty => write!(f, "the image has no pixels"),
            TraceError::Size { expected, actual } => {
                write!(f, "expected {expected} channel values, got {actual}")
            }
            TraceError::TooLarge { width, height } => {
                write!(f, "an image of {width} by {height} pixels is too large")
            }
            TraceError::OutOfMemory => write!(f, "the trace ran out of memory"),
            TraceError::Cancelled => write!(f, "the trace was cancelled"),
            TraceError::Finished => write!(f, "the trace is already finished"),
            TraceError::Internal(Internal { what, location }) => write!(
                f,
                "internal error in ollev at {location}: expected {what} (please report this)"
            ),
        }
    }
}

impl std::error::Error for TraceError {}

impl From<TryReserveError> for TraceError {
    fn from(_: TryReserveError) -> Self {
        TraceError::OutOfMemory
    }
}

/// The internal error for `what` not holding, at the caller.
#[track_caller]
pub(crate) fn internal(what: &'static str) -> TraceError {
    TraceError::Internal(Internal {
        what,
        location: Location::caller(),
    })
}

/// An `Option` the tracer expects to hold a value: the checked `expect`.
pub(crate) trait Invariant<T> {
    /// The value, else [`TraceError::Internal`] for `what` at the caller.
    fn invariant(self, what: &'static str) -> Result<T, TraceError>;
}

impl<T> Invariant<T> for Option<T> {
    #[track_caller]
    #[inline]
    fn invariant(self, what: &'static str) -> Result<T, TraceError> {
        match self {
            Some(value) => Ok(value),
            None => Err(internal(what)),
        }
    }
}

/// A vector with room for `n` values, else [`TraceError::OutOfMemory`].
pub(crate) fn with_capacity<T>(n: usize) -> Result<Vec<T>, TraceError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)?;
    Ok(v)
}

/// `n` copies of `value`, else [`TraceError::OutOfMemory`].
pub(crate) fn filled<T: Clone>(value: T, n: usize) -> Result<Vec<T>, TraceError> {
    let mut v = with_capacity(n)?;
    v.resize(n, value);
    Ok(v)
}

/// The values a `width` by `height` raster of `per` values a pixel holds, else
/// [`TraceError::TooLarge`].
pub(crate) fn area(width: usize, height: usize, per: usize) -> Result<usize, TraceError> {
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(per))
        .ok_or(TraceError::TooLarge { width, height })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_impossible_allocation_is_an_error() {
        assert_eq!(with_capacity::<u64>(usize::MAX).err(), Some(TraceError::OutOfMemory));
        assert_eq!(filled(0u8, usize::MAX).err(), Some(TraceError::OutOfMemory));
    }

    #[test]
    fn an_overflowing_area_is_too_large() {
        assert_eq!(area(3, 4, 4), Ok(48));
        assert_eq!(
            area(usize::MAX, 2, 1),
            Err(TraceError::TooLarge {
                width: usize::MAX,
                height: 2
            })
        );
    }

    #[test]
    fn a_broken_invariant_says_where() {
        let Err(TraceError::Internal(e)) = None::<()>.invariant("a value") else {
            panic!("an internal error");
        };
        assert_eq!(e.what(), "a value");
        assert_eq!(e.location().file(), file!());
    }
}
