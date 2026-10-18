//! What a trace reports and dumps for its developers, with the `diagnostics` feature, as its
//! caller asks through [`Diagnostics`]; nothing by default. Without the feature the logs' tests
//! are constant false and the dumps are not compiled.

#[cfg(feature = "diagnostics")]
pub use asked::*;

#[cfg(not(feature = "diagnostics"))]
pub(crate) use off::*;

#[cfg(feature = "diagnostics")]
mod asked {
    use std::cell::RefCell;
    use std::path::PathBuf;
    use std::sync::Arc;

    /// The reports and dumps asked for.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct Diagnostics {
        /// Stage timings, per-pass counts and the longest gaps between progress asks to stderr.
        pub log: bool,
        /// The translucency decisions to stderr.
        pub levels: bool,
        /// Detailed reads of the translucency stages to stderr.
        pub debug: bool,
        /// Where the kept layers' regions are written as `layer-N.pgm`.
        pub mask_dir: Option<PathBuf>,
        /// Where the images beneath peeled layers and their traces are written.
        pub peel_dir: Option<PathBuf>,
        /// A truth mask of the picture's layer, one value per pixel, true inside, which the
        /// log scores candidate regions against.
        pub truth: Option<Arc<Vec<bool>>>,
        /// Every shape left opaque: no layer peeled, for the lab.
        pub no_peel: bool,
    }

    thread_local! {
        static CURRENT: RefCell<Option<Arc<Diagnostics>>> = const { RefCell::new(None) };
    }

    /// Runs `f` with `diagnostics` asked for; scopes nest, the inner one standing while it runs.
    pub fn scope<R>(diagnostics: &Diagnostics, f: impl FnOnce() -> R) -> R {
        struct Restore(Option<Arc<Diagnostics>>);
        impl Drop for Restore {
            fn drop(&mut self) {
                CURRENT.set(self.0.take());
            }
        }
        let _restore = Restore(CURRENT.replace(Some(Arc::new(diagnostics.clone()))));
        f()
    }

    fn read<T>(f: impl FnOnce(&Diagnostics) -> T) -> Option<T> {
        CURRENT.with_borrow(|d| d.as_deref().map(f))
    }

    /// Whether stage timings and per-pass counts are logged.
    pub fn log() -> bool {
        read(|d| d.log).unwrap_or(false)
    }

    /// Whether the translucency decisions are logged.
    pub fn levels() -> bool {
        read(|d| d.levels).unwrap_or(false)
    }

    /// Whether the translucency stages' detailed reads are logged.
    pub fn debug() -> bool {
        read(|d| d.debug).unwrap_or(false)
    }

    /// Whether every shape is left opaque.
    pub fn no_peel() -> bool {
        read(|d| d.no_peel).unwrap_or(false)
    }

    /// Where the kept layers' regions go, if anywhere.
    pub fn mask_dir() -> Option<PathBuf> {
        read(|d| d.mask_dir.clone()).flatten()
    }

    /// Where the images beneath peeled layers go, if anywhere.
    pub fn peel_dir() -> Option<PathBuf> {
        read(|d| d.peel_dir.clone()).flatten()
    }

    /// The truth mask the log scores regions against, if one is given.
    pub fn truth() -> Option<Arc<Vec<bool>>> {
        read(|d| d.truth.clone()).flatten()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// What a scope asks for holds inside it only, an inner scope's while it runs.
        #[test]
        fn diagnostics_hold_in_their_scope() {
            assert!(!log() && mask_dir().is_none());
            let asked = Diagnostics {
                log: true,
                mask_dir: Some("masks".into()),
                ..Diagnostics::default()
            };
            scope(&asked, || {
                assert!(log() && !levels());
                assert_eq!(mask_dir(), Some(PathBuf::from("masks")));
                scope(&Diagnostics::default(), || assert!(!log()));
                assert!(log());
            });
            assert!(!log());
        }
    }
}

/// Without the feature every report is off, at no cost.
#[cfg(not(feature = "diagnostics"))]
mod off {
    #[inline(always)]
    pub(crate) const fn log() -> bool {
        false
    }

    #[inline(always)]
    pub(crate) const fn levels() -> bool {
        false
    }

    #[inline(always)]
    pub(crate) const fn debug() -> bool {
        false
    }

    #[inline(always)]
    pub(crate) const fn no_peel() -> bool {
        false
    }
}
