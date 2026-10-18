//! Progress and cancellation: a host's callback, installed for one trace by [`scope`], hears
//! the stage the trace is in and how far, and the scene as it grows; breaking stops the trace.
//! The tracer asks at every timed stage and in its long loops, so it stops within a fraction of
//! a second; once stopped every later step returns at once, its result to be thrown away.

use std::cell::Cell;
use std::ops::ControlFlow;
#[cfg(feature = "diagnostics")]
use std::panic::Location;
use std::ptr::NonNull;

use crate::Scene;

#[cfg(feature = "diagnostics")]
mod gaps;

/// Where a trace is.
#[derive(Clone, Copy, Debug)]
pub struct Progress<'a> {
    /// The stage, by name (`analyze`, `plan`, `draw`, `families`, `beneath`, `economy`...);
    /// the stages beneath a translucent layer repeat those of the picture.
    pub stage: &'static str,
    /// How much of the stage is done, from 0 to 1, where the stage knows (else 0).
    pub fraction: f64,
    /// The scene as it stands, where the trace has just changed it.
    pub scene: Option<&'a Scene>,
}

/// A host's callback: `ControlFlow::Break` cancels the trace.
pub type Callback<'a> = dyn FnMut(&Progress) -> ControlFlow<()> + 'a;

/// What a callback may answer: nothing (go on) or whether to go on.
pub trait Answer {
    fn flow(self) -> ControlFlow<()>;
}

impl Answer for () {
    fn flow(self) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

impl Answer for ControlFlow<()> {
    fn flow(self) -> ControlFlow<()> {
        self
    }
}

thread_local! {
    /// The installed callback, its lifetime erased; only [`scope`] sets it, and only for the
    /// time its borrow lasts.
    static CALLBACK: Cell<Option<NonNull<Callback<'static>>>> = const { Cell::new(None) };
    static CANCELLED: Cell<bool> = const { Cell::new(false) };
    static STAGE: Cell<(&'static str, f64)> = const { Cell::new(("start", 0.)) };
}

/// Runs `f` with `callback` hearing its progress; returns `f`'s value and whether the
/// callback cancelled it. Scopes nest: an inner one hears its own trace only.
#[track_caller]
pub fn scope<R>(callback: &mut Callback, f: impl FnOnce() -> R) -> (R, bool) {
    // SAFETY: the pointer is only read while this frame lives (the guard takes it out on any
    // exit, unwinding included), and never while the callback runs (see `call`)
    let erased: NonNull<Callback<'static>> =
        unsafe { std::mem::transmute(NonNull::from(callback)) };
    struct Restore {
        callback: Option<NonNull<Callback<'static>>>,
        cancelled: bool,
        stage: (&'static str, f64),
        #[cfg(feature = "diagnostics")]
        gaps: Option<gaps::Gaps>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            CALLBACK.set(self.callback);
            CANCELLED.set(self.cancelled);
            STAGE.set(self.stage);
            #[cfg(feature = "diagnostics")]
            gaps::GAPS_SEEN.set(self.gaps.take());
        }
    }
    let restore = Restore {
        callback: CALLBACK.replace(Some(erased)),
        cancelled: CANCELLED.replace(false),
        stage: STAGE.replace(("start", 0.)),
        #[cfg(feature = "diagnostics")]
        gaps: gaps::GAPS_SEEN.replace(crate::diagnostics::log().then(gaps::Gaps::default)),
    };
    let value = f();
    let cancelled = CANCELLED.get();
    // The stretch since the last ask, and the longest gaps, for the log
    #[cfg(feature = "diagnostics")]
    {
        gaps::note(cancelled, Location::caller());
        gaps::report();
    }
    drop(restore);
    (value, cancelled)
}

/// Whether the host has cancelled the trace.
pub fn cancelled() -> bool {
    CANCELLED.get()
}

/// Enters `stage` and asks the host; true once cancelled.
#[track_caller]
pub fn stage(stage: &'static str) -> bool {
    STAGE.set((stage, 0.));
    call(None)
}

/// Tells the host `done` of `of` steps of the stage are done; true once cancelled.
#[track_caller]
pub fn at(done: usize, of: usize) -> bool {
    let (stage, _) = STAGE.get();
    STAGE.set((stage, done as f64 / of.max(1) as f64));
    call(None)
}

/// Asks the host whether to go on; true once cancelled.
#[track_caller]
pub fn poll() -> bool {
    call(None)
}

/// Shows the host the scene as it now stands; true once cancelled.
#[track_caller]
pub fn scene(scene: &Scene) -> bool {
    call(Some(scene))
}

#[track_caller]
fn call(scene: Option<&Scene>) -> bool {
    let cancelled = CANCELLED.get();
    #[cfg(feature = "diagnostics")]
    gaps::note(cancelled, Location::caller());
    if cancelled {
        return true;
    }
    // Taken out while it runs, so a trace it starts hears nothing of this one's
    let Some(mut callback) = CALLBACK.take() else {
        return false;
    };
    let (stage, fraction) = STAGE.get();
    let progress = Progress {
        stage,
        fraction,
        scene,
    };
    // SAFETY: set by `scope`, whose borrow outlives this call; no other reference to it exists
    // while it is out of the cell
    let flow = unsafe { callback.as_mut()(&progress) };
    CALLBACK.set(Some(callback));
    if flow.is_break() {
        CANCELLED.set(true);
    }
    CANCELLED.get()
}

#[cfg(test)]
mod tests;
