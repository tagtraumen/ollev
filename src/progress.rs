//! Progress and time slices: the trace is an `async` pipeline that asks the host at its
//! checkpoints (the [`Ask`] futures below). [`drive`] polls it once with the host's callback
//! installed; a checkpoint whose callback breaks suspends the trace there, to be resumed by the
//! next [`drive`] (or dropped, which cancels it). No executor and no waker take part.

use std::cell::Cell;
use std::future::Future;
use std::ops::ControlFlow;
#[cfg(feature = "diagnostics")]
use std::panic::Location;
use std::pin::Pin;
use std::ptr::NonNull;
use std::task::Poll;

use crate::Scene;
use crate::model::Model;

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
    finish: Option<&'a crate::tracing::Finish>,
}

impl Progress<'_> {
    /// The scene as it stands, as the model the finished trace returns (scaled to the
    /// image's size and lowered to the host); none where this checkpoint shows no scene.
    /// Lowering a large scene takes a while, so a host asks for this only when it shows it.
    #[must_use]
    pub fn model(&self) -> Option<Model> {
        let scene = self.scene?;
        Some(self.finish.map_or_else(|| scene.model(), |f| f.model(scene)))
    }
}

/// A host's callback: `ControlFlow::Break` ends the slice (see [`crate::Tracing::step`]).
pub type Callback<'a> = dyn FnMut(&Progress) -> ControlFlow<()> + 'a;

/// What a callback may answer: nothing (go on) or whether to go on.
pub trait Answer {
    /// Whether the trace goes on.
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

/// A trace's place between two slices: its stage, and where the last slice ended.
#[derive(Default)]
pub(crate) struct Place {
    stage: (&'static str, f64),
    /// How deep the traces beneath peeled layers are nested where the trace is.
    depth: usize,
    /// The checkpoint the trace is suspended at: its stage and the scene it shows, which the
    /// suspended trace holds borrowed, so it stands until the next [`drive`].
    paused: Option<(&'static str, f64, Option<NonNull<Scene>>)>,
    finish: Option<crate::tracing::Finish>,
    #[cfg(feature = "diagnostics")]
    gaps: Option<gaps::Gaps>,
}

impl Place {
    /// A trace's place before its first slice, its scenes made the host's model by `finish`.
    pub(crate) fn new(finish: crate::tracing::Finish) -> Self {
        Self {
            finish: Some(finish),
            ..Self::default()
        }
    }

    /// `scene` as the host's model.
    pub(crate) fn model(&self, scene: &Scene) -> Model {
        match &self.finish {
            Some(finish) => finish.model(scene),
            None => scene.model(),
        }
    }

    /// Where the last slice ended, valid while the trace stays suspended.
    pub(crate) fn paused(&self) -> Progress<'_> {
        let (stage, fraction, scene) = self.paused.unwrap_or((self.stage.0, self.stage.1, None));
        Progress {
            stage,
            fraction,
            // SAFETY: the scene is borrowed by the suspended trace, which `&self`'s owner
            // does not poll again while this borrow lasts
            scene: scene.map(|s| unsafe { s.as_ref() }),
            finish: self.finish.as_ref(),
        }
    }
}

/// What the checkpoints of the trace being polled reach: the host's callback and the place.
struct Driver<'a> {
    callback: &'a mut Callback<'a>,
    place: &'a mut Place,
}

thread_local! {
    /// The driver of the trace being polled; only [`drive`] sets it, for the time of a poll.
    static DRIVER: Cell<Option<NonNull<Driver<'static>>>> = const { Cell::new(None) };
}

/// Puts back the driver a [`drive`] found, on any exit.
struct Restore(Option<NonNull<Driver<'static>>>);

impl Drop for Restore {
    fn drop(&mut self) {
        DRIVER.set(self.0);
    }
}

/// Polls `trace` once with `callback` hearing its checkpoints: ready, or suspended at the
/// first checkpoint whose callback breaks. Drives nest: a trace a callback starts is its own.
pub(crate) fn drive<F: Future + ?Sized>(
    trace: Pin<&mut F>,
    place: &mut Place,
    callback: &mut Callback,
) -> Poll<F::Output> {
    place.paused = None;
    #[cfg(feature = "diagnostics")]
    {
        if place.gaps.is_none() && crate::diagnostics::log() {
            place.gaps = Some(gaps::Gaps::default());
        }
        // the host's time between two slices is no gap
        if let Some(g) = &mut place.gaps {
            g.resume();
        }
    }
    let mut driver = Driver { callback, place };
    // SAFETY: the pointer is only read while this frame lives (the guard takes it out on any
    // exit, unwinding included), and by one checkpoint at a time
    let erased: NonNull<Driver<'static>> =
        unsafe { std::mem::transmute(NonNull::from(&mut driver)) };
    let restore = Restore(DRIVER.replace(Some(erased)));
    let poll = trace.poll(&mut std::task::Context::from_waker(std::task::Waker::noop()));
    drop(restore);
    #[cfg(feature = "diagnostics")]
    if poll.is_ready()
        && let Some(g) = driver.place.gaps.take()
    {
        g.report();
    }
    poll
}

/// Runs `trace` to its end in one slice, hearing nothing.
#[cfg(any(test, feature = "diagnostics"))]
pub fn finish<F: Future>(trace: F) -> F::Output {
    let mut trace = std::pin::pin!(trace);
    match drive(trace.as_mut(), &mut Place::default(), &mut |_| {
        ControlFlow::Continue(())
    }) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("a callback that never breaks"),
    }
}

/// A checkpoint: asks the host on its first poll, and is ready on the next where the host's
/// callback broke, the trace having been suspended in between.
pub(crate) struct Ask<'a> {
    set: Set,
    scene: Option<&'a Scene>,
    asked: bool,
    #[cfg(feature = "diagnostics")]
    at: &'static Location<'static>,
}

/// What a checkpoint tells of the stage.
#[derive(Clone, Copy)]
enum Set {
    Nothing,
    Stage(&'static str),
    At(usize, usize),
}

impl Future for Ask<'_> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _: &mut std::task::Context) -> Poll<()> {
        if std::mem::replace(&mut self.asked, true) {
            return Poll::Ready(());
        }
        let Some(mut driver) = DRIVER.get() else {
            return Poll::Ready(());
        };
        // SAFETY: set by `drive` for the time of this poll; the callback runs with no other
        // reference to the driver in use
        let driver = unsafe { driver.as_mut() };
        let place = &mut *driver.place;
        #[cfg(feature = "diagnostics")]
        if let Some(g) = &mut place.gaps {
            g.note(self.at);
        }
        match self.set {
            Set::Nothing => {}
            Set::Stage(stage) => place.stage = (stage, 0.),
            Set::At(done, of) => place.stage.1 = done as f64 / of.max(1) as f64,
        }
        let progress = Progress {
            stage: place.stage.0,
            fraction: place.stage.1,
            scene: self.scene,
            finish: place.finish.as_ref(),
        };
        if (driver.callback)(&progress).is_continue() {
            return Poll::Ready(());
        }
        place.paused = Some((place.stage.0, place.stage.1, self.scene.map(NonNull::from)));
        Poll::Pending
    }
}

/// `f` on the place of the trace being polled, if any.
fn with_place<R>(f: impl FnOnce(&mut Place) -> R) -> Option<R> {
    // SAFETY: set by `drive` for the time of this poll; no checkpoint is in its callback
    DRIVER.get().map(|mut d| f(unsafe { d.as_mut() }.place))
}

/// How deep the traces beneath peeled layers are nested where the trace is: 0 in the
/// picture's own.
pub(crate) fn depth() -> usize {
    with_place(|p| p.depth).unwrap_or(0)
}

/// Enters a trace beneath a peeled layer (`deeper`), or leaves it.
pub(crate) fn nest(deeper: bool) {
    with_place(|p| p.depth = if deeper { p.depth + 1 } else { p.depth - 1 });
}

#[track_caller]
fn ask(set: Set, scene: Option<&Scene>) -> Ask<'_> {
    Ask {
        set,
        scene,
        asked: false,
        #[cfg(feature = "diagnostics")]
        at: Location::caller(),
    }
}

/// Enters `stage` and asks the host.
#[track_caller]
pub(crate) fn stage(stage: &'static str) -> Ask<'static> {
    ask(Set::Stage(stage), None)
}

/// Tells the host `done` of `of` steps of the stage are done.
#[track_caller]
pub(crate) fn at(done: usize, of: usize) -> Ask<'static> {
    ask(Set::At(done, of), None)
}

/// Asks the host whether to go on.
#[track_caller]
pub(crate) fn poll() -> Ask<'static> {
    ask(Set::Nothing, None)
}

/// Shows the host the scene as it now stands, where `show` (the picture's own trace, not a
/// trace beneath a layer).
#[track_caller]
pub(crate) fn scene(scene: &Scene, show: bool) -> Ask<'_> {
    ask(Set::Nothing, show.then_some(scene))
}
