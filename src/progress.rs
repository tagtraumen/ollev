//! Progress and time slices: the trace is an `async` pipeline that stops at its checkpoints
//! (the [`Ask`] futures below). [`drive`] polls it, telling the host's callback of each
//! checkpoint between two polls; a callback that breaks leaves the trace suspended there, to be
//! resumed by the next [`drive`] (or dropped, which cancels it). No executor and no waker.

use std::cell::RefCell;
use std::future::Future;
use std::ops::ControlFlow;
#[cfg(feature = "diagnostics")]
use std::panic::Location;
use std::pin::Pin;
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
        Some(
            self.finish
                .map_or_else(|| scene.model(), |f| f.model(scene)),
        )
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

/// A trace's place between two polls: what its checkpoints reach, and what makes its scenes
/// the host's model.
#[derive(Default)]
pub(crate) struct Place {
    track: Track,
    finish: Option<crate::tracing::Finish>,
}

/// What the checkpoints of a trace reach while it is polled (see [`TRACK`]).
#[derive(Default)]
struct Track {
    stage: (&'static str, f64),
    /// How deep the traces beneath peeled layers are nested where the trace is.
    depth: usize,
    /// Whether a checkpoint has suspended the trace in the poll.
    asked: bool,
    /// The scene the suspending checkpoint shows, lent until the trace is polled again.
    shown: Option<Scene>,
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

    /// The checkpoint the trace is suspended at.
    pub(crate) fn progress(&self) -> Progress<'_> {
        Progress {
            stage: self.track.stage.0,
            fraction: self.track.stage.1,
            scene: self.track.shown.as_ref(),
            finish: self.finish.as_ref(),
        }
    }
}

thread_local! {
    /// The track of the trace being polled; only [`drive`] sets it, for the time of a poll.
    static TRACK: RefCell<Option<Track>> = const { RefCell::new(None) };
}

/// A trace's track lent to [`TRACK`] for one poll; given back, and the track a nesting
/// [`drive`] found put back, on any exit.
struct Lent<'a> {
    place: &'a mut Place,
    outer: Option<Track>,
}

impl<'a> Lent<'a> {
    fn new(place: &'a mut Place) -> Self {
        let outer = TRACK.replace(Some(std::mem::take(&mut place.track)));
        Self { place, outer }
    }
}

impl Drop for Lent<'_> {
    fn drop(&mut self) {
        if let Some(track) = TRACK.replace(self.outer.take()) {
            self.place.track = track;
        }
    }
}

/// Runs `trace` with `callback` hearing its checkpoints: ready, or suspended at the first
/// checkpoint whose callback breaks. The callback runs between two polls, so a trace it
/// starts is its own.
pub(crate) fn drive<F: Future + ?Sized>(
    mut trace: Pin<&mut F>,
    place: &mut Place,
    callback: &mut Callback,
) -> Poll<F::Output> {
    #[cfg(feature = "diagnostics")]
    {
        let gaps = &mut place.track.gaps;
        if gaps.is_none() && crate::diagnostics::log() {
            *gaps = Some(gaps::Gaps::default());
        }
        // the host's time between two slices is no gap
        if let Some(g) = gaps {
            g.resume();
        }
    }
    loop {
        place.track.asked = false;
        let lent = Lent::new(place);
        let poll = trace
            .as_mut()
            .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()));
        drop(lent);
        if let Poll::Ready(value) = poll {
            #[cfg(feature = "diagnostics")]
            if let Some(g) = place.track.gaps.take() {
                g.report();
            }
            return Poll::Ready(value);
        }
        assert!(
            place.track.asked,
            "a trace suspends only at its checkpoints"
        );
        if callback(&place.progress()).is_break() {
            return Poll::Pending;
        }
    }
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

/// A checkpoint: its first poll tells the stage and suspends the trace, its next is ready.
/// Where no trace is being driven, it is ready at once.
pub(crate) struct Ask {
    set: Set,
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

impl Ask {
    /// Polls the checkpoint, lending the trace's track `scene` on suspending; the scene comes
    /// back on the poll that is ready.
    fn ask(&mut self, scene: &mut Option<Scene>) -> Poll<()> {
        TRACK.with_borrow_mut(|track| {
            let Some(track) = track else {
                return Poll::Ready(());
            };
            if std::mem::replace(&mut self.asked, true) {
                if let Some(shown) = track.shown.take() {
                    *scene = Some(shown);
                }
                return Poll::Ready(());
            }
            #[cfg(feature = "diagnostics")]
            if let Some(g) = &mut track.gaps {
                g.note(self.at);
            }
            match self.set {
                Set::Nothing => {}
                Set::Stage(stage) => track.stage = (stage, 0.),
                Set::At(done, of) => track.stage.1 = done as f64 / of.max(1) as f64,
            }
            track.asked = true;
            track.shown = scene.take();
            Poll::Pending
        })
    }
}

impl Future for Ask {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _: &mut std::task::Context) -> Poll<()> {
        self.ask(&mut None)
    }
}

/// A checkpoint showing the host a scene, which it hands back.
pub(crate) struct Show {
    ask: Ask,
    scene: Option<Scene>,
    show: bool,
}

impl Future for Show {
    type Output = Scene;

    fn poll(mut self: Pin<&mut Self>, _: &mut std::task::Context) -> Poll<Scene> {
        let Self { ask, scene, show } = &mut *self;
        let mut shown = if *show { scene.take() } else { None };
        let poll = ask.ask(&mut shown);
        if shown.is_some() {
            *scene = shown;
        }
        poll.map(|()| scene.take().expect("the scene shown"))
    }
}

/// `f` on the track of the trace being polled, if any.
fn with_track<R>(f: impl FnOnce(&mut Track) -> R) -> Option<R> {
    TRACK.with_borrow_mut(|track| track.as_mut().map(f))
}

/// How deep the traces beneath peeled layers are nested where the trace is: 0 in the
/// picture's own.
pub(crate) fn depth() -> usize {
    with_track(|t| t.depth).unwrap_or(0)
}

/// Enters a trace beneath a peeled layer (`deeper`), or leaves it.
pub(crate) fn nest(deeper: bool) {
    with_track(|t| t.depth = if deeper { t.depth + 1 } else { t.depth - 1 });
}

#[track_caller]
fn ask(set: Set) -> Ask {
    Ask {
        set,
        asked: false,
        #[cfg(feature = "diagnostics")]
        at: Location::caller(),
    }
}

/// Enters `stage` and asks the host.
#[track_caller]
pub(crate) fn stage(stage: &'static str) -> Ask {
    ask(Set::Stage(stage))
}

/// Tells the host `done` of `of` steps of the stage are done.
#[track_caller]
pub(crate) fn at(done: usize, of: usize) -> Ask {
    ask(Set::At(done, of))
}

/// Asks the host whether to go on.
#[track_caller]
pub(crate) fn poll() -> Ask {
    ask(Set::Nothing)
}

/// Shows the host `scene` as it now stands, where `show` (the picture's own trace, not a
/// trace beneath a layer), and hands it back.
#[track_caller]
pub(crate) fn scene(scene: Scene, show: bool) -> Show {
    Show {
        ask: ask(Set::Nothing),
        scene: Some(scene),
        show,
    }
}
