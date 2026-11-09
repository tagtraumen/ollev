//! Progress and time slices: the trace is an `async` pipeline that stops at its checkpoints
//! (the [`Ask`] futures below). [`drive`] polls it, telling the host's callback of each
//! checkpoint between two polls; a callback that breaks leaves the trace suspended there, to be
//! resumed by the next [`drive`] (or dropped, which cancels it). No executor and no waker.
//!
//! How much of the whole trace is done ([`Progress::overall`]) is read off nested [`Span`]s:
//! the trace's own stages, a stage's steps, a trace beneath a layer within its step, each with
//! weights for the time its parts take. A step entered splits what is left of its span among
//! itself and the steps after it, so a part that turns out larger or never runs moves the
//! split of the rest, never what is done.

// Panic ratchet: the lints this module has yet to meet, each removed as it is cleared
#![expect(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    reason = "panic ratchet: remove a lint once the module is clean of it"
)]

use std::cell::RefCell;
use std::future::Future;
use std::ops::ControlFlow;
#[cfg(feature = "diagnostics")]
use std::panic::Location;
use std::pin::Pin;
use std::task::Poll;

use crate::error::{Invariant, TraceError};
use crate::model::{Fidelity, Model};
use crate::{Scene, Shape};

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
    /// How much of the whole trace is done, from 0 to 1: never less than at an earlier
    /// checkpoint, 1 only once the trace is done. The trace's parts are weighted by the time
    /// they take on typical pictures, the families of translucent layers by how many there
    /// turn out to be, so the time left is an estimate.
    pub overall: f64,
    /// The scene as it stands, where the trace has just changed it.
    pub scene: Option<&'a Scene>,
    /// The latest scene kept for showing, and how many were kept up to it.
    latest: Option<&'a Model>,
    revision: u64,
    fidelity: Option<&'a Fidelity>,
    finish: Option<&'a crate::tracing::Finish>,
}

impl Progress<'_> {
    /// The latest scene the trace kept for showing, as a model in the form the finished trace
    /// returns (scaled to the image's size and lowered to the host); none before the first.
    /// A large picture shows flat regions within milliseconds, refined in steps, then the
    /// trace's own scenes as they grow, kept at most checkpoints. Objects lowered for an
    /// earlier call are kept; others take the cheap form of each rewrite (a blur or a radial
    /// focus dropped), which the result does not.
    #[must_use]
    pub fn model(&self) -> Option<Model> {
        let model = self.latest?.clone();
        Some(match self.finish {
            Some(f) => f.partial(model, self.fidelity),
            None => model,
        })
    }

    /// Whether [`Progress::model`] has a scene to show.
    #[must_use]
    pub fn showable(&self) -> bool {
        self.latest.is_some()
    }

    /// How many scenes the trace has kept for showing: [`Progress::model`] is the same at two
    /// checkpoints of one revision.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
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
    finish: Option<std::sync::Arc<crate::tracing::Finish>>,
}

/// What the checkpoints of a trace reach while it is polled (see [`TRACK`]).
#[derive(Default)]
struct Track {
    /// The stage the host is told of, with where it starts and ends in the whole trace.
    stage: (&'static str, f64, f64),
    /// How much of the whole trace is done (see [`Progress::overall`]).
    overall: f64,
    /// The spans the trace is in, outermost first.
    spans: Vec<Span>,
    /// How deep the traces beneath peeled layers are nested where the trace is.
    depth: usize,
    /// Whether a checkpoint has suspended the trace in the poll.
    asked: bool,
    /// The scene the suspending checkpoint shows, lent until the trace is polled again.
    shown: Option<Scene>,
    /// The latest scene kept for showing, its count of objects and of the scene's changes (see
    /// [`keep`]), and how many were kept.
    latest: Option<Model>,
    kept: usize,
    kept_changes: u64,
    revision: u64,
    /// Whether the latest is a first look (see [`preview`]).
    first_look: bool,
    /// What a partial scene's lowering is judged by, from the trace's start.
    fidelity: Option<Fidelity>,
    #[cfg(feature = "diagnostics")]
    gaps: Option<gaps::Gaps>,
}

/// A part of the whole trace, from where it was entered to `end`, split into weighted steps
/// (see [`enter`]).
struct Span {
    end: f64,
    /// The steps by name with their weights, and the one the span is in (none before the
    /// first).
    steps: Vec<(&'static str, f64)>,
    step: Option<usize>,
    /// Where the step the span is in starts and ends in the whole trace.
    from: f64,
    to: f64,
    /// Whether the span's scenes are the picture's own, which the host is shown.
    shown: bool,
    /// The stage the host was told of where the span was entered.
    outer: (&'static str, f64, f64),
}

impl Track {
    /// Moves the whole trace's progress to `at`, where that is further.
    fn reach(&mut self, at: f64) {
        self.overall = self.overall.max(at.min(1.));
    }

    /// Enters the step `name` of the innermost span, the next one by that name: it takes its
    /// weight's share of what is left of the span. A name the span does not list leaves it.
    fn step(&mut self, name: &str) {
        let at = self.overall;
        let Some(span) = self.spans.last_mut() else {
            return;
        };
        let next = span.step.map_or(0, |k| k + 1);
        let Some(k) = span.steps.iter().skip(next).position(|s| s.0 == name) else {
            return;
        };
        let k = k + next;
        span.step = Some(k);
        span.from = at;
        span.to = span.share(k);
    }

    /// Whether a scene kept now is the picture's own.
    fn shown(&self) -> bool {
        self.depth == 0 && self.spans.iter().all(|s| s.shown)
    }

    /// How much of the stage the host is told of is done.
    fn fraction(&self) -> f64 {
        let (_, from, to) = self.stage;
        if to > from {
            ((self.overall - from) / (to - from)).clamp(0., 1.)
        } else {
            0.
        }
    }

    /// Where the step the innermost span is in starts and ends, or what is left of the
    /// whole trace outside any.
    fn range(&self) -> (f64, f64) {
        match self.spans.last() {
            Some(s) if s.step.is_some() => (s.from, s.to),
            Some(s) => (self.overall, s.end),
            None => (self.overall, 1.),
        }
    }
}

impl Span {
    /// Where step `k` ends, entered at `from`: its weight's share of the span left after it
    /// and the steps after it.
    fn share(&self, k: usize) -> f64 {
        let rest: f64 = self.steps.iter().skip(k).map(|s| s.1).sum();
        let weight = self.steps.get(k).map_or(0., |s| s.1);
        if rest > 0. {
            self.from + (self.end - self.from).max(0.) * weight / rest
        } else {
            self.from
        }
    }
}

impl Place {
    /// A trace's place before its first slice, its scenes made the host's model by `finish`.
    pub(crate) fn new(finish: std::sync::Arc<crate::tracing::Finish>) -> Self {
        Self {
            finish: Some(finish),
            ..Self::default()
        }
    }

    /// The checkpoint the trace is suspended at.
    pub(crate) fn progress(&self) -> Progress<'_> {
        Progress {
            stage: self.track.stage.0,
            fraction: self.track.fraction(),
            overall: self.track.overall,
            scene: self.track.shown.as_ref(),
            latest: self.track.latest.as_ref(),
            revision: self.track.revision,
            fidelity: self.track.fidelity.as_ref(),
            finish: self.finish.as_deref(),
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
        debug_assert!(
            place.track.asked,
            "a trace suspends only at its checkpoints"
        );
        if callback(&place.progress()).is_break() {
            return Poll::Pending;
        }
    }
}

/// Runs `trace` to its end in one slice, hearing nothing.
pub fn finish<F: Future>(trace: F) -> F::Output {
    let mut trace = std::pin::pin!(trace);
    let mut place = Place::default();
    // the callback never breaks, so the first drive is the last
    loop {
        if let Poll::Ready(value) =
            drive(trace.as_mut(), &mut place, &mut |_| ControlFlow::Continue(()))
        {
            return value;
        }
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
    Step(&'static str),
    At(usize, usize),
}

impl Ask {
    /// Polls the checkpoint, lending the trace's track `scene` on suspending (kept for showing
    /// where `kept`); the scene comes back on the poll that is ready.
    fn ask(&mut self, scene: &mut Option<Scene>, kept: bool) -> Poll<()> {
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
                Set::Stage(stage) => {
                    track.step(stage);
                    let (from, to) = track.range();
                    track.stage = (stage, from, to);
                }
                Set::Step(step) => track.step(step),
                Set::At(done, of) => {
                    let (from, to) = track.range();
                    track.reach(from + (to - from) * (done as f64 / of.max(1) as f64).min(1.));
                }
            }
            track.asked = true;
            if let Some(shown) = scene.as_ref()
                && kept
                && track.shown()
            {
                keep(track, shown.shapes.len(), shown.changes(), false, || {
                    shown.model()
                });
            }
            track.shown = scene.take();
            Poll::Pending
        })
    }
}

impl Future for Ask {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _: &mut std::task::Context) -> Poll<()> {
        self.ask(&mut None, false)
    }
}

/// A checkpoint showing the host a scene, which it hands back.
pub(crate) struct Show {
    ask: Ask,
    scene: Option<Scene>,
    /// Whether the host sees the scene.
    visible: bool,
    /// Whether the scene is kept for showing (see [`keep`]).
    kept: bool,
}

impl Future for Show {
    type Output = Result<Scene, TraceError>;

    fn poll(mut self: Pin<&mut Self>, _: &mut std::task::Context) -> Poll<Self::Output> {
        let Self {
            ask,
            scene,
            visible,
            kept,
        } = &mut *self;
        let mut shown = if *visible { scene.take() } else { None };
        let poll = ask.ask(&mut shown, *kept);
        if shown.is_some() {
            *scene = shown;
        }
        poll.map(|()| scene.take().invariant("the scene shown handed back"))
    }
}

/// `f` on the track of the trace being polled, if any.
fn with_track<R>(f: impl FnOnce(&mut Track) -> R) -> Option<R> {
    TRACK.with_borrow_mut(|track| track.as_mut().map(f))
}

/// Keeps `model` as the latest to show, a scene of `n` objects after `changes` changes to it,
/// where `always` or where its objects or changes have moved by a 32nd of the objects since
/// the last kept (a model per change would cost the square of the shapes); a first look gives
/// way only to as many objects as it has regions.
fn keep(track: &mut Track, n: usize, changes: u64, always: bool, model: impl FnOnce() -> Model) {
    let moved = (n.abs_diff(track.kept) as u64).max(changes.abs_diff(track.kept_changes));
    let held = track.first_look && n < track.kept;
    if always || (!held && moved >= (track.kept as u64 / 32).max(1)) {
        track.latest = Some(model());
        track.kept = n;
        track.kept_changes = changes;
        track.first_look = false;
        track.revision += 1;
    }
}

/// Whether a host hears the trace's checkpoints.
pub(crate) fn driven() -> bool {
    with_track(|_| ()).is_some()
}

/// Keeps `model` as the scene to show until the picture's own trace shows another: a first
/// look before its first scene.
pub(crate) fn preview(model: Model) {
    with_track(|t| {
        if t.shown() {
            t.kept = model.objects().len();
            t.latest = Some(model);
            t.first_look = true;
            t.revision += 1;
        }
    });
}

/// Keeps the picture's own `scene` (not one beneath a layer) as the scene to show, where a
/// stage leaves it whole.
pub(crate) fn snapshot(scene: &Scene) {
    with_track(|t| {
        if t.shown() {
            keep(t, scene.shapes.len(), scene.changes(), true, || {
                scene.model()
            });
        }
    });
}

/// Keeps the picture's own `scene` as the scene to show where it has changed enough since the
/// last kept (see [`keep`]): a stage changing the scene in place shows it as it goes.
pub(crate) fn glimpse(scene: &Scene) {
    with_track(|t| {
        if t.shown() {
            keep(t, scene.shapes.len(), scene.changes(), false, || {
                scene.model()
            });
        }
    });
}

/// [`glimpse`] of `scene` with the shapes `planned` to be drawn after it on top, as they are
/// before their fit (or kept `always`): the whole plan shows from the start of the drawing,
/// each shape fitted in its turn.
pub(crate) fn glimpse_over(scene: &Scene, planned: &[Shape], always: bool) {
    with_track(|t| {
        if t.shown() {
            let n = scene.shapes.len().saturating_add(planned.len());
            keep(t, n, scene.changes(), always, || {
                let mut model = scene.model();
                model.items.extend(planned.iter().map(|s| {
                    crate::model::Item::Object(Box::new(crate::model::Object::of_shape(s, None)))
                }));
                model
            });
        }
    });
}

/// What a partial scene's lowering is judged by: the trace's own measure, before the result
/// carries it.
pub(crate) fn fidelity(fidelity: Fidelity) {
    with_track(|t| t.fidelity = Some(fidelity));
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

/// Enters a span of `steps` (by name, with their weights for the time they take) over what is
/// left of the step the trace is in; its scenes are the picture's own where `shown`. Each
/// [`enter`] is closed by a [`leave`].
pub(crate) fn enter(steps: &[(&'static str, f64)], shown: bool) {
    with_track(|t| {
        let end = t.range().1;
        let at = t.overall;
        t.spans.push(Span {
            end,
            steps: steps.to_vec(),
            step: None,
            from: at,
            to: at,
            shown,
            outer: t.stage,
        });
    });
}

/// Leaves the innermost span (see [`enter`]), done: the trace is at its end, and the host is
/// told of the stage it was entered in again.
pub(crate) fn leave() {
    with_track(|t| {
        if let Some(span) = t.spans.pop() {
            t.reach(span.end);
            t.stage = span.outer;
        }
    });
}

/// Weighs the named steps anew, each the one the trace is in or a later one of the innermost
/// span listing it, where the trace learns how much they hold: what is left of each span is
/// split again from its current step's start, and the spans within moved to end with their
/// steps. A step entered at weight 0 does not run, and stays at 0.
pub(crate) fn weigh(steps: &[(&'static str, f64)]) {
    with_track(|t| {
        for &(name, weight) in steps {
            let found = t.spans.iter_mut().rev().find_map(|span| {
                let current = span.step.unwrap_or(0);
                span.steps.iter_mut().skip(current).find(|s| s.0 == name)
            });
            if let Some(step) = found
                && step.1 > 0.
            {
                step.1 = weight;
            }
        }
        // Outermost first, each span within ending where its parent's step now ends
        let mut end: Option<f64> = None;
        for span in &mut t.spans {
            if let Some(end) = end {
                span.end = end;
            }
            if let Some(k) = span.step {
                span.to = span.share(k);
                end = Some(span.to);
            } else {
                end = Some(span.end);
            }
        }
        // the stage the host is told of ends with its step
        let name = t.stage.0;
        if let Some(to) = t.spans.iter().rev().find_map(|s| {
            s.step
                .and_then(|k| s.steps.get(k))
                .is_some_and(|step| step.0 == name)
                .then_some(s.to)
        }) {
            t.stage.2 = to;
        }
    });
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

/// Enters `stage`, a step of the innermost span by the same name, and asks the host.
#[track_caller]
pub(crate) fn stage(stage: &'static str) -> Ask {
    ask(Set::Stage(stage))
}

/// Enters the step `step` of the innermost span (see [`enter`]), within the stage the host is
/// told of, and asks the host.
#[track_caller]
pub(crate) fn step(step: &'static str) -> Ask {
    ask(Set::Step(step))
}

/// Tells the host `done` of `of` parts of the step the trace is in are done.
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
        visible: show,
        kept: true,
    }
}

/// [`scene`] lent to the host but not kept for showing: the stage keeps what it shows itself
/// (see [`glimpse_over`]).
#[track_caller]
pub(crate) fn lend(scene: Scene, show: bool) -> Show {
    Show {
        kept: false,
        ..self::scene(scene, show)
    }
}
