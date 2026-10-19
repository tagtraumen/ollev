//! The host's entry: a trace run in time slices ([`Tracing`]), or to its end ([`trace`],
//! [`trace_with`]).

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::Poll;

use crate::input::{Detail, Image, TraceError, TraceOptions};
use crate::lower::Capabilities;
use crate::model::{Fidelity, Model};
use crate::progress::{Answer, Place, Progress};

/// Traces `image` into a scene model in what the host draws.
///
/// # Errors
///
/// [`TraceError::Empty`] or [`TraceError::Size`] where the image is not one to trace.
pub fn trace(image: &Image, options: &TraceOptions) -> Result<Model, TraceError> {
    trace_with(image, options, |_| ())
}

/// [`trace`], telling `progress` the stage, how far it is and the scene as it grows; where
/// `progress` breaks, the trace stops (within a fraction of a second).
///
/// # Errors
///
/// As [`trace`], and [`TraceError::Cancelled`] where `progress` broke.
pub fn trace_with<A: Answer>(
    image: &Image,
    options: &TraceOptions,
    mut progress: impl FnMut(&Progress) -> A,
) -> Result<Model, TraceError> {
    match Tracing::new(image, options)?.step(|p| progress(p).flow()) {
        Step::Done(model) => Ok(model),
        Step::Paused(_) => Err(TraceError::Cancelled),
    }
}

/// A trace that runs in slices on the host's thread: each [`Tracing::step`] runs it until the
/// host's callback ends the slice at a checkpoint, and the next step resumes it there. The
/// checkpoints come a few tenths of a second apart at most on a megapixel picture; dropping
/// the handle cancels the trace. The handle is `Send`: a host may step it on any one thread at
/// a time.
///
/// ```
/// let pixels = [255u8, 0, 0, 255].repeat(16 * 16);
/// let image = ollev::Image {
///     pixels: ollev::Pixels::Rgba8(&pixels),
///     width: 16,
///     height: 16,
/// };
/// let mut tracing = ollev::Tracing::new(&image, &ollev::TraceOptions::default()).unwrap();
/// // a host gives each slice a time budget, read off its own clock: here three checkpoints
/// let model = loop {
///     let mut asks = 0;
///     match tracing.step(|_| {
///         asks += 1;
///         if asks < 3 {
///             std::ops::ControlFlow::Continue(())
///         } else {
///             std::ops::ControlFlow::Break(())
///         }
///     }) {
///         ollev::Step::Paused(progress) => {
///             // show progress.stage, progress.fraction, progress.model()
///         }
///         ollev::Step::Done(model) => break model,
///     }
/// };
/// assert!(!model.objects().is_empty());
/// ```
pub struct Tracing {
    trace: Option<Pin<Box<dyn Future<Output = Model> + Send>>>,
    place: Place,
    #[cfg(feature = "diagnostics")]
    diagnostics: crate::Diagnostics,
}

// A host may move a trace between threads between two steps
const _: fn() = || {
    fn send<T: Send>() {}
    send::<Tracing>();
};

/// What a slice of a [`Tracing`] ends with.
#[derive(Debug)]
pub enum Step<'a> {
    /// The callback ended the slice at this checkpoint, where the next step resumes.
    Paused(Progress<'a>),
    /// The trace is done.
    Done(Model),
}

impl Tracing {
    /// A trace of `image`, its pixels copied; nothing is traced before the first step.
    ///
    /// # Errors
    ///
    /// As [`trace`].
    pub fn new(image: &Image, options: &TraceOptions) -> Result<Self, TraceError> {
        let (rgb, alpha) = crate::input::pixels8(image, options.matte)?;
        let (w, h) = (image.width, image.height);
        // Over the budget, the picture is traced at the size that holds it, one scale both ways
        let (rgb, tw, th, scale) = match options.max_pixels.map(|most| most.max(1)) {
            Some(most) if w * h > most => {
                let tw = ((w as f64 * (most as f64 / (w * h) as f64).sqrt()) as usize).max(1);
                let s = w as f64 / tw as f64;
                let th = ((h as f64 / s).round() as usize).max(1);
                (crate::input::downsampled(&rgb, w, h, tw, th, s), tw, th, s)
            }
            _ => (rgb, w, h, 1.),
        };
        let auto = || crate::tracer::auto_detail(&rgb, tw, th) * Detail::AUTO_SHARE;
        let detail = match options.detail {
            Detail::Auto => auto(),
            Detail::Absolute(d) => d,
            Detail::Scaled(f) => auto() * f,
        };
        let plan = crate::plan::Options {
            stages: options.stages,
            ..crate::plan::Options::new(options.tolerance)
        };
        let clip = (options.keep_transparency && alpha.iter().any(|&a| a < 1.)).then(|| {
            crate::model::Clip {
                loops: crate::lower::layers::loops_at(&alpha, w, h, 0.5),
                rule: crate::model::FillRule::EvenOdd,
                parent: None,
            }
        });
        let finish = Arc::new(Finish {
            scale,
            width: w,
            height: h,
            clip,
            host: options.host,
            memo: Mutex::default(),
        });
        let place = Place::new(finish.clone());
        Ok(Self {
            trace: Some(Box::pin(async move {
                let scene = crate::plan::traced(&rgb, tw, th, detail, plan).await.0;
                finish.finished(&scene).await
            })),
            place,
            #[cfg(feature = "diagnostics")]
            diagnostics: options.diagnostics.clone(),
        })
    }

    /// Runs the trace until `progress` breaks at a checkpoint (where the next step resumes)
    /// or the trace is done. A host with a clock gives each slice a budget: a callback that
    /// breaks once the time is up.
    ///
    /// # Panics
    ///
    /// When stepped again after [`Step::Done`].
    pub fn step<A: Answer>(&mut self, mut progress: impl FnMut(&Progress) -> A) -> Step<'_> {
        let trace = self.trace.as_mut().expect("a trace not yet done");
        let mut callback = |p: &Progress| progress(p).flow();
        let place = &mut self.place;
        #[cfg(feature = "diagnostics")]
        let poll = crate::diagnostics::scope(&self.diagnostics, || {
            crate::progress::drive(trace.as_mut(), place, &mut callback)
        });
        #[cfg(not(feature = "diagnostics"))]
        let poll = crate::progress::drive(trace.as_mut(), place, &mut callback);
        match poll {
            Poll::Ready(model) => {
                self.trace = None;
                Step::Done(model)
            }
            Poll::Pending => Step::Paused(self.place.progress()),
        }
    }
}

/// What turns a scene into the host's model: scaled back to the image's size, clipped to its
/// opaque part where transparency is kept, and lowered to what the host draws, each object
/// lowered once over the partial scenes and the result.
#[derive(Debug)]
pub(crate) struct Finish {
    scale: f64,
    width: usize,
    height: usize,
    clip: Option<crate::model::Clip>,
    host: Capabilities,
    memo: Mutex<crate::lower::Memo>,
}

impl Finish {
    /// The model of a scene at the image's size and clip, judged by `fidelity` where it has
    /// none of its own.
    fn sized(&self, mut model: Model, fidelity: Option<&Fidelity>) -> Model {
        if model.fidelity.is_none() {
            model.fidelity = fidelity.cloned();
        }
        if self.scale != 1. {
            model = model.scaled(self.scale, self.width, self.height);
        }
        if let Some(clip) = &self.clip {
            model.clips.push(clip.clone());
            model.clip = Some(model.clips.len() - 1);
        }
        model
    }

    /// A partial scene's model in the host's form: what was lowered before kept, the rest
    /// lowered quickly (see [`crate::lower`]).
    pub(crate) fn partial(&self, model: Model, fidelity: Option<&Fidelity>) -> Model {
        let model = self.sized(model, fidelity);
        crate::lower::lower_kept(&model, &self.host, &mut crate::lower::lock(&self.memo), true)
    }

    /// The trace's result in the host's form, its objects lowered one per checkpoint.
    async fn finished(&self, scene: &crate::Scene) -> Model {
        let model = self.sized(scene.model(), None);
        crate::lower::keep(&model, &self.host, &self.memo).await;
        crate::lower::lower_kept(&model, &self.host, &mut crate::lower::lock(&self.memo), false)
    }
}

#[cfg(test)]
mod tests;
