//! The host's entry: a trace run in time slices ([`Tracing`]), or to its end ([`trace`],
//! [`trace_with`]).

use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use crate::input::{Detail, Image, TraceError, TraceOptions};
use crate::lower::Capabilities;
use crate::model::Model;
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
/// `progress` breaks, the trace stops (within about a second).
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
/// checkpoints come at least about once a second; dropping the handle cancels the trace. The
/// handle stays on the thread that made it.
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
    trace: Option<Pin<Box<dyn Future<Output = crate::Scene>>>>,
    place: Place,
    #[cfg(feature = "diagnostics")]
    diagnostics: crate::Diagnostics,
}

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
        let detail = match options.detail {
            Detail::Auto => crate::tracer::auto_detail(&rgb, tw, th),
            Detail::Absolute(d) => d,
            Detail::Scaled(f) => crate::tracer::auto_detail(&rgb, tw, th) * f,
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
        let place = Place::new(Finish {
            scale,
            width: w,
            height: h,
            clip,
            host: options.host,
        });
        Ok(Self {
            trace: Some(Box::pin(async move {
                crate::plan::traced(&rgb, tw, th, detail, plan).await.0
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
            Poll::Ready(scene) => {
                self.trace = None;
                Step::Done(self.place.model(&scene))
            }
            Poll::Pending => Step::Paused(self.place.paused()),
        }
    }
}

/// What turns a scene into the host's model: scaled back to the image's size, clipped to its
/// opaque part where transparency is kept, and lowered to what the host draws.
#[derive(Debug)]
pub(crate) struct Finish {
    scale: f64,
    width: usize,
    height: usize,
    clip: Option<crate::model::Clip>,
    host: Capabilities,
}

impl Finish {
    pub(crate) fn model(&self, scene: &crate::Scene) -> Model {
        let mut model = scene.model();
        if self.scale != 1. {
            model = model.scaled(self.scale, self.width, self.height);
        }
        if let Some(clip) = &self.clip {
            model.clips.push(clip.clone());
            model.clip = Some(model.clips.len() - 1);
        }
        crate::lower::lower(&model, &self.host).0
    }
}

#[cfg(test)]
mod tests;
