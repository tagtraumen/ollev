//! Where the time goes: named stages with their total time and number of calls, printed when the
//! caller asks for the log (see [`crate::diagnostics`]). Timing a stage costs a clock read on each side.

use std::time::Duration;

/// A clock the trace's logs and reports measure with. Where the platform has none (wasm without
/// a host, whose `Instant::now` panics) it reads nothing and every span is zero; the trace
/// itself never reads it, so its output is the same everywhere.
#[derive(Clone, Copy, Debug)]
pub struct Clock(
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))] std::time::Instant,
);

impl Clock {
    /// The clock's reading now.
    pub fn now() -> Self {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        return Self(std::time::Instant::now());
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        Self()
    }

    /// The time since [`Clock::now`], zero without a clock.
    pub fn elapsed(&self) -> Duration {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        return self.0.elapsed();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        Duration::ZERO
    }
}

/// Named stages with their total time and number of calls, in the order they first ran.
#[derive(Default)]
pub struct Timings {
    stages: Vec<(&'static str, Duration, usize)>,
}

impl Timings {
    /// Runs `f` as part of `stage`, adding its time to the stage's total; a checkpoint first
    /// (see [`crate::progress`]).
    #[track_caller]
    pub fn time<T>(
        &mut self,
        stage: &'static str,
        f: impl AsyncFnOnce() -> T,
    ) -> impl Future<Output = T> {
        let ask = crate::progress::poll();
        async move {
            ask.await;
            if !enabled() {
                return f().await;
            }
            let start = Clock::now();
            let value = f().await;
            self.add(stage, start.elapsed());
            value
        }
    }

    /// Adds a call of `duration` to `stage`.
    pub fn add(&mut self, stage: &'static str, duration: Duration) {
        match self.stages.iter_mut().find(|(name, _, _)| *name == stage) {
            Some((_, total, calls)) => {
                *total += duration;
                *calls += 1;
            }
            None => self.stages.push((stage, duration, 1)),
        }
    }

    /// Prints the stages in the order they first ran, when the log is asked for.
    pub fn report(&self, title: &str) {
        if !enabled() {
            return;
        }
        let total: Duration = self.stages.iter().map(|(_, d, _)| *d).sum();
        eprintln!("{title}: {total:.1?} in stages");
        for (stage, duration, calls) in &self.stages {
            eprintln!("  {stage:<16} {duration:>9.2?} {calls:>6}x");
        }
    }
}

/// Whether detailed logs are wanted (see [`crate::diagnostics`]).
pub fn enabled() -> bool {
    crate::diagnostics::log()
}
