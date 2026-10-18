//! The gap profiler, for the log: the longest stretches between two checkpoints, which bound
//! how far a slice runs past its budget.

use std::panic::Location;
use std::time::Duration;

use crate::timing::Clock;

/// The longest gaps between checkpoints kept for the log.
const GAPS: usize = 12;

/// The gaps between checkpoints: the clock at the last one and where it was, and the longest
/// gaps with their two places.
#[derive(Default)]
pub(super) struct Gaps {
    last: Option<(Clock, &'static Location<'static>)>,
    asks: usize,
    longest: Vec<Gap>,
}

type Gap = (
    Duration,
    &'static Location<'static>,
    &'static Location<'static>,
);

impl Gaps {
    /// A slice begins: the time since the last checkpoint was the host's.
    pub(super) fn resume(&mut self) {
        if let Some((clock, _)) = &mut self.last {
            *clock = Clock::now();
        }
    }

    /// The gap since the last checkpoint, which was `at`.
    pub(super) fn note(&mut self, at: &'static Location<'static>) {
        let now = Clock::now();
        if let Some((last, from)) = self.last {
            self.asks += 1;
            let gap = (last.elapsed(), from, at);
            if self.longest.len() < GAPS || gap.0 > self.longest[GAPS - 1].0 {
                self.longest.push(gap);
                self.longest.sort_by_key(|a| std::cmp::Reverse(a.0));
                self.longest.truncate(GAPS);
            }
        }
        self.last = Some((now, at));
    }

    /// The longest gaps, to the log.
    pub(super) fn report(&self) {
        eprintln!("progress: {} asks, the longest gaps:", self.asks);
        for (gap, from, to) in &self.longest {
            eprintln!("  {gap:>9.2?} from {from} to {to}");
        }
    }
}
