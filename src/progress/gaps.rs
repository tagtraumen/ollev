//! The gap profiler, for the log: the longest stretches between two asks of the host, which
//! bound how fast a cancel returns.

use std::cell::RefCell;
use std::panic::Location;
use std::time::Duration;

use crate::timing::Clock;

/// The longest gaps between two asks kept for the log.
const GAPS: usize = 12;

/// The gaps between asks: the clock at the last ask and where it was, and the longest gaps
/// with their two places.
#[derive(Default)]
pub(super) struct Gaps {
    last: Option<(Clock, &'static Location<'static>)>,
    asks: usize,
    longest: Vec<Gap>,
    /// The longest once cancelled, which bound how fast the trace returns.
    after: Vec<Gap>,
}

type Gap = (
    Duration,
    &'static Location<'static>,
    &'static Location<'static>,
);

/// Keeps `gap` among the [`GAPS`] longest of `list`.
fn keep(list: &mut Vec<Gap>, gap: Gap) {
    if list.len() < GAPS || gap.0 > list[GAPS - 1].0 {
        list.push(gap);
        list.sort_by_key(|a| std::cmp::Reverse(a.0));
        list.truncate(GAPS);
    }
}

thread_local! {
    pub(super) static GAPS_SEEN: RefCell<Option<Gaps>> = const { RefCell::new(None) };
}

/// The gap since the last ask, among those once cancelled where `cancelled`.
pub(super) fn note(cancelled: bool, at: &'static Location<'static>) {
    GAPS_SEEN.with_borrow_mut(|g| {
        if let Some(g) = g {
            let now = Clock::now();
            if let Some((last, from)) = g.last {
                g.asks += 1;
                let gap = (last.elapsed(), from, at);
                keep(
                    if cancelled {
                        &mut g.after
                    } else {
                        &mut g.longest
                    },
                    gap,
                );
            }
            g.last = Some((now, at));
        }
    });
}

/// The longest gaps between asks, to the log.
pub(super) fn report() {
    GAPS_SEEN.with_borrow(|g| {
        let Some(g) = g else { return };
        eprintln!("progress: {} asks, the longest gaps:", g.asks);
        for (gap, from, to) in &g.longest {
            eprintln!("  {gap:>9.2?} from {from} to {to}");
        }
        if !g.after.is_empty() {
            eprintln!("progress: the longest gaps once cancelled:");
        }
        for (gap, from, to) in &g.after {
            eprintln!("  {gap:>9.2?} from {from} to {to}");
        }
    });
}
