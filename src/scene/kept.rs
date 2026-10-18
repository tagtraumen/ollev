//! The caches a scene fills in as it is read, behind a lock so a scene can be shared.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// A value behind a lock, cloned with what it holds.
#[derive(Debug, Default)]
pub struct Kept<T>(Mutex<T>);

impl<T> Kept<T> {
    pub const fn new(value: T) -> Self {
        Self(Mutex::new(value))
    }

    /// The value, for the time of the guard.
    pub fn lock(&self) -> MutexGuard<'_, T> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T: Clone> Clone for Kept<T> {
    fn clone(&self) -> Self {
        Self::new(self.lock().clone())
    }
}
