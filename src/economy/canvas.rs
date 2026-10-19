//! The whole scene rendered over black as the economy last saw it, read in windows and patched
//! where a step changes the scene, so a window's error needs no render of every shape over it.

use std::sync::{Mutex, PoisonError};

use glam::Vec3;

use crate::{Scene, Window};

/// The render and the count of changes of the scene it shows.
#[derive(Default)]
pub(super) struct Canvas(Mutex<Option<(u64, Vec<Vec3>)>>);

impl Canvas {
    /// `scene` rendered over black in `window`, pixel for pixel as [`Scene::render`] of every
    /// shape over that window renders it.
    pub(super) fn shown(&self, scene: &Scene, window: Window) -> Vec<Vec3> {
        let mut kept = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if kept.as_ref().is_none_or(|(at, _)| *at != scene.changes()) {
            let full = Window::full(scene.width, scene.height);
            let pixels = scene.render(full, 0..scene.shapes.len(), [0.; 3]);
            *kept = Some((scene.changes(), pixels));
        }
        let (_, pixels) = kept.as_ref().expect("just rendered");
        window.pixels(scene.width).map(|i| pixels[i]).collect()
    }

    /// `scene` is back as it was at the count of changes `at`.
    pub(super) fn restored(&self, at: u64, scene: &Scene) {
        self.patched(at, scene, Window::full(0, 0), &[]);
    }

    /// `scene` changed since the count of changes `at` only within `window`, which now renders
    /// as `pixels`.
    pub(super) fn patched(&self, at: u64, scene: &Scene, window: Window, pixels: &[Vec3]) {
        let mut kept = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((current, all)) = kept.as_mut()
            && *current == at
        {
            for (i, &p) in window.pixels(scene.width).zip(pixels) {
                all[i] = p;
            }
            *current = scene.changes();
        }
    }
}
