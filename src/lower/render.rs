//! How much a rewrite misses the object it replaces, by the verify loop's measure (see the
//! `emit` crate): the squared colour distance beyond the range the original takes over a
//! pixel's 3 x 3 neighbourhood (an anti-aliasing pixel of position) and beyond the noise.

use glam::Vec3;

use super::Lowering;
use crate::model::{Item, Model, Object};
use crate::translucency::raster::NOISE;

/// The miss of `form` against `original`, each drawn alone over white with the lowering's
/// clips, over the pixels either paints.
pub(super) fn miss(l: &Lowering, original: &[Object], form: &[Object]) -> f64 {
    let (w, h) = (l.width, l.height);
    let mut window: Option<[usize; 4]> = None;
    for o in original.iter().chain(form) {
        let r = o.painted();
        let (x1, y1) = ((r.x0 + r.w + 1).min(w), (r.y0 + r.h + 1).min(h));
        let (x0, y0) = (
            r.x0.saturating_sub(1).min(x1),
            r.y0.saturating_sub(1).min(y1),
        );
        window = Some(match window {
            None => [x0, y0, x1, y1],
            Some([a, b, c, d]) => [a.min(x0), b.min(y0), c.max(x1), d.max(y1)],
        });
    }
    let Some([x0, y0, x1, y1]) = window else {
        return 0.;
    };
    let (ww, wh) = (x1 - x0, y1 - y0);
    if ww == 0 || wh == 0 {
        return 0.;
    }
    let model = |objects: &[Object]| Model {
        width: w,
        height: h,
        items: objects
            .iter()
            .map(|o| Item::Object(Box::new(o.clone())))
            .collect(),
        clips: l.clips.clone(),
        definitions: Vec::new(),
        fidelity: None,
        clip: None,
    };
    let window = [x0, y0, ww, wh];
    let reference = model(original).render_window(window);
    let emitted = model(form).render_window(window);
    let floor = NOISE as f32;
    let mut miss = 0f64;
    for y in 0..wh {
        for x in 0..ww {
            let (mut lo, mut hi) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
            for yy in y.saturating_sub(1)..(y + 2).min(wh) {
                for xx in x.saturating_sub(1)..(x + 2).min(ww) {
                    let v = reference[yy * ww + xx];
                    (lo, hi) = (lo.min(v), hi.max(v));
                }
            }
            let e = emitted[y * ww + x];
            let tolerance = l
                .tolerance
                .as_ref()
                .and_then(|t| t.get((y0 + y) * w + x0 + x))
                .map_or(floor, |&t| t.max(floor));
            let over = ((e.clamp(lo, hi) - e).length() - tolerance).max(0.) as f64;
            miss += over * over;
        }
    }
    miss
}
