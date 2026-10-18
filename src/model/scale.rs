//! A model traced at a smaller size, scaled back to the picture's: every coordinate and length
//! by one factor, the fidelity's per-pixel tolerance spread over the larger pixels.

use std::sync::Arc;

use super::{Clip, Fidelity, Item, Model, Object, Paint, Structure};

impl Model {
    /// The model scaled by `s` about the origin onto a frame of `width` by `height`.
    pub fn scaled(&self, s: f64, width: usize, height: usize) -> Model {
        let object = |o: &Object| Object {
            geometry: o.geometry.scaled(s),
            paint: match &o.paint {
                Paint::Gradient(g) => Paint::Gradient(g.mapped(&[s, 0., 0., s, 0., 0.])),
                solid => solid.clone(),
            },
            blur: o.blur * s,
            ..o.clone()
        };
        fn items(list: &[Item], s: f64, object: &dyn Fn(&Object) -> Object) -> Vec<Item> {
            let instance = |i: &super::Instance| super::Instance {
                transform: {
                    let t = i.transform;
                    [t[0], t[1], t[2], t[3], t[4] * s, t[5] * s]
                },
                ..i.clone()
            };
            list.iter()
                .map(|item| match item {
                    Item::Object(o) => Item::Object(Box::new(object(o))),
                    Item::Group { layer, items: g } => Item::Group {
                        layer: *layer,
                        items: items(g, s, object),
                    },
                    Item::Instance(i) => Item::Instance(instance(i)),
                    Item::Repeat {
                        structure,
                        instances,
                    } => Item::Repeat {
                        structure: match *structure {
                            Structure::List => Structure::List,
                            Structure::Linear { step } => Structure::Linear { step: step * s },
                            Structure::Grid { a, b, columns } => Structure::Grid {
                                a: a * s,
                                b: b * s,
                                columns,
                            },
                            Structure::Circular { center, step } => Structure::Circular {
                                center: center * s,
                                step,
                            },
                        },
                        instances: instances.iter().map(instance).collect(),
                    },
                })
                .collect()
        }
        // Each pixel takes the tolerance of the traced pixel it lies in; a segment's price
        // grows with the area its error is summed over
        let (w0, h0) = (self.width, self.height);
        let fidelity = self.fidelity.as_ref().map(|f| Fidelity {
            tolerance: f.tolerance.as_ref().map(|t| {
                let at =
                    |v: usize, n: usize| ((v as f64 + 0.5) / s).floor().min(n as f64 - 1.) as usize;
                Arc::new(
                    (0..width * height)
                        .map(|i| t[at(i / width, h0) * w0 + at(i % width, w0)])
                        .collect(),
                )
            }),
            segment_price: f.segment_price * s * s,
        });
        Model {
            width,
            height,
            items: items(&self.items, s, &object),
            clips: self
                .clips
                .iter()
                .map(|c| Clip {
                    loops: c
                        .loops
                        .iter()
                        .map(|l| l.iter().map(|q| q.map(|p| p * s)).collect())
                        .collect(),
                    ..c.clone()
                })
                .collect(),
            definitions: self
                .definitions
                .iter()
                .map(|d| items(d, s, &object))
                .collect(),
            fidelity,
            clip: self.clip,
        }
    }
}
