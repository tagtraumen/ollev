//! The model fitted to what a host can draw: each feature the host lacks is rewritten in the
//! ones it has, every object at the coarsest form whose render misses the model's by less than
//! a segment's price (the verify loop's rule, judged on the object alone over its box).
//!
//! The rewrites, in the order they run (each may use what the later ones remove):
//! a path gradient as bands along its curve ([`bands`]); a radial focus dropped, or the
//! gradient as nested discs ([`focus`]); a blur, a ramp's alpha or a clip baked into nested
//! shapes traced from the object's coverage ([`layers`]); dashes cut into strokes ([`dashes`]);
//! even-odd loops wound for the nonzero rule ([`nonzero`]).

use crate::model::{Clip, Item, Model, Object};
pub(crate) use memo::Memo;

pub mod bands;
mod dashes;
mod focus;
pub(crate) mod layers;
mod memo;
mod nonzero;
mod render;
#[cfg(feature = "svg")]
pub(crate) use render::ranges;
#[cfg(test)]
mod tests;

/// What a host draws beyond filled and stroked cubic paths painted with a solid colour or a
/// linear or centred radial gradient, at an opacity. Every feature off is lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Fills by the even-odd rule; without, loops are wound for the nonzero rule.
    pub even_odd: bool,
    /// A gradient along a curve ([`crate::model::GradientShape::Path`]).
    pub path_gradients: bool,
    /// A Gaussian blur of an object's coverage ([`Object::blur`]).
    pub blur: bool,
    /// A radial gradient's focus away from its centre.
    pub radial_focus: bool,
    /// Clips ([`Object::clip`], [`Model::clip`]).
    pub clips: bool,
    /// A stroke's dash pattern.
    pub dashes: bool,
    /// Gradient stops with alpha, interpolated in straight colour as SVG does.
    pub transparent_stops: bool,
}

impl Capabilities {
    /// Everything the model uses: SVG, which needs no lowering.
    pub const ALL: Self = Self {
        even_odd: true,
        path_gradients: true,
        blur: true,
        radial_focus: true,
        clips: true,
        dashes: true,
        transparent_stops: true,
    };
    /// Only plain filled and stroked paths.
    pub const NONE: Self = Self {
        even_odd: false,
        path_gradients: false,
        blur: false,
        radial_focus: false,
        clips: false,
        dashes: false,
        transparent_stops: false,
    };
}

impl Default for Capabilities {
    fn default() -> Self {
        Self::ALL
    }
}

/// What the lowering did: objects rewritten per feature, the objects written for them, and
/// those whose best form still misses the model by more than a segment's price.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// Path gradients drawn as bands.
    pub path_gradients: usize,
    /// Radial gradients whose focus was dropped or drawn as discs.
    pub foci: usize,
    /// Objects baked into nested shapes.
    pub layered: usize,
    /// Dashed strokes cut into dashes.
    pub dashed: usize,
    /// Even-odd loops wound for the nonzero rule.
    pub wound: usize,
    /// Objects written in place of the rewritten ones.
    pub written: usize,
    /// Rewritten objects missing by more than a segment's price.
    pub missed: usize,
    /// The largest miss of a rewritten object, in segment prices.
    pub worst: f64,
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{} path gradients, {} foci, {} layered, {} dashed, {} wound; {} objects written, {} missed, worst {:.2}",
            self.path_gradients,
            self.foci,
            self.layered,
            self.dashed,
            self.wound,
            self.written,
            self.missed,
            self.worst
        )
    }
}

/// The model in what `host` draws, and what it took.
#[cfg(any(test, feature = "diagnostics"))]
pub fn lower(model: &Model, host: &Capabilities) -> (Model, Report) {
    let mut report = Report::default();
    let model = lowered(model, host, &mut report, None);
    (model, report)
}

/// The model in what `host` draws, each object lowered once in the life of `memo`; `quick`
/// for a preview, which takes the cheap form of each rewrite an object has not had yet.
pub(crate) fn lower_kept(model: &Model, host: &Capabilities, memo: &mut Memo, quick: bool) -> Model {
    lowered(model, host, &mut Report::default(), Some((memo, quick)))
}

/// The model before its objects are lowered: without clips, the drawing's own clip goes onto
/// every object, so instances are drawn out.
fn prepared(model: &Model, host: &Capabilities) -> Model {
    let mut out = model.clone();
    if !host.clips
        && let Some(top) = out.clip.take()
    {
        out.items = expanded(&out, &out.items);
        out.definitions.clear();
        for c in 0..out.clips.len() {
            if c != top && out.clips[c].parent.is_none() {
                out.clips[c].parent = Some(top);
            }
        }
        clip_all(&mut out.items, top);
    }
    out
}

/// [`lower`], measuring each rewrite's miss for the report where no memo keeps the objects.
fn lowered(
    model: &Model,
    host: &Capabilities,
    report: &mut Report,
    memo: Option<(&mut Memo, bool)>,
) -> Model {
    if *host == Capabilities::ALL {
        return model.clone();
    }
    let mut out = prepared(model, host);
    let mut lowering = Lowering::new(model, host, std::mem::take(&mut out.clips), report);
    lowering.measure = memo.is_none();
    lowering.quick = memo.as_ref().is_some_and(|m| m.1);
    let mut memo = memo.map(|m| m.0);
    if let Some(memo) = memo.as_deref_mut() {
        memo.begin(&lowering);
    }
    out.items = lowering.items(std::mem::take(&mut out.items), &mut memo);
    out.definitions = std::mem::take(&mut out.definitions)
        .into_iter()
        .map(|d| lowering.items(d, &mut memo))
        .collect();
    let mut clips = std::mem::take(&mut lowering.clips);
    if !host.even_odd {
        for c in &mut clips {
            match memo.as_deref_mut() {
                Some(memo) => memo.clip(c, lowering.quick, |c| {
                    crate::progress::finish(nonzero::clip(&lowering, c));
                }),
                None => crate::progress::finish(nonzero::clip(&lowering, c)),
            }
        }
    }
    if let Some(memo) = memo {
        memo.sweep();
    }
    out.clips = clips;
    out
}

/// Lowers into `memo` what [`lower_kept`] would lower of `model`, a checkpoint before each
/// object and clip, so the lowering that follows finds every one of them kept.
pub(crate) async fn keep(model: &Model, host: &Capabilities, memo: &std::sync::Mutex<Memo>) {
    if *host == Capabilities::ALL {
        return;
    }
    let mut out = prepared(model, host);
    let mut report = Report::default();
    let mut lowering = Lowering::new(model, host, std::mem::take(&mut out.clips), &mut report);
    lowering.measure = false;
    let base = lowering.clips.len();
    lock(memo).begin(&lowering);
    let mut objects = Vec::new();
    for items in std::iter::once(&out.items).chain(&out.definitions) {
        objects_of(items, &mut objects);
    }
    for o in objects {
        crate::progress::poll().await;
        let key = memo::key(o, &lowering.clips);
        if lock(memo).touch(&key) {
            continue;
        }
        let lowered = lowering.object(o.clone()).await;
        let added = lowering.clips.split_off(base);
        lock(memo).put(key, base, lowered, added);
    }
    if !host.even_odd {
        for k in 0..base {
            crate::progress::poll().await;
            let mut c = lowering.clips[k].clone();
            lock(memo).clip(&mut c, false, |c| {
                crate::progress::finish(nonzero::clip(&lowering, c));
            });
        }
    }
}

/// The objects among `items`, those in groups included.
fn objects_of<'a>(items: &'a [Item], out: &mut Vec<&'a Object>) {
    for item in items {
        match item {
            Item::Object(o) => out.push(o),
            Item::Group { items, .. } => objects_of(items, out),
            Item::Instance(_) | Item::Repeat { .. } => {}
        }
    }
}

/// The memo, whatever a panic that held it left.
pub(crate) fn lock(memo: &std::sync::Mutex<Memo>) -> std::sync::MutexGuard<'_, Memo> {
    memo.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The items with every instance drawn out as the objects of its definition.
fn expanded(model: &Model, items: &[Item]) -> Vec<Item> {
    let mut out = Vec::new();
    for item in items {
        match item {
            Item::Group { layer, items } => out.push(Item::Group {
                layer: *layer,
                items: expanded(model, items),
            }),
            Item::Object(_) => out.push(item.clone()),
            Item::Instance(_) | Item::Repeat { .. } => {
                let one = Model {
                    items: vec![item.clone()],
                    ..model.clone()
                };
                out.extend(
                    one.drawn()
                        .into_iter()
                        .map(|(_, o)| Item::Object(Box::new(o))),
                );
            }
        }
    }
    out
}

/// Every object drawn under clip `top`: objects without a clip take it, the rest have it as
/// their clips' root.
fn clip_all(items: &mut [Item], top: usize) {
    for item in items {
        match item {
            Item::Object(o) if o.clip.is_none() => o.clip = Some(top),
            Item::Group { items, .. } => clip_all(items, top),
            _ => {}
        }
    }
}

/// The rewrites, in the order they run (see the module's documentation).
#[derive(Clone, Copy)]
enum Rewrite {
    Bands,
    Focus,
    Layers,
    Dashes,
    Nonzero,
}

/// One lowering of a model: the host, the price an object's miss is judged by, and the clips,
/// to which the rewrites add.
struct Lowering<'a> {
    host: Capabilities,
    /// Whether each rewritten object's miss is measured for the report.
    measure: bool,
    /// A preview's lowering: a blur or a focus dropped, bands at the finest level and loops
    /// wound, none of them measured.
    quick: bool,
    width: usize,
    height: usize,
    /// The price of a segment; `None` takes the finest form of each rewrite.
    price: Option<f64>,
    /// The picture's uncertainty per pixel, below which a miss is no error.
    tolerance: Option<std::sync::Arc<Vec<f32>>>,
    clips: Vec<Clip>,
    report: &'a mut Report,
}

impl<'a> Lowering<'a> {
    fn new(model: &Model, host: &Capabilities, clips: Vec<Clip>, report: &'a mut Report) -> Self {
        Lowering {
            host: *host,
            measure: true,
            quick: false,
            width: model.width,
            height: model.height,
            price: model.fidelity.as_ref().map(|f| f.segment_price),
            tolerance: model.fidelity.as_ref().and_then(|f| f.tolerance.clone()),
            clips,
            report,
        }
    }

    fn items(&mut self, items: Vec<Item>, memo: &mut Option<&mut Memo>) -> Vec<Item> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Item::Object(o) => {
                    let lowered = match memo.as_deref_mut() {
                        Some(memo) => memo.object(self, *o, |l, o| {
                            crate::progress::finish(l.object(o))
                        }),
                        None => crate::progress::finish(self.object(*o)),
                    };
                    out.extend(lowered.into_iter().map(|o| Item::Object(Box::new(o))));
                }
                Item::Group { layer, items } => out.push(Item::Group {
                    layer,
                    items: self.items(items, memo),
                }),
                other => out.push(other),
            }
        }
        out
    }

    /// The object in the host's features: each rewrite in turn over what the last one wrote,
    /// a checkpoint before each object a rewrite takes.
    async fn object(&mut self, o: Object) -> Vec<Object> {
        let original = o.clone();
        let mut objects = vec![o];
        let mut changed = false;
        let rewrites = [
            (!self.host.path_gradients, Rewrite::Bands),
            (!self.host.radial_focus, Rewrite::Focus),
            (true, Rewrite::Layers),
            (!self.host.dashes, Rewrite::Dashes),
            (!self.host.even_odd, Rewrite::Nonzero),
        ];
        for (applies, rewrite) in rewrites {
            if applies {
                objects = self.each(objects, &mut changed, rewrite).await;
            }
        }
        if changed && self.measure {
            self.report.written += objects.len();
            let miss = self.miss(std::slice::from_ref(&original), &objects);
            if let Some(price) = self.price {
                self.report.missed += usize::from(miss > price);
                self.report.worst = self.report.worst.max(miss / price.max(f64::MIN_POSITIVE));
            }
        }
        objects
    }

    /// Each object rewritten by `rewrite`, where it rewrites it (`Some`), else kept.
    async fn each(
        &mut self,
        objects: Vec<Object>,
        changed: &mut bool,
        rewrite: Rewrite,
    ) -> Vec<Object> {
        let mut out = Vec::with_capacity(objects.len());
        for o in objects {
            crate::progress::poll().await;
            let rewritten = match rewrite {
                Rewrite::Bands => bands::lower(self, &o),
                Rewrite::Focus => focus::lower(self, &o),
                Rewrite::Layers => layers::lower(self, &o).await,
                Rewrite::Dashes => {
                    let cut = dashes::lower(&o);
                    self.report.dashed += usize::from(cut.is_some());
                    cut
                }
                Rewrite::Nonzero => nonzero::lower(self, &o).await,
            };
            match rewritten {
                Some(rewritten) => {
                    *changed = true;
                    out.extend(rewritten);
                }
                None => out.push(o),
            }
        }
        out
    }

    /// Adds a clip, returning its place.
    fn clip(&mut self, clip: Clip) -> usize {
        self.clips.push(clip);
        self.clips.len() - 1
    }

    /// Of candidate forms of `original`, coarsest first, the first whose miss is within a
    /// segment's price, else the one that missed least (the finest without a price).
    fn coarsest(
        &mut self,
        original: &Object,
        forms: impl IntoIterator<Item = Vec<Object>, IntoIter: DoubleEndedIterator>,
    ) -> Vec<Object> {
        // Without a price the finest form, built alone
        let Some(price) = self.price.filter(|_| !self.quick) else {
            return forms.into_iter().next_back().unwrap_or_default();
        };
        let mut best: Option<(f64, Vec<Object>)> = None;
        for form in forms {
            let miss = self.miss(std::slice::from_ref(original), &form);
            if miss <= price {
                return form;
            }
            if best.as_ref().is_none_or(|b| miss < b.0) {
                best = Some((miss, form));
            }
        }
        best.map(|b| b.1).unwrap_or_default()
    }

    /// How much `form` misses `original` by over the original's box, each drawn alone over
    /// white: the squared colour distance beyond the anti-aliasing pixel and the noise.
    fn miss(&self, original: &[Object], form: &[Object]) -> f64 {
        render::miss(self, original, form)
    }
}
