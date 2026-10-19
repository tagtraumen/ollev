//! The plan: a dynamic program over the region tree that decides every shape before fitting.

use super::*;

/// A node the plan draws, in the color of what it shows, on top of the drawn node `parent`
/// (an index into the plan; the background has none), and whether the plan painted it with a
/// gradient.
pub struct Drawn {
    pub node: u32,
    pub parent: Option<usize>,
    pub color: DVec3,
    pub gradient: bool,
}

/// What the plan paints a node with: one color, a linear gradient, or a linear gradient with
/// stops, fitted to one node and shared by the nodes below it.
#[derive(Clone, Debug)]
pub(super) enum Model {
    Solid(DVec3),
    Linear(LinearFill),
    Radial(RadialFill),
    Ramp(std::sync::Arc<RampFill>),
}

impl Model {
    pub(super) fn at(&self, p: DVec2) -> DVec3 {
        match self {
            Model::Solid(c) => *c,
            Model::Linear(f) => f.at(p),
            Model::Radial(f) => f.at(p),
            Model::Ramp(r) => r.at(p),
        }
    }

    /// Whether the model is a gradient, which costs [`crate::fill::GRADIENT_SEGMENTS`].
    pub(super) fn painted(&self) -> bool {
        !matches!(self, Model::Solid(_))
    }
}

/// The dynamic program over the tree.
pub(super) struct Planner<'a> {
    pub(super) ctx: &'a Context,
    pub(super) hierarchy: &'a Hierarchy,
    /// Segments each node's outline takes: estimated from its perimeter, or counted once traced.
    pub(super) segments: Vec<f64>,
    /// What each node is drawn with if it is: its mean, its gradient, or the model of one of
    /// its parts, and the color that comes to at the node's centroid.
    pub(super) model: Vec<Model>,
    pub(super) color: Vec<DVec3>,
    /// The cost of each node's subtree when the node is drawn: its segments and path, plus its
    /// children at their best against its color.
    pub(super) drawn: Vec<f64>,
    /// Each node's ramp once fitted, which depends on its pixels alone and so holds over the
    /// rounds of planning.
    pub(super) ramps: Vec<std::sync::OnceLock<Option<std::sync::Arc<RampFill>>>>,
    /// Each node's radial as painted once fitted, likewise.
    pub(super) ellipses: Vec<std::sync::OnceLock<Option<std::sync::Arc<RampFill>>>>,
    /// Under the weighted edge term, each node's pixels at their weights, the error's sums
    /// (see [`crate::edges::weighted_stats`]).
    pub(super) weighted: Option<Vec<Stats>>,
    /// Under a tolerance ([`Context::tolerance`]), each node's mean squared tolerance over its
    /// pixels and each leaf's tolerances, so that its error counts beyond them (see
    /// [`Planner::beyond`]).
    pub(super) tolerance: Option<(Vec<f64>, Vec<Option<crate::uncertainty::Spread>>)>,
}

impl<'a> Planner<'a> {
    pub(super) async fn new(ctx: &'a Context, hierarchy: &'a Hierarchy) -> Self {
        let segments = hierarchy
            .nodes
            .iter()
            .map(|n| {
                (n.perimeter as f64 / PIXELS_PER_SEGMENT + n.frame as f64 / FRAME_PER_SEGMENT)
                    .max(MIN_SEGMENTS)
            })
            .collect();
        let color: Vec<DVec3> = hierarchy.nodes.iter().map(|n| n.stats.mean()).collect();
        let tolerance = match &ctx.tolerance {
            Some(t) => {
                let mut leaves = Vec::with_capacity(hierarchy.nodes.len());
                for (n, node) in hierarchy.nodes.iter().enumerate() {
                    if n % 4096 == 0 {
                        crate::progress::poll().await;
                    }
                    leaves.push(node.children.is_none().then(|| {
                        crate::uncertainty::Spread::new(
                            hierarchy
                                .pixels(n as u32)
                                .iter()
                                .map(|&i| t[i as usize])
                                .collect(),
                        )
                    }));
                }
                crate::progress::poll().await;
                Some((mean_squares(hierarchy, t), leaves))
            }
            None => None,
        };
        let model = color.iter().map(|&c| Model::Solid(c)).collect();
        Self {
            ctx,
            hierarchy,
            segments,
            model,
            color,
            drawn: vec![0.; hierarchy.nodes.len()],
            ramps: vec![std::sync::OnceLock::new(); hierarchy.nodes.len()],
            ellipses: vec![std::sync::OnceLock::new(); hierarchy.nodes.len()],
            weighted: None,
            tolerance,
        }
    }

    /// A node's squared error `raw` counted beyond its tolerance: for a leaf (one colour, so
    /// an even residual) its RMS residual beyond each pixel's tolerance, summed (see
    /// [`crate::uncertainty::excess`]); for a larger node, beyond its RMS tolerance.
    pub(super) fn beyond(&self, node: u32, raw: f64) -> f64 {
        let Some((pooled, leaves)) = &self.tolerance else {
            return raw;
        };
        let n = self.err(node).n;
        if n <= 0. {
            return raw;
        }
        let d = (raw / n).max(0.).sqrt();
        match &leaves[node as usize] {
            // the leaf's pixels at the weights the error is summed at, on average
            Some(spread) if !spread.is_empty() => spread.excess(d) * n / spread.len() as f64,
            _ => crate::uncertainty::excess64(raw / n, pooled[node as usize].sqrt()) * n,
        }
    }

    /// The planner with the edge term `term` on the target's edges `field`.
    pub(super) fn with_edges(
        mut self,
        field: &crate::edges::EdgeField,
        term: crate::edges::EdgeTerm,
    ) -> Self {
        self.weighted = Some(crate::edges::weighted_stats(self.hierarchy, field, term.lambda));
        self
    }

    pub(super) fn stats(&self, node: u32) -> &Stats {
        &self.hierarchy.nodes[node as usize].stats
    }

    /// The error of showing `node` painted with `p`: for a color, the pixels' spread plus
    /// their mean's distance from it beyond the slack (see [`Context::saving`]); for a
    /// gradient, the residual about it as it is, which can be less than the spread.
    pub(super) fn shown(&self, node: u32, p: &Model) -> f64 {
        let stats = self.err(node);
        match p {
            Model::Solid(c) => self.beyond(node, stats.sse() + self.ctx.saving(stats, *c)),
            _ => self.sse_about(node, p),
        }
    }

    /// The statistics a node's error is summed over: its pixels, or with the edge term
    /// weighted, its pixels each at its weight (see [`crate::edges::weighted_stats`]). Colors
    /// and gradients are fitted to the pixels as they are.
    pub(super) fn err(&self, node: u32) -> &Stats {
        match &self.weighted {
            Some(w) => &w[node as usize],
            None => self.stats(node),
        }
    }

    /// The error of `node` about a ramp. A ramp's residual is tabled over the pixels as they
    /// are; with the edge term weighted it is taken as spread evenly over the node's pixels,
    /// so it counts at their mean weight (ramps are for shading, where edges are few).
    fn ramp_sse(&self, r: &RampFill, node: u32) -> f64 {
        self.evenly(r.sse(self.hierarchy, node), node)
    }

    /// A squared error `sse` over the pixels of `node` as they are, counted at their mean weight
    /// under the weighted edge term.
    pub(super) fn evenly(&self, sse: f64, node: u32) -> f64 {
        match &self.weighted {
            Some(w) => sse * w[node as usize].n / self.stats(node).n.max(1.),
            None => sse,
        }
    }

    /// Squared error of the pixels of `node` around `p`, beyond the tolerance where there is
    /// one (see [`Planner::beyond`]).
    pub(super) fn sse_about(&self, node: u32, p: &Model) -> f64 {
        self.beyond(node, self.raw_about(node, p))
    }

    /// Squared error of the pixels of `node` around `p`, as it is.
    fn raw_about(&self, node: u32, p: &Model) -> f64 {
        let stats = self.err(node);
        match p {
            Model::Solid(c) => stats.sse_about(*c),
            Model::Linear(f) => stats.sse_about_linear(f),
            Model::Radial(f) => stats.sse_about_radial(f),
            Model::Ramp(r) => self.ramp_sse(r, node),
        }
    }

    /// [`Planner::shown`] for `node`, a solid colour's saving weighted by its salience: its
    /// contrast against the spread of its surroundings, clamped to 0.5 to 2.
    fn shown_node(&self, node: u32, p: &Model) -> f64 {
        let stats = self.err(node);
        let Model::Solid(color) = *p else {
            return self.shown(node, p);
        };
        // The region this one merged with is its surroundings; its spread per pixel is what a
        // difference there has to exceed to be seen
        let n = &self.hierarchy.nodes[node as usize];
        let surroundings = n
            .parent
            .and_then(|q| self.hierarchy.nodes[q as usize].children)
            .map(|[a, b]| if a == node { b } else { a })
            .map_or(*self.stats(node), |s| *self.stats(s));
        let spread = (surroundings.sse() / surroundings.n.max(1.)).sqrt();
        let salience =
            (self.stats(node).mean().distance(color) / spread.max(SALIENCE_FLOOR)).clamp(0.5, 2.);
        self.beyond(node, stats.sse() + self.ctx.saving(stats, color) * salience)
    }

    /// The cost of a node's subtree when the node itself is not drawn and shows `p`.
    pub(super) fn skipped(&self, node: u32, p: &Model) -> f64 {
        let Some([a, b]) = self.hierarchy.nodes[node as usize].children else {
            return self.shown_node(node, p);
        };
        // Nothing drawn inside can save more than the error there, so once that is below what
        // the smallest shape costs, nothing inside is drawn
        let least = self.ctx.accept_path_cost() + MIN_SEGMENTS * self.ctx.accept_cost();
        // (the raw error bounds what drawing inside can save beyond the tolerance too, while
        // the node's tolerance pooled over a large flat field could hide a dot's residual)
        if self.raw_about(node, p) <= least {
            return self.shown_node(node, p);
        }
        self.parts(a, b, p)
    }

    /// The cost of a node's subtree showing `p` where the node is not drawn, drawn or not.
    pub(super) fn best(&self, node: u32, p: &Model) -> f64 {
        self.drawn[node as usize].min(self.skipped(node, p))
    }

    /// The cost of the parts `a` and `b` of a node where both show `p` unless drawn.
    pub(super) fn parts(&self, a: u32, b: u32, p: &Model) -> f64 {
        self.best(a, p) + self.best(b, p)
    }

    /// [`Planner::parts`] where `part` is the node's body, drawn as the node in the part's
    /// model `p`.
    fn body_parts(&self, part: u32, other: u32, p: &Model) -> f64 {
        self.skipped(part, p) + self.best(other, p)
    }

    /// The cost of drawing `node` with its subtree at its best, and its model: its mean, a
    /// gradient, a ramp, or a part's model (that part then its body). A gradient is offered only
    /// where it hides no sharp-edged leaf, a ramp only where it can beat the cheapest option.
    pub(super) fn draw(&self, node: u32) -> (f64, Model) {
        let segments = self.segments[node as usize];
        let own = self.ctx.accept_path_cost() + segments * self.ctx.accept_cost();
        let gradient = crate::fill::GRADIENT_SEGMENTS * self.ctx.segment_cost;
        let stop = crate::fill::STOP_SEGMENTS * self.ctx.segment_cost;
        let least = self.ctx.accept_path_cost() + MIN_SEGMENTS * self.ctx.accept_cost();
        let stats = self.stats(node);
        let mean = Model::Solid(stats.mean());
        let children = self.hierarchy.nodes[node as usize].children;
        // Whether a gradient hides `leaf`, which shows `under` of error beneath it
        let hides = |leaf: u32, under: f64| {
            let (s, n) = (self.err(leaf), &self.hierarchy.nodes[leaf as usize]);
            let hidden =
                under - self.beyond(leaf, s.sse()) > least && self.drawn[leaf as usize] >= under;
            hidden && n.blur < SOFT_EDGE
        };
        let leaves = self.hierarchy.leaves(node);
        let fitted = stats.linear_fill();
        let smooth = |f: &LinearFill| {
            leaves
                .iter()
                .all(|&leaf| !hides(leaf, self.beyond(leaf, self.err(leaf).sse_about_linear(f))))
        };
        let linear = fitted.filter(smooth).map(Model::Linear);
        let radial = stats
            .radial_fill()
            .filter(|f| {
                let under = |leaf: u32| self.beyond(leaf, self.err(leaf).sse_about_radial(f));
                // Like a ramp's stops, the radial shading must pay within the leaves, since its
                // curvature can fit flat regions of different colours
                let shading: f64 = leaves
                    .iter()
                    .map(|&leaf| self.beyond(leaf, self.err(leaf).sse()) - under(leaf))
                    .sum();
                shading > gradient && leaves.iter().all(|&leaf| !hides(leaf, under(leaf)))
            })
            .map(Model::Radial);
        // The options' costs, the node's own outline aside
        let mut options = match children {
            None => {
                let mut options = vec![(self.beyond(node, self.err(node).sse()), mean)];
                for m in [&linear, &radial].into_iter().flatten() {
                    options.push((gradient + self.sse_about(node, m), m.clone()));
                }
                options
            }
            Some([a, b]) => {
                // A part's gradient taken over costs the gradient, like the node's own
                let (model_a, model_b) = (&self.model[a as usize], &self.model[b as usize]);
                let paint = |m: &Model| if m.painted() { gradient } else { 0. };
                let mut options = vec![(self.parts(a, b, &mean), mean)];
                for (part, other, model) in [(a, b, model_a), (b, a, model_b)] {
                    if !matches!(model, Model::Ramp(_)) {
                        options.push((
                            paint(model) + self.body_parts(part, other, model),
                            model.clone(),
                        ));
                    }
                }
                for m in [&linear, &radial].into_iter().flatten() {
                    options.push((gradient + self.parts(a, b, m), m.clone()));
                }
                options
            }
        };
        let cheapest = options.iter().map(|o| o.0).fold(f64::INFINITY, f64::min);
        if let Some(f) = fitted
            && gradient + stop < cheapest
            && let Some(ramp) = self.ramps[node as usize]
                .get_or_init(|| {
                    RampFill::fit(self.hierarchy, node, &f, stop).map(std::sync::Arc::new)
                })
                .clone()
        {
            let smooth = leaves
                .iter()
                .all(|&leaf| !hides(leaf, self.beyond(leaf, self.ramp_sse(&ramp, leaf))));
            // The stops must pay on the shading within the leaves, not on the steps between
            // them, which are edges for shapes to draw
            let stops = stop * ramp.gradient.stops().len() as f64;
            let shading: f64 = leaves
                .iter()
                .map(|&leaf| {
                    self.beyond(leaf, self.err(leaf).sse())
                        - self.beyond(leaf, self.ramp_sse(&ramp, leaf))
                })
                .sum();
            if smooth && shading > stops {
                let paint = gradient + stops;
                let m = Model::Ramp(ramp);
                let rest = match children {
                    None => self.sse_about(node, &m),
                    Some([a, b]) => self.parts(a, b, &m),
                };
                options.push((paint + rest, m));
            }
        }
        // Where the circle shades the node, the radial as painted (linear in the radius,
        // stretched into an ellipse where that fits), its shading paying within the leaves as
        // the circle's does
        let cheapest = options.iter().map(|o| o.0).fold(f64::INFINITY, f64::min);
        if let Some(Model::Radial(f)) = &radial
            && gradient < cheapest
            && let Some(ellipse) = self.ellipses[node as usize]
                .get_or_init(|| {
                    RampFill::fit_radial(self.hierarchy, node, f, self.ctx.segment_cost)
                        .map(std::sync::Arc::new)
                })
                .clone()
        {
            let under = |leaf: u32| self.beyond(leaf, self.ramp_sse(&ellipse, leaf));
            let shape = crate::fill::cost_segments(&crate::Fill {
                color: ellipse.from,
                gradient: Some(ellipse.gradient),
                opacity: 1.,
            });
            let paint = shape * self.ctx.segment_cost;
            // Flat and shaded compared at the same weights, as the ramp's error is spread evenly
            let flat = |leaf: u32| self.beyond(leaf, self.evenly(self.stats(leaf).sse(), leaf));
            let shading: f64 = leaves.iter().map(|&leaf| flat(leaf) - under(leaf)).sum();
            if shading > paint && leaves.iter().all(|&leaf| !hides(leaf, under(leaf))) {
                let m = Model::Ramp(ellipse);
                let rest = match children {
                    None => self.sse_about(node, &m),
                    Some([a, b]) => self.parts(a, b, &m),
                };
                options.push((paint + rest, m));
            }
        }
        let (cost, model) = options
            .into_iter()
            .min_by(|x, y| x.0.total_cmp(&y.0))
            .expect("an option");
        (own + cost, model)
    }

    /// Chooses the nodes to draw: the `background` always, and every node cheaper drawn than
    /// left to its nearest drawn ancestor, each in the mean colour of what it shows. Parents
    /// come before their children, larger siblings first.
    pub(super) async fn plan(&mut self, background: u32) -> Vec<Drawn> {
        let root = self.hierarchy.root;
        let mut plan = Vec::new();
        {
            // Children come before their parents in the node order
            for node in 0..self.hierarchy.nodes.len() as u32 {
                crate::progress::poll().await;
                let (cost, model) = self.draw(node);
                self.drawn[node as usize] = cost;
                self.color[node as usize] = model.at(self.stats(node).centroid());
                self.model[node as usize] = model;
            }
            // A whole image painted with one ramp is its own ground: the root replaces the
            // background where its ramp makes neither part worth drawing and it costs less than
            // the background with the other part
            let model = &self.model[root as usize];
            let background = match self.hierarchy.nodes[root as usize].children {
                Some([a, b])
                    if background != root
                        && matches!(model, Model::Ramp(_))
                        && [a, b]
                            .iter()
                            .all(|&c| self.drawn[c as usize] >= self.skipped(c, model)) =>
                {
                    let other = if a == background { b } else { a };
                    let parted = self.drawn[background as usize]
                        + self.best(other, &self.model[background as usize]);
                    if self.drawn[root as usize] < parted {
                        root
                    } else {
                        background
                    }
                }
                _ => background,
            };
            plan.push(Drawn {
                node: background,
                parent: None,
                color: self.color[background as usize],
                gradient: self.model[background as usize].painted(),
            });
            // (node, what shows where it is not drawn, nearest drawn ancestor in the plan)
            let mut stack: Vec<(u32, Model, usize)> = Vec::new();
            let push_children = |stack: &mut Vec<_>, node: u32, p: Model, parent: usize| {
                if let Some([a, b]) = self.hierarchy.nodes[node as usize].children {
                    // The larger part is pushed last, so it is popped and drawn first
                    let (first, second) = if self.stats(a).n >= self.stats(b).n {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    stack.push((second, p.clone(), parent));
                    stack.push((first, p, parent));
                }
            };
            // The background's parts and whatever else the image holds all lie on the background
            let p = self.model[background as usize].clone();
            push_children(&mut stack, background, p.clone(), 0);
            if background != root {
                for child in self.hierarchy.nodes[root as usize]
                    .children
                    .into_iter()
                    .flatten()
                {
                    if child != background {
                        stack.push((child, p.clone(), 0));
                    }
                }
            }
            while let Some((node, p, parent)) = stack.pop() {
                crate::progress::poll().await;
                let chosen = self.drawn[node as usize] < self.skipped(node, &p);
                if chosen {
                    plan.push(Drawn {
                        node,
                        parent: Some(parent),
                        color: self.color[node as usize],
                        gradient: self.model[node as usize].painted(),
                    });
                    let k = plan.len() - 1;
                    push_children(&mut stack, node, self.model[node as usize].clone(), k);
                } else {
                    push_children(&mut stack, node, p, parent);
                }
            }
        }
        // Each drawn node shows its pixels less those of the drawn nodes on it (the rest of the
        // image lies on the background without being part of it)
        let mut shown: Vec<Stats> = plan.iter().map(|d| *self.stats(d.node)).collect();
        for k in (1..plan.len()).rev() {
            let parent = plan[k].parent.expect("only the background has no parent");
            if self.hierarchy.within(plan[k].node, plan[parent].node) {
                shown[parent] = shown[parent].sub(self.stats(plan[k].node));
            }
        }
        for (d, s) in plan.iter_mut().zip(&shown) {
            if s.n >= 0.5 {
                d.color = s.mean();
            }
        }

        plan
    }
}

/// Each node's mean of the squared tolerance `t` over its pixels, from the leaves up.
fn mean_squares(hierarchy: &Hierarchy, t: &[f32]) -> Vec<f64> {
    let mut sums = vec![0f64; hierarchy.nodes.len()];
    for (i, &t) in t.iter().enumerate() {
        sums[hierarchy.leaf(i) as usize] += (t * t) as f64;
    }
    // Children come before their parents in the node order
    for node in 0..hierarchy.nodes.len() {
        if let Some([a, b]) = hierarchy.nodes[node].children {
            sums[node] = sums[a as usize] + sums[b as usize];
        }
    }
    sums.iter()
        .zip(&hierarchy.nodes)
        .map(|(s, n)| s / n.stats.n.max(1.))
        .collect()
}

/// The fill of every drawn node, fitted to the pixels it shows: its own less those of the
/// drawn nodes on it, each pixel going to the innermost drawn node holding it, and less the
/// [`crate::fill::RIM`] along other nodes; see [`crate::fill::fit`].
pub(super) async fn fit_fills(ctx: &Context, hierarchy: &Hierarchy, plan: &[Drawn]) -> Vec<Fill> {
    let drawn: HashMap<u32, usize> = plan.iter().enumerate().map(|(k, d)| (d.node, k)).collect();
    let mut owner: HashMap<u32, usize> = HashMap::default();
    let (width, height) = (ctx.width, ctx.height);
    let shown: Vec<usize> = (0..width * height)
        .map(|i| {
            let leaf = hierarchy.leaf(i);
            *owner.entry(leaf).or_insert_with(|| {
                let mut node = Some(leaf);
                while let Some(n) = node {
                    if let Some(&k) = drawn.get(&n) {
                        return k;
                    }
                    node = hierarchy.nodes[n as usize].parent;
                }
                0
            })
        })
        .collect();
    // How many steps each pixel lies from another node's, up to the rim: the boundary pixels
    // are one, their neighbors two, and so on
    let rim = crate::fill::RIM;
    let mut depth: Vec<usize> = (0..width * height)
        .map(|i| {
            let (x, y, k) = (i % width, i / width, shown[i]);
            let boundary = (x > 0 && shown[i - 1] != k)
                || (x + 1 < width && shown[i + 1] != k)
                || (y > 0 && shown[i - width] != k)
                || (y + 1 < height && shown[i + width] != k);
            if boundary { 1 } else { rim + 1 }
        })
        .collect();
    for d in 1..rim {
        let inner = depth.clone();
        for i in 0..width * height {
            let (x, y) = (i % width, i / width);
            let next = (x > 0 && inner[i - 1] == d)
                || (x + 1 < width && inner[i + 1] == d)
                || (y > 0 && inner[i - width] == d)
                || (y + 1 < height && inner[i + width] == d);
            if next && inner[i] > d {
                depth[i] = d + 1;
            }
        }
    }
    let mut counts = vec![[0usize; crate::fill::RIM + 2]; plan.len()];
    for i in 0..width * height {
        counts[shown[i]][depth[i]] += 1;
    }
    let erosion: Vec<usize> = counts.iter().map(|c| crate::fill::erosion(c)).collect();
    let mut samples: Vec<Vec<(Vec2, Vec3)>> = vec![Vec::new(); plan.len()];
    let mut tolerances: Vec<Vec<f32>> = vec![Vec::new(); plan.len()];
    let mut extents: Vec<Vec<Vec2>> = vec![Vec::new(); plan.len()];
    for i in 0..width * height {
        let k = shown[i];
        let p = Vec2::new((i % width) as f32 + 0.5, (i / width) as f32 + 0.5);
        extents[k].push(p);
        if depth[i] > erosion[k] {
            samples[k].push((p, ctx.target[i]));
            if ctx.tolerance.is_some() {
                tolerances[k].push(ctx.tolerance_at(i));
            }
        }
    }
    let costs = crate::fill::Costs::at(ctx.segment_cost);
    let mut fills = Vec::with_capacity(plan.len());
    for (((d, samples), extent), tolerance) in plan.iter().zip(&samples).zip(&extents).zip(&tolerances)
    {
        crate::progress::poll().await;
        // A node the plan painted with a gradient takes any gradient that helps: the plan
        // left the detail the gradient explains undrawn. Its stops still pay their way
        let costs = if d.gradient {
            crate::fill::Costs {
                gradient: 0.,
                ..costs
            }
        } else {
            costs
        };
        fills.push(crate::fill::fit(samples, tolerance, ctx.rgb(d.color), None, costs, extent).0);
    }
    fills
}

/// Pixel weights over `window` for fitting a shape in `color`: zero on the leaves of the
/// `masked` parts (each with its color) that look like the part rather than the shape, and next
/// to them.
pub(super) fn detail_weights(
    ctx: &Context,
    hierarchy: &Hierarchy,
    masked: &[(u32, DVec3)],
    color: DVec3,
    window: Window,
) -> Weights {
    let mut leaves = crate::HashSet::default();
    for &(part, part_color) in masked {
        for &leaf in hierarchy.leaves(part) {
            let leaf_color = hierarchy.nodes[leaf as usize].stats.mean();
            if leaf_color.distance(part_color) < leaf_color.distance(color) {
                leaves.insert(leaf);
            }
        }
    }
    let mut weights = Weights::masking(window, ctx.width, |i| leaves.contains(&hierarchy.leaf(i)));
    // The context's free pixels (see [`Context::free`]) are left out too, but not grown as the
    // masked detail is, which would free the gaps between dense line art whole
    if let Some(free) = ctx.free.as_deref() {
        for (w, i) in weights.values.iter_mut().zip(window.pixels(ctx.width)) {
            if free[i] {
                *w = 0.;
            }
        }
    }
    weights
}

/// How much the pixels of shapes drawn later count in a fit: next to nothing, so the shape does
/// not reach for pixels another shape will cover, but not nothing, so its outline is not free
/// there in case that shape never pays.
const LATER_WEIGHT: f32 = 0.01;

/// [`detail_weights`] with the pixels of the `later` shapes' regions at [`LATER_WEIGHT`].
pub(super) fn plan_weights(
    ctx: &Context,
    hierarchy: &Hierarchy,
    masked: &[(u32, DVec3)],
    later: &[u32],
    color: DVec3,
    window: Window,
) -> Weights {
    let mut weights = detail_weights(ctx, hierarchy, masked, color, window);
    let leaves: crate::HashSet<u32> = later
        .iter()
        .flat_map(|&part| hierarchy.leaves(part))
        .copied()
        .collect();
    for (w, i) in weights.values.iter_mut().zip(window.pixels(ctx.width)) {
        if leaves.contains(&hierarchy.leaf(i)) {
            *w = w.min(LATER_WEIGHT);
        }
    }
    weights
}
