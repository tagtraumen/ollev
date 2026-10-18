#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Point,
    Color,
}

pub struct Adam {
    pub lr_points: f32,
    pub lr_colors: f32,
    pub beta1: f32,
    pub beta2: f32,
    /// Gradients well below this only take proportionally small steps. Adam's usual `1e-8` would
    /// turn float noise near the optimum into full-size steps.
    pub eps: f32,
    /// Multiplies both learning rates, for schedules.
    pub lr_scale: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    /// Steps taken per parameter, for the bias correction of parameters added later
    t: Vec<i32>,
}

impl Adam {
    pub fn new(lr_points: f32, lr_colors: f32) -> Self {
        Self {
            lr_points,
            lr_colors,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-2,
            lr_scale: 1.,
            m: Vec::new(),
            v: Vec::new(),
            t: Vec::new(),
        }
    }

    /// Steps a flat parameter vector, for callers that tie parameters together themselves.
    pub fn step_flat(&mut self, params: &mut [f32], kinds: &[ParamKind], grad: &[f32]) {
        for (i, (param, &kind)) in params.iter_mut().zip(kinds).enumerate() {
            self.update(i, kind, param, grad[i]);
        }
    }

    fn update(&mut self, i: usize, kind: ParamKind, param: &mut f32, g: f32) {
        // New parameters start with fresh moments
        if i >= self.m.len() {
            self.m.resize(i + 1, 0.);
            self.v.resize(i + 1, 0.);
            self.t.resize(i + 1, 0);
        }
        let (m, v) = (&mut self.m[i], &mut self.v[i]);
        self.t[i] += 1;
        let correction1 = 1. - self.beta1.powi(self.t[i]);
        let correction2 = 1. - self.beta2.powi(self.t[i]);
        *m = self.beta1 * *m + (1. - self.beta1) * g;
        *v = self.beta2 * *v + (1. - self.beta2) * g * g;
        let step = (*m / correction1) / ((*v / correction2).sqrt() + self.eps);
        match kind {
            ParamKind::Point => *param -= self.lr_scale * self.lr_points * step,
            ParamKind::Color => {
                *param = (*param - self.lr_scale * self.lr_colors * step).clamp(0., 1.);
            }
        }
    }
}
