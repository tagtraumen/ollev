use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// Scalar operations shared by `f32` and [`Dual`], so the coverage kernel is written once and used
/// for both the fast forward pass and the derivative pass.
pub trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
{
    fn cst(v: f32) -> Self;
    fn val(self) -> f32;

}

impl Real for f32 {
    fn cst(v: f32) -> Self {
        v
    }
    fn val(self) -> f32 {
        self
    }
}

/// Forward-mode dual number carrying `N` partial derivatives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<const N: usize> {
    pub v: f32,
    pub d: [f32; N],
}

impl<const N: usize> Dual<N> {
    /// The `i`-th input variable.
    pub fn var(v: f32, i: usize) -> Self {
        let mut d = [0.; N];
        d[i] = 1.;
        Self { v, d }
    }

    fn zip(self, rhs: Self, f: impl Fn(f32, f32) -> f32) -> [f32; N] {
        std::array::from_fn(|i| f(self.d[i], rhs.d[i]))
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            v: self.v + rhs.v,
            d: self.zip(rhs, |a, b| a + b),
        }
    }
}

impl<const N: usize> AddAssign for Dual<N> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            v: self.v - rhs.v,
            d: self.zip(rhs, |a, b| a - b),
        }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    #[expect(clippy::suspicious_arithmetic_impl, reason = "product rule")]
    fn mul(self, rhs: Self) -> Self {
        Self {
            v: self.v * rhs.v,
            d: self.zip(rhs, |a, b| a * rhs.v + self.v * b),
        }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let inv = 1. / rhs.v;
        Self {
            v: self.v * inv,
            d: self.zip(rhs, |a, b| (a - self.v * inv * b) * inv),
        }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: self.d.map(|a| -a),
        }
    }
}

impl<const N: usize> Real for Dual<N> {
    fn cst(v: f32) -> Self {
        Self { v, d: [0.; N] }
    }
    fn val(self) -> f32 {
        self.v
    }
}
