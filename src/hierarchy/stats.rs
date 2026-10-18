//! A region's colour sums and the fills fitted from them.

use super::*;

impl Stats {
    pub fn pixel(c: DVec3) -> Self {
        Self {
            n: 1.,
            sum: c,
            squares: c.length_squared(),
            products: [
                c.x * c.x,
                c.x * c.y,
                c.x * c.z,
                c.y * c.y,
                c.y * c.z,
                c.z * c.z,
            ],
            ..Self::default()
        }
    }

    /// The pixel with center `p`.
    pub fn pixel_at(c: DVec3, p: DVec2) -> Self {
        let r2 = p.length_squared();
        Self {
            positions: p,
            position_products: [p.x * p.x, p.x * p.y, p.y * p.y],
            color_positions: [c * p.x, c * p.y],
            radial_moments: [r2 * p.x, r2 * p.y, r2 * r2],
            color_radial: c * r2,
            ..Self::pixel(c)
        }
    }

    pub fn add(&self, o: &Self) -> Self {
        Self {
            n: self.n + o.n,
            sum: self.sum + o.sum,
            squares: self.squares + o.squares,
            products: std::array::from_fn(|k| self.products[k] + o.products[k]),
            positions: self.positions + o.positions,
            position_products: std::array::from_fn(|k| {
                self.position_products[k] + o.position_products[k]
            }),
            color_positions: [
                self.color_positions[0] + o.color_positions[0],
                self.color_positions[1] + o.color_positions[1],
            ],
            radial_moments: std::array::from_fn(|k| self.radial_moments[k] + o.radial_moments[k]),
            color_radial: self.color_radial + o.color_radial,
        }
    }

    /// These pixels each counted `w` times, for sums of a weighted error.
    pub fn scaled(&self, w: f64) -> Self {
        Self {
            n: self.n * w,
            sum: self.sum * w,
            squares: self.squares * w,
            products: self.products.map(|p| p * w),
            positions: self.positions * w,
            position_products: self.position_products.map(|p| p * w),
            color_positions: self.color_positions.map(|p| p * w),
            radial_moments: self.radial_moments.map(|p| p * w),
            color_radial: self.color_radial * w,
        }
    }

    pub fn sub(&self, o: &Self) -> Self {
        Self {
            n: self.n - o.n,
            sum: self.sum - o.sum,
            squares: self.squares - o.squares,
            products: std::array::from_fn(|k| self.products[k] - o.products[k]),
            positions: self.positions - o.positions,
            position_products: std::array::from_fn(|k| {
                self.position_products[k] - o.position_products[k]
            }),
            color_positions: [
                self.color_positions[0] - o.color_positions[0],
                self.color_positions[1] - o.color_positions[1],
            ],
            radial_moments: std::array::from_fn(|k| self.radial_moments[k] - o.radial_moments[k]),
            color_radial: self.color_radial - o.color_radial,
        }
    }

    /// The mean position of the pixels.
    pub fn centroid(&self) -> DVec2 {
        if self.n < 0.5 {
            DVec2::ZERO
        } else {
            self.positions / self.n
        }
    }

    /// The least squares linear gradient: the affine color model over the plane reduced to the
    /// direction it changes most along. None for too few pixels, pixels on a line, or colors
    /// that do not vary with the position.
    pub fn linear_fill(&self) -> Option<LinearFill> {
        if self.n < 4. {
            return None;
        }
        let (n, pc, m) = (self.n, self.centroid(), self.mean());
        // Position scatter and color-position cross products about the centroid and mean
        let [xx, xy, yy] = self.position_products;
        let (sxx, sxy, syy) = (
            xx - n * pc.x * pc.x,
            xy - n * pc.x * pc.y,
            yy - n * pc.y * pc.y,
        );
        let tx = self.color_positions[0] - m * self.positions.x;
        let ty = self.color_positions[1] - m * self.positions.y;
        let det = sxx * syy - sxy * sxy;
        if det <= 1e-9 * (sxx + syy).powi(2).max(1e-12) {
            return None;
        }
        let gx = (tx * syy - ty * sxy) / det;
        let gy = (ty * sxx - tx * sxy) / det;
        // The direction the colors change most along: the top eigenvector of G^T G
        let (a, b, c) = (gx.dot(gx), gx.dot(gy), gy.dot(gy));
        let axis = if b.abs() < 1e-15 {
            if a >= c { DVec2::X } else { DVec2::Y }
        } else {
            let top = 0.5 * (a + c) + (0.25 * (a - c).powi(2) + b * b).sqrt();
            DVec2::new(b, top - a).normalize()
        };
        let slope = gx * axis.x + gy * axis.y;
        if slope.length_squared() < 1e-15 {
            return None;
        }
        Some(LinearFill {
            origin: pc,
            axis,
            color: m,
            slope,
        })
    }

    /// Squared error of the pixels around `fill`.
    pub fn sse_about_linear(&self, fill: &LinearFill) -> f64 {
        let (d, m, s) = (fill.axis, fill.color, fill.slope);
        let shift = fill.origin.dot(d);
        // Sums over the pixels of q = (p - origin) . axis, of q^2 and of c q
        let pd = self.positions.dot(d);
        let sum_q = pd - self.n * shift;
        let [xx, xy, yy] = self.position_products;
        let sum_pd2 = d.x * d.x * xx + 2. * d.x * d.y * xy + d.y * d.y * yy;
        let sum_q2 = sum_pd2 - 2. * shift * pd + self.n * shift * shift;
        let sum_cq =
            self.color_positions[0] * d.x + self.color_positions[1] * d.y - self.sum * shift;
        (self.squares - 2. * m.dot(self.sum) - 2. * s.dot(sum_cq)
            + self.n * m.length_squared()
            + 2. * m.dot(s) * sum_q
            + s.length_squared() * sum_q2)
            .max(0.)
    }

    /// Sums over the pixels of `s = |p - center|^2`, of `s^2`, and of each color times `s`.
    pub(super) fn radial_sums(&self, center: DVec2) -> (f64, f64, DVec3) {
        let q = center;
        let [xx, xy, yy] = self.position_products;
        let [rx, ry, r4] = self.radial_moments;
        let r2 = xx + yy;
        let qq = q.length_squared();
        let qp = q.dot(self.positions);
        let sum_s = r2 - 2. * qp + self.n * qq;
        // (r^2 - 2 q.p + |q|^2)^2, term by term
        let sum_qp2 = q.x * q.x * xx + 2. * q.x * q.y * xy + q.y * q.y * yy;
        let sum_s2 = r4 - 4. * (q.x * rx + q.y * ry) + 4. * sum_qp2 + 2. * qq * r2 - 4. * qq * qp
            + self.n * qq * qq;
        let sum_cs = self.color_radial
            - 2. * (self.color_positions[0] * q.x + self.color_positions[1] * q.y)
            + self.sum * qq;
        (sum_s, sum_s2, sum_cs)
    }

    /// The least squares radial fill: colors quadratic in the distance from the curvature-weighted
    /// extremum of each channel's model `a + b . p + c |p|^2`. None for too few pixels, no
    /// curvature, or a center beyond the pixels' extent (there the linear fill is the one).
    pub fn radial_fill(&self) -> Option<RadialFill> {
        if self.n < 6. {
            return None;
        }
        let (n, pc) = (self.n, self.centroid());
        // The normal equations over the basis 1, x, y, r^2, about the centroid for their
        // conditioning: x' = x - pc.x and so on, with r'^2 = |p - pc|^2
        let [xx, xy, yy] = self.position_products;
        let (sxx, sxy, syy) = (
            xx - n * pc.x * pc.x,
            xy - n * pc.x * pc.y,
            yy - n * pc.y * pc.y,
        );
        let (s_r, s_r2, _) = self.radial_sums(pc);
        // Sums of x' r'^2 and y' r'^2: the radial sums about pc, moved
        let [rx, ry, _] = self.radial_moments;
        // The sum of (p_a - pc_a) |p - pc|^2 along axis a, expanded: sum p_a r^2
        // - 2 sum p_a (pc . p) + |pc|^2 sum p_a - pc_a sum |p - pc|^2
        let shifted = |r: f64, at: f64, sum: f64, cross: [f64; 2]| {
            r - 2. * (pc.x * cross[0] + pc.y * cross[1]) + pc.length_squared() * sum - at * s_r
        };
        let sxr = shifted(rx, pc.x, self.positions.x, [xx, xy]);
        let syr = shifted(ry, pc.y, self.positions.y, [xy, yy]);
        let m = [
            [n, 0., 0., s_r],
            [0., sxx, sxy, sxr],
            [0., sxy, syy, syr],
            [s_r, sxr, syr, s_r2],
        ];
        let (_, _, s_cr) = self.radial_sums(pc);
        let tx = self.color_positions[0] - self.sum * pc.x;
        let ty = self.color_positions[1] - self.sum * pc.y;
        let rhs = [self.sum, tx, ty, s_cr];
        let coefficients = solve4(m, rhs)?;
        let [_, gx, gy, curvature] = coefficients;
        // Per channel the extremum lies at -g / (2 c); one center for all, weighted by the
        // curvature: the least squares q of g + 2 c q = 0 over the channels
        let weight = curvature.length_squared();
        if weight < 1e-18 {
            return None;
        }
        let offset = -DVec2::new(gx.dot(curvature), gy.dot(curvature)) / (2. * weight);
        let spread = (sxx + syy) / n;
        if offset.length_squared() > 4. * spread {
            return None;
        }
        let center = pc + offset;
        // The colors as a line in s = |p - center|^2, by regression
        let (sum_s, sum_s2, sum_cs) = self.radial_sums(center);
        let var = sum_s2 - sum_s * sum_s / n;
        if var < 1e-12 {
            return None;
        }
        let slope = (sum_cs - self.sum * (sum_s / n)) / var;
        let color = (self.sum - slope * sum_s) / n;
        if slope.length_squared() < 1e-24 {
            return None;
        }
        Some(RadialFill {
            center,
            color,
            slope,
        })
    }

    /// Squared error of the pixels around `fill`.
    pub fn sse_about_radial(&self, fill: &RadialFill) -> f64 {
        let (sum_s, sum_s2, sum_cs) = self.radial_sums(fill.center);
        let (a, b) = (fill.color, fill.slope);
        (self.squares - 2. * a.dot(self.sum) - 2. * b.dot(sum_cs)
            + self.n * a.length_squared()
            + 2. * a.dot(b) * sum_s
            + b.length_squared() * sum_s2)
            .max(0.)
    }

    pub fn mean(&self) -> DVec3 {
        if self.n < 0.5 {
            DVec3::ZERO
        } else {
            self.sum / self.n
        }
    }

    /// Squared error of the pixels around their mean.
    pub fn sse(&self) -> f64 {
        if self.n < 0.5 {
            0.
        } else {
            (self.squares - self.sum.length_squared() / self.n).max(0.)
        }
    }

    /// Squared error of the pixels around `color`.
    pub fn sse_about(&self, color: DVec3) -> f64 {
        (self.squares - 2. * color.dot(self.sum) + self.n * color.length_squared()).max(0.)
    }

    /// How much squared error drawing `self` and `o` in one color adds over drawing each in its
    /// own (Ward's criterion).
    pub(super) fn merge_cost(&self, o: &Self) -> f64 {
        self.n * o.n / (self.n + o.n) * self.mean().distance_squared(o.mean())
    }
}
