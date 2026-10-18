use glam::{DVec3, IVec2};

#[derive(Clone, Copy, Debug)]
pub struct AntialiasParams {
    /// Minimum color distance between the two sides of an edge for its pixels to count as blends.
    /// Keeps smooth gradients, whose neighbors differ only slightly, from being sharpened.
    pub edge_contrast: f64,
    /// Max distance between a pixel and the blend of its two sides that still counts as a blend.
    pub residual: f64,
}

/// Coverage of a blended pixel by the side it was assigned to, kept for placing edges at sub-pixel
/// positions later.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code, reason = "not used for edge placement yet")]
pub struct Blend {
    pub coverage: f64,
}

/// Neighbor pairs on opposite sides of a pixel, including offset pairs that span blends two pixels
/// wide (as left by downscaling), where each blended pixel sits between a pure color and another
/// blended pixel.
const PAIRS: [(IVec2, IVec2); 12] = {
    const DIRS: [IVec2; 4] = [
        IVec2::new(1, 0),
        IVec2::new(0, 1),
        IVec2::new(1, 1),
        IVec2::new(1, -1),
    ];
    let mut pairs = [(IVec2::ZERO, IVec2::ZERO); 12];
    let mut i = 0;
    while i < 4 {
        let d = DIRS[i];
        let (d2, nd, nd2) = (
            IVec2::new(2 * d.x, 2 * d.y),
            IVec2::new(-d.x, -d.y),
            IVec2::new(-2 * d.x, -2 * d.y),
        );
        pairs[3 * i] = (nd, d);
        pairs[3 * i + 1] = (nd, d2);
        pairs[3 * i + 2] = (nd2, d);
        i += 1;
    }
    pairs
};

/// Replaces anti-aliased pixels by the pure color of the side they mostly belong to, so the
/// segmentation sees hard edges and does not turn the blended band along every edge into a sliver
/// region of its own.
pub fn unmix(
    colors: &[DVec3],
    width: usize,
    height: usize,
    params: &AntialiasParams,
) -> (Vec<DVec3>, Vec<Option<Blend>>) {
    let (w, h) = (width as i32, height as i32);
    let at = |p: IVec2| {
        (p.x >= 0 && p.y >= 0 && p.x < w && p.y < h).then(|| colors[(p.y * w + p.x) as usize])
    };

    let mut out = colors.to_vec();
    let mut blends = vec![None; colors.len()];
    for y in 0..h {
        for x in 0..w {
            let p = IVec2::new(x, y);
            let c = colors[(y * w + x) as usize];
            // The strongest edge this pixel is a blend across
            let mut best: Option<(f64, DVec3, f64)> = None;
            for (da, db) in PAIRS {
                let (Some(a), Some(b)) = (at(p + da), at(p + db)) else {
                    continue;
                };
                let axis = a - b;
                let contrast = axis.length_squared();
                if contrast < params.edge_contrast * params.edge_contrast
                    || best.is_some_and(|(s, _, _)| s >= contrast)
                {
                    continue;
                }
                let alpha = (c - b).dot(axis) / contrast;
                let residual = (c - (b + alpha * axis)).length();
                if alpha > 0.05 && alpha < 0.95 && residual < params.residual {
                    let (pure, coverage) = if alpha >= 0.5 {
                        (a, alpha)
                    } else {
                        (b, 1. - alpha)
                    };
                    best = Some((contrast, pure, coverage));
                }
            }
            if let Some((_, pure, coverage)) = best {
                out[(y * w + x) as usize] = pure;
                blends[(y * w + x) as usize] = Some(Blend { coverage });
            }
        }
    }
    (out, blends)
}
