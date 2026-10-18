//! The tracer for a host (Graphite): a raw RGBA image in, the scene model out. The tracer reads
//! sRGB-encoded colours without alpha, so a pixel is brought to that encoding and composited
//! over the matte; the model's colours are sRGB-encoded in 0..1, as SVG reads them.

#[cfg(feature = "diagnostics")]
use crate::diagnostics::Diagnostics;
use crate::lower::Capabilities;

/// A raw image's pixels, row by row from the top, four channels per pixel.
#[derive(Clone, Copy, Debug)]
pub enum Pixels<'a> {
    /// sRGB-encoded bytes with straight alpha.
    Rgba8(&'a [u8]),
    /// Floats in 0..1, linear light or sRGB-encoded, with straight or premultiplied alpha.
    RgbaF32 {
        /// The channels, four per pixel.
        data: &'a [f32],
        /// Whether the colour channels are linear light, else sRGB-encoded.
        linear: bool,
        /// Whether the colour channels are multiplied by the alpha.
        premultiplied: bool,
    },
}

/// An image to trace: its pixels and size.
#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    /// The pixels, `width * height` of them.
    pub pixels: Pixels<'a>,
    /// Pixels per row.
    pub width: usize,
    /// Rows.
    pub height: usize,
}

/// How much detail a trace keeps: the price of a segment, scaled by a level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Detail {
    /// The level the picture's busyness asks for: 1 for flat art, up to 3 for busy pictures.
    Auto,
    /// This level.
    Absolute(f64),
    /// The automatic level times this factor (a host's detail slider).
    Scaled(f64),
}

/// The stages of the pipeline a trace runs, all by default. A stage switched off is skipped,
/// its work not done, so the trace is faster; what it would have drawn is drawn by the stages
/// that stay (the plain shapes in the end).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stages {
    /// Translucent layers: the families and the discovery proposing a layer over what it
    /// covers, the beneath traced under a kept layer, and the layer's marks (lenses, hems,
    /// lines). Off, every region is drawn opaque.
    pub translucency: bool,
    /// Strokes: bands recognized as a pen along a centreline, strokes continued across gaps,
    /// dashed and dotted lines, and the linework that draws the lines the fills miss. Off,
    /// lines are drawn as filled shapes.
    pub strokes: bool,
    /// Primitives: outlines refitted as circles, ellipses and rectangles (rounded or not).
    /// Off, every outline stays a path of fitted cubics.
    pub primitives: bool,
    /// Effects: runs of shaded bands merged into one shape with a gradient, grounds drawn flat
    /// under glows fading to transparent, and edges blurred where the picture shows them as a
    /// ramp. Off, edges stay hard (a fill's own linear or radial gradient stays).
    pub effects: bool,
    /// Instancing: repeated shapes drawn as instances of one definition. Off, every copy is an
    /// object of its own.
    pub instancing: bool,
}

impl Stages {
    /// Every stage.
    pub const ALL: Self = Self {
        translucency: true,
        strokes: true,
        primitives: true,
        effects: true,
        instancing: true,
    };
}

impl Default for Stages {
    fn default() -> Self {
        Self::ALL
    }
}

/// How a host traces an image.
#[derive(Clone, Debug)]
pub struct TraceOptions {
    /// The detail slider: a shape, stroke or mark is drawn where it removes residual beyond this
    /// multiple of the picture's uncertainty, so lower is more detail.
    pub tolerance: f64,
    /// The price of a segment against the error it removes (see [`Detail`]).
    pub detail: Detail,
    /// What a pixel is composited over by the part it leaves transparent, sRGB-encoded in 0..1.
    pub matte: [f32; 3],
    /// Whether what the picture leaves transparent stays so: the model is then clipped to the
    /// picture's opaque part ([`crate::model::Model::clip`], its alpha at a half), its colours
    /// still read over the matte.
    pub keep_transparency: bool,
    /// The most pixels traced: a larger picture is traced at the size that holds this many
    /// and its model scaled back up (the trace's time and memory grow with its pixels).
    pub max_pixels: Option<usize>,
    /// What the host draws; the model is lowered to it, each feature the host lacks rewritten
    /// in the ones it has.
    pub host: Capabilities,
    /// The stages the trace runs.
    pub stages: Stages,
    /// What the trace reports and dumps (none by default).
    #[cfg(feature = "diagnostics")]
    pub diagnostics: Diagnostics,
}

impl Default for TraceOptions {
    fn default() -> Self {
        Self {
            tolerance: crate::plan::TOLERANCE,
            detail: Detail::Auto,
            matte: [1.; 3],
            keep_transparency: false,
            max_pixels: None,
            host: Capabilities::ALL,
            stages: Stages::ALL,
            #[cfg(feature = "diagnostics")]
            diagnostics: Diagnostics::default(),
        }
    }
}

/// Why an image is not traced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceError {
    /// A side of zero pixels.
    Empty,
    /// The pixels do not hold four channels for each pixel of the size.
    Size {
        /// The channel values the size asks for.
        expected: usize,
        /// The channel values given.
        actual: usize,
    },
    /// The progress callback asked to stop ([`crate::trace_with`]).
    Cancelled,
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraceError::Empty => write!(f, "the image has no pixels"),
            TraceError::Size { expected, actual } => {
                write!(f, "expected {expected} channel values, got {actual}")
            }
            TraceError::Cancelled => write!(f, "the trace was cancelled"),
        }
    }
}

impl std::error::Error for TraceError {}

/// The picture downsampled from `w` by `h` to `tw` by `th`, each new pixel the mean of the `s`
/// by `s` square of old pixels it covers (fractions at its sides), in the sRGB encoding SVG
/// composites in.
pub(crate) fn downsampled(rgb: &[u8], w: usize, h: usize, tw: usize, th: usize, s: f64) -> Vec<u8> {
    let mut out = Vec::with_capacity(3 * tw * th);
    // The old pixels a new one spans along an axis, with their shares
    let spans = |n: usize, old: usize| -> Vec<Vec<(usize, f64)>> {
        (0..n)
            .map(|k| {
                let (a, b) = (k as f64 * s, ((k + 1) as f64 * s).min(old as f64));
                (a.floor() as usize..(b.ceil() as usize).min(old))
                    .map(|i| (i, (b.min(i as f64 + 1.) - a.max(i as f64)).max(0.)))
                    .collect()
            })
            .collect()
    };
    let (xs, ys) = (spans(tw, w), spans(th, h));
    for row in &ys {
        for col in &xs {
            let (mut sum, mut weight) = ([0f64; 3], 0.);
            for &(y, wy) in row {
                for &(x, wx) in col {
                    let p = &rgb[3 * (y * w + x)..][..3];
                    for c in 0..3 {
                        sum[c] += f64::from(p[c]) * wx * wy;
                    }
                    weight += wx * wy;
                }
            }
            out.extend(sum.map(|v| (v / weight.max(f64::MIN_POSITIVE)).round() as u8));
        }
    }
    out
}

/// The image as the tracer reads it, sRGB-encoded bytes, three per pixel, composited over
/// `matte` in that encoding (as SVG composites); and each pixel's alpha.
pub(crate) fn pixels8(image: &Image, matte: [f32; 3]) -> Result<(Vec<u8>, Vec<f32>), TraceError> {
    let n = image.width * image.height;
    if n == 0 {
        return Err(TraceError::Empty);
    }
    let actual = match image.pixels {
        Pixels::Rgba8(d) => d.len(),
        Pixels::RgbaF32 { data, .. } => data.len(),
    };
    if actual != 4 * n {
        return Err(TraceError::Size {
            expected: 4 * n,
            actual,
        });
    }
    let byte = |v: f32| (v.clamp(0., 1.) * 255.).round() as u8;
    let mut out = Vec::with_capacity(3 * n);
    let mut alphas = Vec::with_capacity(n);
    for i in 0..n {
        let (c, a) = match image.pixels {
            Pixels::Rgba8(d) => {
                let p = &d[4 * i..4 * i + 4];
                (
                    [p[0], p[1], p[2]].map(|v| f32::from(v) / 255.),
                    f32::from(p[3]) / 255.,
                )
            }
            Pixels::RgbaF32 {
                data,
                linear,
                premultiplied,
            } => {
                let p = &data[4 * i..4 * i + 4];
                let finite = |v: f32| if v.is_finite() { v } else { 0. };
                let a = finite(p[3]).clamp(0., 1.);
                let c = [p[0], p[1], p[2]].map(|v| {
                    let v = finite(v);
                    let v = if premultiplied {
                        if a > 0. { v / a } else { 0. }
                    } else {
                        v
                    };
                    if linear { encode(v.clamp(0., 1.)) } else { v }
                });
                (c, a)
            }
        };
        for (c, m) in c.iter().zip(matte) {
            out.push(byte(c * a + m * (1. - a)));
        }
        alphas.push(a);
    }
    Ok((out, alphas))
}

/// A linear-light value in sRGB's encoding.
fn encode(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1. / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{trace, trace_with};

    /// The image's colours as the tracer reads them.
    fn rgb8(image: &Image, matte: [f32; 3]) -> Result<Vec<u8>, TraceError> {
        pixels8(image, matte).map(|(rgb, _)| rgb)
    }

    fn image(pixels: Pixels, width: usize, height: usize) -> Image {
        Image {
            pixels,
            width,
            height,
        }
    }

    #[test]
    fn each_encoding_reads_as_the_same_bytes() {
        let bytes = [200u8, 40, 10, 255, 0, 128, 255, 128, 9, 9, 9, 0];
        let straight: Vec<f32> = bytes.iter().map(|&v| f32::from(v) / 255.).collect();
        let premultiplied: Vec<f32> = straight
            .chunks(4)
            .flat_map(|p| [p[0] * p[3], p[1] * p[3], p[2] * p[3], p[3]])
            .collect();
        let linear: Vec<f32> = straight
            .chunks(4)
            .flat_map(|p| {
                let l = |v: f32| {
                    if v <= 0.04045 {
                        v / 12.92
                    } else {
                        ((v + 0.055) / 1.055).powf(2.4)
                    }
                };
                [l(p[0]), l(p[1]), l(p[2]), p[3]]
            })
            .collect();
        let white = [1.; 3];
        let want = rgb8(&image(Pixels::Rgba8(&bytes), 3, 1), white).unwrap();
        // opaque as given, half covered over white, transparent the matte
        assert_eq!(&want[..3], &[200, 40, 10]);
        assert_eq!(&want[3..6], &[127, 191, 255]);
        assert_eq!(&want[6..], &[255, 255, 255]);
        for (data, linear, premultiplied) in [
            (&straight, false, false),
            (&premultiplied, false, true),
            (&linear, true, false),
        ] {
            let got = rgb8(
                &image(
                    Pixels::RgbaF32 {
                        data,
                        linear,
                        premultiplied,
                    },
                    3,
                    1,
                ),
                white,
            )
            .unwrap();
            assert_eq!(got, want, "linear {linear}, premultiplied {premultiplied}");
        }
    }

    #[test]
    fn a_wrong_size_is_an_error() {
        let bytes = [0u8; 15];
        assert_eq!(
            rgb8(&image(Pixels::Rgba8(&bytes), 2, 2), [1.; 3]),
            Err(TraceError::Size {
                expected: 16,
                actual: 15
            })
        );
        assert_eq!(
            rgb8(&image(Pixels::Rgba8(&[]), 0, 3), [1.; 3]),
            Err(TraceError::Empty)
        );
    }

    #[test]
    fn a_disc_traces_to_a_model() {
        let (w, h) = (48, 40);
        let mut px = vec![0u8; 4 * w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 - 24., y as f32 - 20.);
                let inside = dx * dx + dy * dy < 12. * 12.;
                let p = &mut px[4 * (y * w + x)..][..4];
                p.copy_from_slice(&if inside {
                    [220, 30, 30, 255]
                } else {
                    [0, 0, 0, 0]
                });
            }
        }
        let model = trace(&image(Pixels::Rgba8(&px), w, h), &TraceOptions::default()).unwrap();
        assert_eq!((model.width, model.height), (w, h));
        assert!(!model.objects().is_empty());
    }

    /// Degenerate pictures trace for each host, transparency kept or not, without a panic under
    /// glam's asserts (`--features ollev/glam-assert`): tiny, one colour, clear, a checker.
    #[test]
    fn degenerate_images_trace() {
        let flat = |w: usize, h: usize, p: [u8; 4]| (w, h, p.repeat(w * h));
        let checker = (0..64)
            .flat_map(|i| match (i % 8 + i / 8) % 2 {
                0 => [0, 0, 0, 255],
                _ => [255; 4],
            })
            .collect();
        let pictures = [
            flat(1, 1, [128, 128, 128, 255]),
            flat(2, 1, [128, 128, 128, 255]),
            flat(1, 3, [128, 128, 128, 255]),
            flat(3, 3, [128, 128, 128, 255]),
            flat(40, 30, [30, 90, 200, 255]),
            flat(16, 16, [0, 0, 0, 0]),
            flat(16, 16, [255, 0, 0, 128]),
            (8, 8, checker),
        ];
        let graphite = Capabilities {
            even_odd: false,
            path_gradients: false,
            blur: false,
            radial_focus: false,
            ..Capabilities::ALL
        };
        for (w, h, px) in &pictures {
            for host in [Capabilities::ALL, graphite, Capabilities::NONE] {
                for keep_transparency in [false, true] {
                    let options = TraceOptions {
                        keep_transparency,
                        host,
                        ..TraceOptions::default()
                    };
                    trace(&image(Pixels::Rgba8(px), *w, *h), &options).unwrap();
                }
            }
        }
    }

    /// Two discs and a ring on a ground, transparent outside an inset frame where `clear`.
    fn picture(w: usize, h: usize, clear: bool) -> Vec<u8> {
        let mut px = vec![0u8; 4 * w * h];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
                let d = |cx: f32, cy: f32| ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
                let mut c = [230, 220, 190];
                if d(0.3, 0.4) < 0.18 {
                    c = [200, 40, 30];
                }
                if (0.1..0.16).contains(&d(0.7, 0.6)) {
                    c = [30, 50, 140];
                }
                let inside = (0.1..0.9).contains(&fx) && (0.1..0.9).contains(&fy);
                let a = if clear && !inside { 0 } else { 255 };
                px[4 * (y * w + x)..][..4].copy_from_slice(&[c[0], c[1], c[2], a]);
            }
        }
        px
    }

    fn psnr(a: &[glam::Vec3], b: &[glam::Vec3]) -> f64 {
        let se: f64 = a
            .iter()
            .zip(b)
            .map(|(p, q)| f64::from((*p - *q).length_squared()))
            .sum();
        10. * (3. * a.len() as f64 / se.max(1e-12)).log10()
    }

    /// Scaled is the automatic level times its factor, Auto the factor one.
    #[test]
    fn detail_modes_agree() {
        let (w, h) = (64, 48);
        let px = picture(w, h, false);
        let img = image(Pixels::Rgba8(&px), w, h);
        let at = |detail| {
            let options = TraceOptions {
                detail,
                ..TraceOptions::default()
            };
            format!("{:?}", trace(&img, &options).unwrap())
        };
        let auto = crate::tracer::auto_detail(&rgb8(&img, [1.; 3]).unwrap(), w, h);
        assert_eq!(at(Detail::Auto), at(Detail::Scaled(1.)));
        assert_eq!(at(Detail::Auto), at(Detail::Absolute(auto)));
        assert_eq!(at(Detail::Scaled(2.5)), at(Detail::Absolute(2.5 * auto)));
    }

    /// A picture over the pixel budget is traced smaller and its model scaled back: the frame
    /// is the picture's, and it renders near the full trace.
    #[test]
    fn a_pixel_budget_traces_smaller_and_scales_back() {
        let (w, h) = (160, 120);
        let px = picture(w, h, false);
        let img = image(Pixels::Rgba8(&px), w, h);
        let full = trace(&img, &TraceOptions::default()).unwrap();
        let options = TraceOptions {
            max_pixels: Some(w * h / 4),
            ..TraceOptions::default()
        };
        let small = trace(&img, &options).unwrap();
        assert_eq!((small.width, small.height), (w, h));
        let db = psnr(&small.render().0, &full.render().0);
        assert!(db > 25., "{db:.1} dB against the full trace");
    }

    /// With transparency kept, the drawing is clipped to the picture's opaque part: the frame
    /// outside it renders as the white the host composites over, inside as without the clip.
    #[test]
    fn transparency_kept_clips_the_drawing() {
        let (w, h) = (80, 60);
        let px = picture(w, h, true);
        let img = image(Pixels::Rgba8(&px), w, h);
        let options = TraceOptions {
            keep_transparency: true,
            matte: [0., 0., 0.],
            ..TraceOptions::default()
        };
        let model = trace(&img, &options).unwrap();
        assert!(model.clip.is_some());
        let (pixels, _) = model.render();
        // a corner the picture leaves clear, and the red disc's centre
        assert!((pixels[w + 1] - glam::Vec3::ONE).length() < 0.02);
        let red = pixels[(h * 4 / 10) * w + w * 3 / 10];
        assert!(
            (red - glam::Vec3::new(200., 40., 30.) / 255.).length() < 0.05,
            "{red}"
        );
        let opaque = trace(&image(Pixels::Rgba8(&picture(w, h, false)), w, h), &options).unwrap();
        assert!(opaque.clip.is_none());
    }

    /// A trace whose callback breaks returns the cancellation, and soon.
    #[test]
    fn a_cancelled_trace_says_so() {
        let (w, h) = (96, 72);
        let px = picture(w, h, false);
        let img = image(Pixels::Rgba8(&px), w, h);
        let mut asks = 0;
        let result = trace_with(&img, &TraceOptions::default(), |_| {
            asks += 1;
            if asks > 20 {
                std::ops::ControlFlow::Break(())
            } else {
                std::ops::ControlFlow::Continue(())
            }
        });
        assert_eq!(result.err(), Some(TraceError::Cancelled));
        assert_eq!(asks, 21);
    }

    /// The host's capabilities reach the model: no even-odd fill is left for a host without.
    #[test]
    fn the_model_is_lowered_for_the_host() {
        let (w, h) = (64, 48);
        let px = picture(w, h, false);
        let options = TraceOptions {
            host: Capabilities::NONE,
            ..TraceOptions::default()
        };
        let model = trace(&image(Pixels::Rgba8(&px), w, h), &options).unwrap();
        assert!(model.objects().iter().all(|o| !matches!(
            o.geometry,
            crate::model::Geometry::Fill {
                rule: crate::model::FillRule::EvenOdd,
                ..
            }
        )));
    }
}
