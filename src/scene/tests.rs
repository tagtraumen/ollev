use super::*;

/// A shape moved by an offset covers the same pixels moved by it, its gradient with it.
#[test]
fn a_moved_shape_paints_the_same_moved() {
    let fill = Fill {
        color: Vec3::ZERO,
        gradient: Some(Gradient::Linear {
            start: Vec2::new(1., 1.),
            end: Vec2::new(5., 1.),
            to: [1., 1., 1., 1.],
            stops: crate::render::Stops::NONE,
        }),
        opacity: 1.,
    };
    let shape = Shape::new(square(Vec2::new(1., 1.), Vec2::new(5., 4.)), fill);
    let mut moved = shape.clone();
    moved.translate(Vec2::new(3., 2.));
    assert_eq!(
        (moved.min, moved.max),
        (shape.min + Vec2::new(3., 2.), shape.max + Vec2::new(3., 2.))
    );
    let (a, b) = (shape.raster(), moved.raster());
    assert_eq!((b.x0, b.y0, b.w, b.h), (a.x0 + 3, a.y0 + 2, a.w, a.h));
    assert_eq!(a.values, b.values);
    // the gradient runs from the moved start: dark at x 4, light at x 8
    let (dark, light) = (moved.fill.at_pixel(4, 3), moved.fill.at_pixel(7, 3));
    assert!(dark.x < 0.2 && light.x > 0.6, "{dark} {light}");
}

/// The layer of a translucent shape lets through what its opacity leaves, as the render
/// composites it: a half-opaque square over white and over black.
#[test]
fn a_translucent_layer_lets_its_share_through() {
    let mut scene = Scene::with_background(8, 8, Vec3::ONE);
    scene.push(Shape::new(
        square(Vec2::ZERO, Vec2::splat(8.)),
        Fill {
            opacity: 0.5,
            ..Fill::solid(Vec3::X)
        },
    ));
    let window = Window::full(8, 8);
    let layer = scene
        .layer(window, 1)
        .expect("the square reaches the window");
    let j = 4 * 8 + 4;
    assert!((layer.transmittance[j] - Vec3::splat(0.5)).abs().max_element() < 1e-4);
    let over_white = layer.color[j] + layer.transmittance[j] * Vec3::ONE;
    let rendered = scene.render(window, 0..2, [1.; 3])[j];
    assert!(
        (over_white - rendered).abs().max_element() < 1e-4,
        "{over_white} {rendered}"
    );
}

/// The loops of an axis-aligned square from `a` to `b`, straight cubics.
pub fn square(a: Vec2, b: Vec2) -> Vec<Vec<Cubic>> {
    let corners = [a, Vec2::new(b.x, a.y), b, Vec2::new(a.x, b.y)];
    vec![
        (0..4)
            .map(|k| {
                let (p, q) = (corners[k], corners[(k + 1) % 4]);
                [p, p.lerp(q, 1. / 3.), p.lerp(q, 2. / 3.), q]
            })
            .collect(),
    ]
}

/// The loop of a disc of radius `r` around `c`, four cubics.
pub fn disc(c: Vec2, r: f32) -> Vec<Vec<Cubic>> {
    let k = 0.552_284_8 * r;
    let p = [Vec2::X, Vec2::Y, -Vec2::X, -Vec2::Y].map(|d| c + d * r);
    let t = [Vec2::Y, -Vec2::X, -Vec2::Y, Vec2::X];
    vec![
        (0..4)
            .map(|i| {
                let j = (i + 1) % 4;
                [p[i], p[i] + t[i] * k, p[j] - t[j] * k, p[j]]
            })
            .collect(),
    ]
}
