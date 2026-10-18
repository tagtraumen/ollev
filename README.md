# ollev

Raster images traced into layered vector scenes. Each region of the picture becomes a shape
drawn over the shape it lies in; lines become strokes (dashed where the picture dashes them),
shading becomes linear, radial and path gradients, soft edges become blurs, repeated objects
become instances of one definition, and translucent layers are lifted off what they cover.
One objective decides everything: the squared colour error plus a price per cubic segment,
set by a detail slider.

ollev is the tracer behind Graphite's vectorize node.

## Use

```rust
let pixels: Vec<u8> = [255, 0, 0, 255].repeat(64 * 64); // RGBA, row by row
let image = ollev::Image {
    pixels: ollev::Pixels::Rgba8(&pixels),
    width: 64,
    height: 64,
};
let model = ollev::trace(&image, &ollev::TraceOptions::default()).expect("a traceable image");
for object in model.objects() {
    // object.geometry (fill loops or a stroke), object.paint, object.opacity, object.blur
}
```

`trace_with` takes a progress callback as well: it hears the stage and how far it is, and
the trace stops within a fraction of a second when the callback breaks.

`TraceOptions` holds:

- `tolerance`: the detail slider, as a multiple of the picture's own uncertainty; lower keeps
  more detail.
- `detail`: the price of a segment, read from how busy the picture is (`Detail::Auto`),
  scaled (`Detail::Scaled`) or given (`Detail::Absolute`).
- `matte` and `keep_transparency`: what transparent pixels are composited over, and whether
  the model is clipped to the picture's opaque part.
- `max_pixels`: a larger picture is traced at the size that holds this many pixels and its
  model scaled back up.
- `host`: what the host draws (`Capabilities`); every feature it lacks (even-odd fills, path
  gradients, blurs, radial focus, clips, dashes, transparent gradient stops) is rewritten in
  the ones it has.
- `stages`: the pipeline's stages (`Stages`: translucency, strokes, primitives, effects,
  instancing), all on by default; one switched off is skipped, so the trace is faster.

The model's coordinates are pixels of the traced image, its colours sRGB-encoded with
straight alpha; the `model` module documents the contract.

## Features

- `svg`: the `emit` module, which writes a model as SVG and checks every output against the
  model's own render (with resvg).
- `diagnostics`: the developers' logs and dumps (`Diagnostics` in `TraceOptions`); without
  it none of that code is compiled.
- `glam-assert`: glam's argument checks as panics; no change in behaviour.

The default build has no dependency besides glam and builds for `wasm32-unknown-unknown`.
