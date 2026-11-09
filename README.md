# ollev

Raster images traced into layered vector scenes. Each region of the picture becomes a shape
drawn over the shape it lies in; lines become strokes (dashed where the picture dashes them),
shading becomes linear, radial and path gradients, soft edges become blurs, repeated objects
become instances of one definition, and translucent layers are lifted off what they cover.
One objective decides everything: the squared colour error plus a price per cubic segment,
set by a detail slider.

ollev is the tracer behind Graphite's vectorize node.

Disclaimer: Claude was used to aid with the development and not every line was reviewed by a human.


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
the trace stops (within a fraction of a second) when the callback breaks. `progress.overall`
is how much of the whole trace is done, from 0 to 1 and never going back, for a progress
bar: each part of the trace weighted by an estimate of its time, refined as the trace learns
the picture (its runs of colour, its planned shapes, its translucent layers).

A host that must not block, such as a UI thread or wasm in a browser, runs the trace in time
slices with `Tracing`. No threads and no executor: each `step` runs the trace on the caller's
thread until its callback breaks at a checkpoint, and the next `step` resumes it there.
Checkpoints come many times a second, on a megapixel picture a few tenths of a second apart
at most on a desktop CPU; the host supplies the clock. Dropping the handle cancels the trace;
the handle is `Send`, stepped on one thread at a time.

At a pause, `progress.model()` is the latest scene the trace kept for showing, in the form the
finished trace returns: a large picture's flat regions within milliseconds, then the plan in
flat colours, then the trace's own scenes as each shape is fitted and as the later stages change
them. It is cheap enough to ask for every second or two, since what was lowered for an earlier
call is kept; `progress.revision()` tells when the scene changed.

```rust
let mut tracing = ollev::Tracing::new(&image, &ollev::TraceOptions::default())?;
let model = loop {
    let end = now() + 0.1; // the host's clock, in seconds
    match tracing.step(|_| {
        if now() < end { std::ops::ControlFlow::Continue(()) } else { std::ops::ControlFlow::Break(()) }
    })? {
        // where it paused: stage, fraction, and the latest scene kept for showing
        // (`progress.model()`, in the form the finished trace returns)
        ollev::Step::Paused(progress) => continue,
        ollev::Step::Done(model) => break model,
    }
};
```

We strive to reduce panics throughout the library. If you encounter panics through either programming logic errors or failed allocations please report them.

`TraceOptions` holds:

- `tolerance`: the detail slider, as a multiple of the picture's own uncertainty; lower keeps
  more detail.
- `detail`: the price of a segment, read from how busy the picture is (`Detail::Auto`, at
  `Detail::AUTO_SHARE` of the level read, a coarse and quick trace), scaled
  (`Detail::Scaled`, 2 for the level read itself) or given (`Detail::Absolute`).
- `matte` and `keep_transparency`: what transparent pixels are composited over, and whether
  the model is clipped to the picture's opaque part.
- `clip_to_image`: whether the model is clipped to the image's rectangle (on by default): a
  filled shape running past the edge is cut there, a stroke or blurred shape is clipped.
- `max_pixels`: a larger picture is traced at the size that holds this many pixels and its
  model scaled back up.
- `host`: what the host draws (`Capabilities`); every feature it lacks (even-odd fills, path
  gradients, blurs, radial focus, clips, dashes, transparent gradient stops) is rewritten in
  the ones it has.
- `stages`: the pipeline's stages (`Stages`: translucency, strokes, primitives, effects,
  instancing, path gradients); by default (`Stages::DEFAULT`) the quick ones, primitives and
  instancing, run and the others are skipped, `Stages::ALL` runs every one.

The model's coordinates are pixels of the traced image, its colours sRGB-encoded with
straight alpha; the `model` module documents the contract.

## Features

- `svg`: the `emit` module, which writes a model as SVG and checks every output against the
  model's own render (with resvg).
- `diagnostics`: the developers' logs and dumps (`Diagnostics` in `TraceOptions`); without
  it none of that code is compiled.
- `glam-assert`: glam's argument checks as panics; no change in behaviour.

The default build has no dependency besides glam and builds for `wasm32-unknown-unknown`.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
