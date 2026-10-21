# Ocean

A custom real-time ocean renderer in Rust + wgpu: FFT waves, physically driven whitecaps,
hardware ray-traced reflections, at 60 fps in a 1280×720 window.

Work in progress — see `docs/superpowers/specs/` for the design and `docs/superpowers/plans/`
for the build plan.

## Requirements

- Windows, a GPU with Vulkan ray-query support (developed on an NVIDIA RTX 2000 Ada laptop GPU).
  Without ray query it falls back to screen-space reflections.

## Build and run

```
cargo run --release
```
