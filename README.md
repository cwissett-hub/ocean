# Ocean

A custom real-time sunset ocean in Rust + wgpu: physically based FFT waves, whitecaps driven
by the waves' own folding, hardware ray-traced reflections and spray. Sail a driftwood raft
through procedurally generated desert islands at 60 fps in a 1280×720 window.

![Sunset sea with a floating buoy and the sun glitter path](docs/screenshot.png)

## What's in it

- **Waves**: four 256² FFT cascades (1 km swell down to 12 cm ripples) evolved on the GPU from
  a JONSWAP spectrum (fetch-limited, capped at full development) plus a swell component, with
  finite-depth dispersion for the 25 m deep sea.
- **Whitecaps**: foam is injected where the combined horizontal displacement folds the surface
  (Jacobian below a threshold), accumulates and decays; spray that lands adds to it.
- **Sky**: Hillaire 2020 atmosphere (transmittance, multiple scattering, sky-view LUTs).
- **Water shading**: Fresnel, GGX sun glitter whose roughness carries the slope variance of
  unresolved wave scales, scattering in the water column, a sandy seabed seen through the
  water everywhere (turquoise in the shallows), crest glow, aerial perspective.
- **Islands**: sparse desert islands a couple of kilometres apart, sometimes in archipelagos,
  with dunes, beaches shelving into the sea floor and palm trees; deterministic from a seed,
  streamed in and out around the raft on a worker thread. Waves calm over the shallows and
  leave a line of foam at the beach.
- **Raft**: seven driftwood logs, a mast and a square sail. It rides the GPU wave-height query
  (heaving, pitching and rolling with the swell), sails by the wind, paddles, beaches on sand,
  throws bow spray when it slams into a wave, and pushes the water aside: a GPU wave simulation
  (Tessendorf's iWave, dispersive like real deep-water waves) gives it a bow wave and a Kelvin
  wake, with white water along the hull.
- **Reflections**: hardware ray queries against a 512 m patch of the displaced waves (refitted
  every frame) and every object (raft, buoy, islands, palms), so crests reflect and hide each
  other; screen-space fallback.
- **Lighting**: ray-traced soft sun shadows and contact ambient occlusion from the raft, buoy,
  islands and palms (one jittered ray each per pixel, resolved by temporal anti-aliasing),
  contrast-adaptive sharpening and dithered output.
- **Particles**: spray bursts from breaking crests (from ~8 m/s wind), faint mist.
- **Unbounded world**: render coordinates are rebased every 4 km by whole wave periods, so you
  can sail on indefinitely without f32 precision loss.

GPU frame ≈ 5–6 ms (bench course p95 5.9 ms chase, 5.4 ms high) at 1600×900 on an RTX 2000
Ada laptop GPU.

## Requirements

- Windows; a GPU with Vulkan ray-query support (developed on an NVIDIA RTX 2000 Ada laptop GPU).
  Without ray query it falls back to screen-space reflections. On hybrid-graphics laptops the
  exe asks for the discrete GPU.

## Build and run

```sh
cargo run --release
```

The exe looks for `shaders/` next to itself or in the working directory (for hot reload) and
otherwise uses the copies compiled in.

## Controls

| Key | Action |
| --- | --- |
| A / D | steer the raft |
| W / S | paddle forward / back (push off a beach) |
| Space | sail up / down |
| C | camera: chase → deck → high → free fly |
| WASD, Q / E, right-drag, Shift | free fly: move, down / up, look, fast |
| F1 | settings panel (raft status, wind, fetch, swell, foam, spray, sun, reflections, render scale, vsync) |
| F2 | fps readout (detailed GPU timings via the settings panel) |
| H | show / hide the controls hint |
| F5 | reload shaders from disk (errors shown, last good shaders keep running) |
| Esc | quit |

The first island lies about 400 m ahead, under the sun.

## Command line

- `--set key=value` — set any setting, e.g. `--set wind_speed=15 --set sun_elevation_deg=1`
- `--start-at x,z[,heading]` — start the raft at a world position (metres) and heading (degrees)
- `--bench N [--bench-out file.json] [--bench-camera chase|deck|high]` — sail a scripted
  course past the first island, rendering N frames uncapped after a warm-up, and write GPU and
  CPU frame-time statistics as JSON
- `--panel` — open the settings panel at start
- `--freeze-time S` — pin the waves at time S (for side-by-side comparisons)

## Tests

`cargo test` runs the wave physics, spectrum, foam, raft and buoy physics, island placement
and streaming, cameras, shader validation and other unit tests, plus headless GPU tests (GPU
FFT vs CPU reference, wave query vs CPU surface including the shallows, seabed map vs CPU
heights, ray query).
