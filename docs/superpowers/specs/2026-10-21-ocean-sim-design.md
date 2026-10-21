# Ocean Sim — Design

Date: 2026-10-21
Status: draft, awaiting approval

## Intent

A custom-written, real-time water renderer that looks like footage of a real
sunset sea rather than a game, while holding 60 fps on this laptop. It is a
showpiece first: launch it and watch. It is also the foundation for later
free-flight and a drivable boat, so version 1 is shaped to accept those without
rework.

## The picture

- Open sea, no land, no seabed visible. Sunset: low sun near the horizon,
  warm orange-to-purple sky, long glittering sun path across the swell.
- Camera fixed, roughly 10–30 m above the water, pitched steeply down but not
  straight down, so the frame is mostly water with a strip of glowing horizon
  at the top. (Exact height/pitch: builder's call, tuned by eye.)
- Gentle rolling swell from one dominant direction plus smaller wind chop. At
  the default settings almost nothing breaks: only occasional small foam
  patches on the steepest crests.
- Sun-facing crests glow green-gold (light passing through thin water);
  troughs fall into deep blue-teal.

## Goals (version 1)

1. **Physically based waves.** Tessendorf FFT ocean using a measured spectrum
   (JONSWAP with directional spreading; builder may add a swell component),
   3–4 cascades spanning ~1 km swell down to ~0.3 m ripples, with horizontal
   displacement ("choppy") so crests are sharp and troughs broad.
2. **Physically driven whitecaps.** Foam appears where the Jacobian of the
   displacement shows the surface folding (where real waves break), accumulates
   over time and decays with streaky texture. Not painted.
3. **Hardware ray-traced reflections** of the sky, sun and any scene objects,
   using the GPU's RT cores. Must have a non-RT fallback (screen-space
   reflection + sky/environment lookup) selectable at runtime, both for
   comparison and in case RT is unavailable.
4. **Refraction and water body.** Ray-traced or screen-space refraction, depth-
   based absorption and scattering (blue-teal colour), subsurface glow on thin
   crests, Fresnel, sun glitter on micro-facets.
5. **Physical sky.** Atmospheric-scattering sky and sun disc (e.g. Hillaire or
   Bruneton style); sun height drives the whole palette.
6. **Faked splashes and particles.** GPU particles: spray thrown off the
   steepest/breaking crests when wind is up, drifting sea mist near the
   surface, and spray that lands feeds back into surface foam. Designed so a
   later boat hull can emit splashes into the same system.
7. **Settings screen.** An in-window panel (toggle with a key) with live
   sliders/toggles at minimum for: wind speed, wind direction, wave size /
   fetch, swell amount, foam amount, sun height, sun direction, reflection
   mode (RT / screen-space), particle amount, render scale, and the
   performance overlay. Changes apply live with no restart.
8. **Performance overlay.** Frame time, fps, and a per-pass GPU timing
   breakdown (FFT, foam, RT, shading, particles), toggleable.
9. **Dev hotkeys.** At least: toggle settings panel, toggle overlay, hot-reload
   shaders.

## Constraints

- **Hardware:** NVIDIA RTX 2000 Ada Laptop GPU (8 GB, 3rd-gen RT cores) +
  Intel Arc iGPU which drives the 1920×1200 @ 60 Hz panel. CPU Core Ultra 9
  185H, 64 GB RAM, Windows 11.
- **Must run on the NVIDIA GPU** without the user fiddling: request the
  high-performance adapter explicitly and also export the
  `NvOptimusEnablement` / `AmdPowerXpressRequestHighPerformance` symbols.
  Show the active adapter name in the overlay.
- **Windowed, not fullscreen.** Opens at 1280×720 and renders at window
  resolution by default (render scale slider allows going lower/higher).
  Resizable.
- **Performance target:** 60 fps steady at 1280×720 with all features on and
  default settings — GPU frame time at or under ~12 ms to leave thermal
  headroom on a laptop. Display is 60 Hz, so frame time is the real metric;
  vsync on by default, with a toggle to uncap for measurement.
- **Custom-written.** No game engine. Language and libraries are the
  builder's call (see below for the recommendation); Rust is preferred since
  the user already ships Rust (taskbar-eq).
- **Repo:** remote `https://github.com/cwissett-hub/ocean` (public; chosen by
  the user). Nothing employer-related may be committed: no internal registry
  names or URLs, no work email, no local user paths.
- Python-free; no runtime downloads; a single exe plus its shaders/assets is
  the deliverable.

## Decided by the user

- Custom-written, not an engine; language is the builder's choice.
- Windowed at below full-screen resolution.
- Sunset, camera above looking down at gentle rolling waves.
- Start with a fixed view; free flight and a drivable boat come later.
- Splashes/particles wanted (faked is fine) and a settings screen.
- Code lives in the user's GitHub repo `cwissett-hub/ocean`.

## Builder's calls already made (by me, the designer — user may override)

- **Stack:** Rust + wgpu on the DX12 or Vulkan backend, using wgpu's
  ray-query feature for hardware RT. Because wgpu's RT support is
  experimental, the builder must verify early (first milestone) that ray
  queries work on this GPU/driver; if they do not, fall back to raw Vulkan
  (`ash`) with `VK_KHR_ray_query` for the RT passes, or DX12 DXR. The
  screen-space path must work regardless.
- **UI:** egui (via egui-wgpu) for the settings panel and overlay.
- **Build order:** get waves + sky + shading + foam on screen first (with
  screen-space reflections), measure, then add RT reflections/refraction,
  then particles, then the settings panel polish. Pixels on screen early.
- Default render resolution is the window size (no upscaler in v1).
- **RT is on by default and must fit the 12 ms budget** at defaults. Reduced-
  resolution rays with temporal accumulation/denoise are acceptable to get
  there; screen-space is the comparison/fallback mode, not the default.
- **If wgpu ray queries do not work** on this GPU/driver, stop and report to
  the user before taking the raw Vulkan/DXR detour; meanwhile continue on the
  screen-space path. Do not silently take on a second graphics API.
- **What RT reflects (user chose both):**
  - *Waves reflecting waves.* A near-field patch of the displaced wave surface
    around the camera (size/resolution tuned to budget, e.g. a few hundred
    metres) is rebuilt or refitted in the acceleration structure every frame,
    so crests reflect and occlude other crests. Rays that leave the patch fall
    back to the sky lookup. If this cannot fit in 12 ms, shrink the patch or
    its resolution before dropping it, and report the numbers to the user.
  - *A floating object.* One navigation buoy (simple modelled mesh, a few
    thousand triangles) in view, floating and bobbing using the wave height
    query — this also proves that query works for the future boat. It is in
    the RT scene, so it reflects in the water and the water's ray-traced
    reflections show it.
- **Refraction in v1** is screen-space plus analytic volumetric absorption and
  scattering (there is no seabed for refraction rays to hit). RT refraction is
  not in v1.
- **Sun-height slider range:** from just below the horizon (civil twilight,
  about −6°) up to midday. No night, moon or stars.

## Hooks for later (design for, don't build)

- **Camera** is a component that can later be driven by input (free flight);
  v1 just uses a fixed pose.
- **Wave queries:** CPU-side (or GPU readback) function returning water
  height, normal and velocity at any world point, so a boat can float and
  bob. v1 must include this function and a test that it matches the rendered
  surface.
- **RT scene** accepts arbitrary meshes (a boat) in its acceleration
  structure, not just the sky.
- **Particle emitters** are generic so a boat hull/wake can emit later.
- Boat wake on the surface (Kelvin wake) is out of scope for v1.

## Non-goals (v1)

- Fullscreen mode, start menus, sound.
- Land, coastline, shoreline waves, visible seabed, underwater view.
- True 3D fluid simulation (FLIP/SPH) of breaking waves.
- Path-traced caustics.
- Flying camera and the boat itself (the buoy is the only object in v1).

## Anti-goals (would make it a failure even if it "works")

- Looks like a game ocean: tiling repetition visible, plastic specular, flat
  blue colour, sine-wave regularity.
- Painted / texture-scrolled foam that ignores the wave shape.
- Drops below 60 fps at defaults on this laptop, or silently runs on the Intel
  iGPU.
- Over-engineering before anything is on screen.

## Left to the builder

- Exact spectrum parameters, cascade sizes, FFT resolution (e.g. 256²/512²).
- Sky model choice, tone mapping (e.g. AgX / ACES), bloom, exposure.
- RT ray budget (e.g. half-res rays + denoise/temporal accumulation, distance
  cutoff to sky lookup) — whatever hits the frame budget.
- Particle technique (billboards, sizes, counts) and how spray couples to foam.
- Mesh strategy for the ocean surface (projected grid, clipmap/LOD rings).
- Project layout, shader language (WGSL vs HLSL/GLSL via naga), testing of
  the non-visual parts (wave maths, height query). No heavy automated GUI
  testing.
