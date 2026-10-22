# Raft and Islands — Design

Date: 2026-10-22
Status: draft, awaiting approval
Builds on: `2026-10-21-ocean-sim-design.md` (v0.1.0) — everything there still holds unless
changed below.

## Intent

Turn the sunset ocean from something you watch into somewhere you go: sail a small driftwood
raft across the same physically based sea, between procedurally generated desert islands,
and look at it from several cameras. It should stay a calm, beautiful, slightly meditative
experience — leisurely sailing, not an arcade game.

## The picture

- A raft of a handful of lashed driftwood logs with a short mast and a square sail, riding
  the swell: it pitches, rolls and heaves with the waves.
- Islands appear over the horizon one at a time as you sail: low sandy dunes, sometimes a
  palm tree or a few. Nothing else on them.
- Around an island the water shoals: deep blue turns to turquoise over a pale sandy bottom,
  and a ring of foam marks where waves meet the beach.
- Same sunset sky, waves, whitecaps, spray, ray-traced reflections and settings panel as
  v1. Islands and the raft reflect in the water.

## Decided by the user

- The boat is a driftwood raft with a small sail.
- You sail through procedurally generated desert islands; an island is just sand and maybe
  a palm tree.
- Several camera options.
- Islands sparse, a couple of km apart, with occasional archipelago formations.
- A paddling mechanic supplements the sail.
- Accepted the playback: water physics like the buoy, wind-driven sail, keyboard steering,
  leisurely pace; chase / deck / high / free-fly cameras; endless procedural islands spread
  out; shallow-water colour and a foam ring at the shore; grounding on sand; islands in the
  RT scene; bow spray and a foam wake; buoy stays as a landmark; no breaking surf.

## Goals

1. **Raft.** A procedurally built driftwood raft mesh (logs, lashings, mast, square sail;
   a few thousand triangles). Floats with a multi-point buoyancy model on the GPU wave query
   (as the buoy, more points), so it pitches, rolls and heaves.
2. **Sailing.** The sail is driven by the sim's wind (speed and direction from the settings).
   Force from the apparent wind on the sail, with keel-like lateral resistance so the raft
   moves mostly where it points, plus water drag. Downwind is fastest; sailing into the wind
   barely moves. Typical speed 1–3 m/s at the default 5 m/s wind.
3. **Controls.** Turn left/right (rudder/steering oar); raise/lower the sail (sail up = full
   force, down = drift); **paddle** (hold a key) to supplement the sail — slow (my call:
   ~0.5–0.8 m/s on flat water), works in any direction including upwind, and is how you push
   off a beach. The sail stays the main way to travel. Exact keys are the builder's call but must be listed on screen in
   the settings panel help line and README.
4. **Cameras**, cycled with a key: **chase** (behind and above the raft, follows heading
   smoothly), **deck** (eye height on the raft, rides its motion), **high** (the v1 sunset
   view: ~18 m up, pitched down, now following the raft), **free fly** (detached; keyboard +
   mouse). Switching is instant or a short blend; no camera ever goes under the water surface
   or into an island.
5. **Islands.** Deterministic procedural placement over an unbounded world from a seed.
   **Sparse: lone islands roughly 2 km apart** (about 10–15 minutes of sailing), and
   **sometimes an archipelago**: a cluster of several islands 100–400 m apart that you can
   sail between. Usually one island or one cluster is in view. Distant islands fade into the
   sunset haze (aerial perspective) instead of popping in; my call: streaming radius ~4 km,
   with coarse far LODs. Each island: a sand heightfield with dune shapes (roughly 30–150 m
   across, up to ~8 m high), sloping beach into a sandy seabed; 0–a few palm trees
   (procedural trunk + fronds). Generated as they come within range, freed when far.
6. **Shallow water.** Water depth from the island/seabed height drives water colour
   (refraction of a visible sandy bottom, absorption by depth) and attenuates wave height
   approaching the shore so the beach isn't swamped. A foam band at the shoreline. No
   breaking surf.
7. **Grounding.** The raft cannot pass through islands: it runs aground on the beach and can
   be sailed or turned off again.
8. **RT and particles.** Islands (terrain + palms) and the raft go into the ray-tracing scene
   so they reflect in the water; the raft's bow throws spray and its wake leaves foam via the
   existing `SpawnRequest` / foam-deposit paths.
9. **Buoy** stays, now world-fixed at a spot near the start, as a landmark.

## Constraints

- Same machine and the same performance target as v1: **GPU frame p95 ≤ 12 ms** at the
  default window (1600×900 physical on this laptop), RT on, while sailing near an island with
  default settings. Measured with the existing `--bench` (extended so it can run with the raft
  sailing a fixed course past an island).
- The world must stay smooth after hours of sailing far from the start: no jitter from f32
  world coordinates (raft speed ~3 m/s ≈ 10 km/h). Builder chooses how (e.g. camera-relative
  rendering or periodic origin rebasing).
- The wave/foam textures already tile every 1024 m and every system that follows the camera
  (ocean mesh, RT patch, particle emitter) must keep working wherever the raft goes.
- Public repo rules unchanged: nothing employer-related, noreply identity, pre-push guard.
- No heavy GUI tests; logic in pure functions tested headlessly; visuals judged by eye with
  screenshots. Subagents never launch the app window.

## Non-goals

- Breaking surf, waves rolling up the beach, wet-sand interaction, swimming.
- Anything on islands beyond sand and palms (rocks, grass, buildings, wildlife, footprints).
- Sounds, a score, goals, inventory, saving.
- Weather or time-of-day changes beyond the existing sun sliders.
- Destructible raft, capsizing, crew.

## Anti-goals

- Feels like an arcade boat: snappy, fast, turning on the spot.
- The raft floats on top of the water like a sticker, or sinks/flickers through the surface.
- Islands look like green-screen blobs: hard waterline, no shoaling, no foam ring.
- Visible pop-in of islands near the camera, or a hitch when one generates.
- Drops below 60 fps while sailing.

## Builder's calls already made (by me; user may override)

- Keep Rust + wgpu and the v1 architecture; islands and raft are new modules.
- Raft physics on the CPU using the async GPU wave query (more sample points than the buoy).
- Island terrain as CPU-generated meshes (heightfield → mesh) built on a worker thread, one
  BLAS per island, TLAS instances per island/palm/raft; the ocean shader gets a coarse
  "seabed depth" texture (camera-following) for shoaling, colour and wave attenuation.
- Default start: raft at the origin with an island visible ahead under the sunset glitter
  path; default camera = chase.

## Left to the builder

- Island noise functions, spacing, palm geometry, sand material.
- Sail/hull force coefficients (tune for the leisurely 1–3 m/s feel).
- Camera smoothing and free-fly controls.
- How depth reaches the ocean shaders and how far wave attenuation extends.
- Island streaming distances, LOD, and generation threading.

## Change 2026-10-22 (user): shallow sea with a visible seabed everywhere

The whole sea is about **20–30 m deep** with a sandy seabed visible through the water
everywhere (gently undulating, a few metres of relief), not just around islands; islands rise
from it. Water shading always refracts the seabed (fading with depth); waves use finite-depth
dispersion for the mean depth (ω² = g·k·tanh(k·h), h ≈ 25 m). Shore wave attenuation still only
applies where it is shallower than ~10 m.
