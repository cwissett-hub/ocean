# Raft and Islands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sail a wind- and paddle-driven driftwood raft across the v1 ocean between endless procedural desert islands (lone islands ~2 km apart plus occasional archipelagos), with four cameras, shoaling shallows, grounding, RT reflections of everything, and bow spray/wake — at ≤ 12 ms GPU p95.

**Architecture:** Gameplay state (raft, cameras, island placement, origin) is plain Rust in `src/world/` and `src/scene/`, tested headlessly. All world positions the renderer sees are *local* f32 coordinates relative to a `WorldOrigin` that rebases in whole multiples of 1024 m (the wave/foam period, so the sea looks identical after a rebase); islands are generated from *world* f64 coordinates. The v1 `BuoyPass` becomes a generic `ObjectPass` that draws any number of mesh instances with a material id. Islands are CPU-generated meshes built on a worker thread, drawn through `ObjectPass`, added to the TLAS, and summarised for the GPU as a camera-following seabed-height texture that drives shallow-water colour, shoreline foam and wave attenuation everywhere waves are sampled.

**Tech Stack:** unchanged from v1 (Rust 1.96, wgpu 30.0.1 Vulkan ray query, winit 0.30.13, egui 0.37, glam 0.34). glam `DVec2` for world coordinates.

**Spec:** `docs/superpowers/specs/2026-10-22-raft-islands-design.md` (builds on `docs/superpowers/specs/2026-10-21-ocean-sim-design.md`)

## Global Constraints

- GPU frame **p95 ≤ 12 ms** at the default window (1600×900 physical here), RT on, defaults, while sailing past an island — measured with `--bench` (foreground; see Task 10).
- No jitter after hours of sailing: renderer sees only local coordinates; origin rebases by multiples of **1024 m** when the raft is > **4096 m** from it.
- Islands: lone islands ≈ **2 km** apart; ~**15 %** of island sites are archipelagos of **3–6** islands **100–400 m** apart; island radius **15–75 m**, dune height ≤ **8 m**; streaming radius **4 km**, coarse LOD beyond **1.5 km**; distant islands fade by aerial perspective (existing `aerial()`), never pop in within 2 km.
- Raft: sail speed **1–3 m/s** downwind at the default 5 m/s wind; upwind nearly 0; paddle **0.5–0.8 m/s** in any direction; turn rate ≤ **15°/s** (no turning on the spot).
- Cameras: chase / deck / high / free fly, cycled with **C**; no camera below the water surface or inside an island.
- Controls (shown in the settings help line and README): **A/D** steer, **W** paddle forward, **S** paddle back, **Space** sail up/down, **C** camera; free fly: **WASD** move, **Q/E** down/up, **right-drag** look, **Shift** fast (raft controls inactive in free fly).
- Same public-repo rules (noreply identity, pre-push guard, nothing employer-related), no heavy GUI tests, subagents never launch the app, `cargo clippy --all-targets -- -D warnings`, `cargo fmt`, and `cargo test` before every commit; push after each task.
- Visual steps are verified by the main session with screenshots (`--freeze-time`, `--set`), launching one window at a time.

## Review Focus

1. **Rebasing mid-motion** (raft crosses 4096 m while spray, foam, temporal history and the wave query are in flight): nothing visibly jumps; pending query results are applied in the new frame of reference — test in Task 1 (`rebase_preserves_wave_phase`) and Task 9 (`raft_state_survives_rebase`).
2. **Grounded and stuck** (beached on the windward side, sail up, wind onshore): paddling back or turning + paddling always frees the raft within a few seconds — test in Task 2 (`paddle_frees_a_beached_raft`).
3. **Island generation spikes** (several islands enter range at once, e.g. an archipelago, or free-fly at speed): no frame stalls on the main thread; meshes arrive over later frames — test in Task 6 (`streaming_plan_limits_work_per_frame`).
4. **Camera clipping** near steep dunes, in a trough, or in free fly diving at the sea: camera clamped above water and terrain — test in Task 3 (`no_camera_mode_goes_below_water_or_terrain`).
5. **Shoreline consistency**: the raft's water height (wave query) uses the same shallow-water wave attenuation as the drawn surface, so the raft neither floats above nor sinks into the shallows — test in Task 7 (`query_attenuates_like_the_surface`).

---

## File Structure

```
src/world/mod.rs          pub mod origin; pub mod islands;
src/world/origin.rs       WorldOrigin: local<->world, rebase decision
src/world/islands.rs      IslandDesc, placement (cells, archipelagos, start island), height fn
src/world/terrain_mesh.rs island heightfield -> mesh (LODs), palm mesh, streaming plan
src/world/streaming.rs    IslandStreamer: worker thread, ready meshes, unload
src/scene/raft.rs         RaftBody, RaftControls, forces, grounding, sample points
src/scene/raft_mesh.rs    build_raft_mesh()
src/scene/cameras.rs      CameraMode, CameraController (chase/deck/high/free fly), clamp
src/scene/mesh.rs         MeshData (shared vertex format), material ids
src/render/object_pass.rs ObjectPass (replaces buoy_pass.rs): instances, materials, opaque pre-pass
src/render/seabed.rs      SeabedMap: camera-following seabed height texture (GPU compute)
src/input.rs              KeyState -> RaftControls / FreeFlyInput (pure)
shaders/object_gbuffer.wgsl   (replaces buoy_gbuffer.wgsl)
shaders/materials.wgsl        (replaces buoy_common.wgsl) albedo/emission per material id
shaders/seabed.wgsl           seabed height map compute (mirrors world::islands::island_height)
shaders/seabed_common.wgsl    seabed sampling + wave attenuation, included by ocean/rt_patch/query/foam/shade
```

Material ids (G-buffer `material.g = id / 8`, decode `round(g * 8)`; `material.b` = per-vertex param): 0 water, 1 buoy, 2 raft wood, 3 sail, 4 sand, 5 palm trunk, 6 palm frond. TLAS custom data uses the same ids; patch stays 0 (water).

---

### Task 1: World origin and island placement (pure)

**Files:** Create `src/world/mod.rs`, `src/world/origin.rs`, `src/world/islands.rs`; Modify `src/lib.rs`.

**Interfaces — Produces:**
- `pub const REBASE_DISTANCE_M: f64 = 4096.0; pub const REBASE_STEP_M: f64 = 1024.0;`
- `pub struct WorldOrigin { pub offset: DVec2 }` with `to_world(local: Vec2) -> DVec2`, `to_local(world: DVec2) -> Vec2`, `rebase_for(local_focus: Vec2) -> Option<Vec2>` (returns the local shift `s`, a multiple of 1024 m per axis, such that the focus moves to within 1024 m of the new origin; `None` while |focus| ≤ 4096), `apply(&mut self, shift: Vec2)` (offset += shift).
- `pub struct IslandDesc { pub id: u64, pub center: DVec2, pub radius: f32, pub height: f32, pub palms: u32, pub seed: u64 }`
- `pub const CELL_M: f64 = 2048.0; pub const WORLD_SEED: u64 = 0x0CEA_2026;`
- `pub fn islands_in_cell(seed: u64, cell: IVec2) -> Vec<IslandDesc>` (deterministic; cell (0,0) always contains the start island)
- `pub fn islands_near(seed: u64, world: DVec2, radius_m: f64) -> Vec<IslandDesc>`
- `pub fn start_island(seed: u64) -> IslandDesc` — 400 m from the world origin along `sun_direction(4°, 30°)` xz, radius 45 m, height 6 m, 2 palms.
- `pub fn island_height(d: &IslandDesc, world: DVec2) -> f32` — dune heightfield above sea level 0, continuing as a beach slope underwater to the sea floor `SEABED_M = -40.0` at ~2.5·radius; returns `-1000.0` (no seabed) beyond 3·radius.
- `pub fn seabed_height(islands: &[IslandDesc], world: DVec2) -> f32` — max over islands.

- [ ] **Step 1: Write failing tests** (in-module):
  - `placement_is_deterministic` — same seed+cell → identical lists; different seed → different.
  - `start_island_is_ahead_under_the_sun` — `start_island(WORLD_SEED)` centre within 1 m of 400·(sin30°, cos30°); `islands_in_cell(WORLD_SEED, (0,0))` contains it.
  - `lone_islands_are_about_two_km_apart` — over cells −8..8², taking each site's centroid: mean nearest-neighbour distance between sites in 1500–3000 m; no two sites closer than 700 m.
  - `archipelagos_exist_with_close_members` — 8–25 % of sites have 3–6 islands; members pairwise 100–400 m apart (nearest member ≤ 400, all ≥ 100).
  - `island_sizes_in_range` — radius 15–75, height ≤ 8.
  - `height_profile_is_a_beach` — `island_height` at centre > 1 m, at radius ≈ 0 (±0.5 m), decreasing monotonically along a ray from the shore to 2.5·radius, reaches ≤ SEABED_M + 1, returns −1000 past 3·radius.
  - `rebase_only_beyond_threshold_and_in_whole_periods` — `rebase_for((4000, 0))` is None; `rebase_for((5000,-6000))` = Some(s) with s/1024 integral and |focus − s| ≤ 1024 per axis.
  - `rebase_preserves_wave_phase` — for a `CpuOcean` field, sampling at local p and at p − s after a rebase by s gives identical heights (every cascade size divides 1024).
- [ ] **Step 2:** `cargo test --lib world` → FAIL (missing items).
- [ ] **Step 3: Implement.** Placement per cell from `Pcg32::new(seed ^ hash(cell))`: 60 % chance of a site, jittered within the cell's inner 60 %; a site is an archipelago with 15 % probability (3–6 islands by rejection sampling within 600 m, min 100 m apart, each within 400 m of another), else one island. Height: radial base `h0·(1 − (r/R)²)` blended with 2-octave value-noise dunes (amplitude 0.4·h0, scaled by `smoothstep(R, 0.3R, r)`), and for r > R a beach `−(r−R)·slope` with slope reaching SEABED_M at 2.5R. Deterministic noise from `seed`.
- [ ] **Step 4:** tests PASS. **Step 5:** commit + push `feat: world origin and procedural island placement`.

---

### Task 2: Raft physics (pure)

**Files:** Create `src/scene/raft.rs`; Modify `src/scene/mod.rs`.

**Interfaces — Consumes:** `WaterSample` (waves::query). **Produces:**
- `pub struct RaftControls { pub steer: f32 /* -1..1, +right */, pub paddle: f32 /* -1..1 */, pub sail_up: bool }`
- `pub const RAFT_SAMPLE_OFFSETS: [Vec3; 9]` — 3×3 grid on the 3.0 m (long) × 2.4 m deck, local, y = 0 waterline.
- `pub struct RaftBody { pub position: Vec3, pub velocity: Vec3, pub rotation: Quat, pub angular_velocity: Vec3, pub heading: f32 /* rad, 0=+z, +=clockwise from above */ }`
- `RaftBody::at(position: Vec3, heading: f32) -> RaftBody`, `sample_points(&self) -> [Vec2; 9]` (world-aligned xz of the offsets rotated by heading), `forward(&self) -> Vec3`, `transform(&self) -> Mat4`, `speed(&self) -> f32`.
- `RaftBody::step(&mut self, water: &[WaterSample; 9], wind: Vec2 /* m/s, toward */, controls: &RaftControls, terrain: &dyn Fn(Vec2) -> f32, dt: f32)`
- `pub fn sail_force(apparent_wind: Vec2, heading: f32, sail_up: bool) -> Vec2` (square sail, auto-trimmed: drag-type force along the apparent wind, scaled by how much of the sail faces it).

- [ ] **Step 1: Write failing tests:**
  - `floats_flat_on_calm_water` (heave/roll settle < 2 cm / < 1° in 20 s, like the buoy).
  - `downwind_speed_is_leisurely` — wind (0,5) toward +z, heading 0, sail up, flat water, 120 s: forward speed 1.0–3.0 m/s.
  - `upwind_barely_moves` — heading π: speed over ground < 0.3 m/s toward the bow.
  - `reaching_moves_across_the_wind` — heading 90° with wind toward +z: mostly forward motion (|lateral| < 0.4·|forward|), speed > 0.4 m/s.
  - `paddle_speed_in_any_direction` — sail down, paddle 1, headings 0/90/180 with wind 5: speed 0.5–0.8 m/s (± wind drift ≤ 0.25 m/s).
  - `turning_is_gentle` — steer 1 for 10 s at sailing speed: yaw rate ≤ 15°/s and no full turn in < 24 s.
  - `cannot_pass_through_terrain` — terrain = beach rising from −3 m at z=10 to +1 m at z=20; sailing toward it never puts any sample point below the terrain height + 0.05 and the raft stops by z ≈ 16–19.
  - `paddle_frees_a_beached_raft` — after grounding (wind onshore, sail up), set sail down + paddle −1 (and alternatively steer 1 + paddle 1): within 8 s the raft is back in water ≥ 0.5 m deep.
  - `stays_finite_at_extremes` — wind 25 m/s, random water ±5 m, dt in {1/240, 1/30, 0.25}, 1000 steps; NaN samples ignored.
- [ ] **Step 2:** FAIL. **Step 3: Implement.** Buoyancy as the buoy (per-point spring-damper toward the water height, ω_heave 2.2, ζ 0.6) but per-unit-mass with the raft's larger footprint; horizontal: `F = sail_force + paddle_force + hull_drag`; hull drag in the hull frame with forward `c_f·v|v|` and lateral `c_l·v|v|` (c_l ≈ 12 c_f) so the raft goes where it points; yaw: `ω̇ = k_r·steer·(0.3 + |v_fwd|) − c_ω·ω`, capped by the 15°/s test; paddle thrust tuned to the 0.5–0.8 m/s band. Grounding: for each sample point, if `terrain(xz) > point.y − 0.05` push the body up/back (position correction along −forward·sign(v_fwd) and +y) and remove the velocity component into the slope, plus friction; paddling reverse still applies thrust.
- [ ] **Step 4:** PASS. **Step 5:** commit + push `feat: raft physics with sail, paddle and grounding`.

---

### Task 3: Cameras (pure)

**Files:** Create `src/scene/cameras.rs`, `src/input.rs`; Modify `src/scene/mod.rs`, `src/lib.rs`.

**Interfaces — Consumes:** `Camera` (scene::camera), `RaftBody`. **Produces:**
- `pub enum CameraMode { Chase, Deck, High, FreeFly }` with `next(self) -> CameraMode` (Chase → Deck → High → FreeFly → Chase).
- `pub struct FreeFlyInput { pub move_local: Vec3 /* x right, y up, z forward, each -1..1 */, pub look_delta: Vec2 /* rad */, pub fast: bool }`
- `pub struct CameraController { pub mode: CameraMode, … }` with `new(raft: &RaftBody) -> Self`, `update(&mut self, dt: f32, raft: &RaftBody, sun_azimuth_deg: f32, free: &FreeFlyInput, ground: &dyn Fn(Vec2) -> f32 /* max(water estimate, terrain) at xz */) -> Camera`.
- `pub fn clamp_above(pos: Vec3, ground_height: f32, clearance: f32) -> Vec3`
- `src/input.rs`: `pub struct KeyState` (set of pressed `winit::keyboard::KeyCode` + right-mouse + accumulated mouse delta) with `raft_controls(&self, sail_up: bool) -> RaftControls`, `free_fly(&mut self) -> FreeFlyInput` (consumes the mouse delta).

- [ ] **Step 1: Write failing tests:** `chase_sits_behind_and_above` (8–12 m behind along −forward, 3–5 m up, looking at the raft); `chase_follows_heading_smoothly` (heading jumps 90°: camera yaw moves < 30° in the first 0.25 s and > 80° within 3 s); `deck_rides_the_raft` (eye 1.5–1.8 m above the deck, forward = raft forward, inherits roll/pitch); `high_view_keeps_raft_in_frame` (raft position projects inside NDC ±0.8 for sun azimuth 0/30/200); `free_fly_moves_and_looks` (move_local z=1 for 1 s → ~8 m forward; fast → ×4; look delta changes forward); `no_camera_mode_goes_below_water_or_terrain` (ground fn returning 5 m everywhere: every mode's camera y ≥ 5 + clearance); `mode_cycles_in_order`; `key_state_maps_controls` (A → steer −1, D → +1, W/S paddle ±1, both A+D → 0).
- [ ] **Step 2:** FAIL. **Step 3: Implement** (chase: critically damped spring on yaw and position, ω = 3 rad/s; high: v1 framing — 18 m up, pitch −25°, yaw = sun azimuth, placed 30 m behind the raft along that yaw; free fly: 8 m/s, Shift ×4). **Step 4:** PASS. **Step 5:** commit + push `feat: chase, deck, high and free-fly cameras`.

---

### Task 4: Generic objects, raft mesh, raft on the water

**Files:** Create `src/scene/mesh.rs`, `src/scene/raft_mesh.rs`, `src/render/object_pass.rs`, `shaders/object_gbuffer.wgsl`, `shaders/materials.wgsl`; Delete `src/render/buoy_pass.rs`, `shaders/buoy_gbuffer.wgsl`, `shaders/buoy_common.wgsl`; Modify `src/scene/buoy.rs` (build_buoy_mesh returns `MeshData`), `src/render/frame.rs`, `src/render/rt_scene.rs` (buoy geometry via ObjectPass), `shaders/shade.wgsl`, `shaders/surfaces.wgsl`, `shaders/reflect_rt.wgsl` (material decode), `src/gpu/shaders.rs`, `src/main.rs`.

**Interfaces — Produces:**
- `pub mod material { pub const WATER: u32 = 0; BUOY = 1; RAFT_WOOD = 2; SAIL = 3; SAND = 4; PALM_TRUNK = 5; PALM_FROND = 6; }`
- `pub struct MeshData { pub vertices: Vec<[f32; 8]> /* pos3, normal3, material, param */, pub indices: Vec<u32> }` with `append(&mut self, other: &MeshData)` and `bounds() -> (Vec3, Vec3)`.
- `pub fn build_raft_mesh() -> MeshData` — 7 logs (irregular radius 0.14–0.2 m, length 2.8–3.2 m, slight bend) side by side, two cross lashings, mast 2.6 m, square sail 1.8×1.6 m with billow (param = sail v for translucency gradient; wood param = per-log tint 0..1).
- `pub struct MeshHandle(pub u32)`; `ObjectPass::new(d, queue, lib, frame_layout)`, `add_mesh(&mut self, d, &MeshData) -> MeshHandle`, `set_instances(&mut self, items: &[(MeshHandle, Mat4)])` (per frame; transforms and previous transforms tracked by slot), `record_opaque(enc, g, frame_group)`, `draw_gbuffer(pass, frame_group)`, `mesh_geometry(h) -> (&Buffer, &Buffer, u32 /*index count*/, u32 /*vertex count*/)` (for BLAS; buffers have `BLAS_INPUT`).
- G-buffer material encoding as in File Structure; `shade.wgsl` dispatches on id: water (existing), buoy (existing look via `materials.wgsl`), raft wood (grey-brown driftwood, rough), sail (off-white cloth with back-lit transmission `∝ max(dot(view, sun), 0)^4` — the sail glows against the low sun), sand (Task 6), palms (Task 6).
- Raft in the frame: `Renderer` owns `RaftBody` (start: 20 m behind the world origin facing the start island), steps it each frame with the 9 query results (extend the `WaveQuery` request to raft 9 + buoy 5 + camera 1 points), wind from params, controls passed into `render` (new field on a `FrameInput { controls: RaftControls, free: FreeFlyInput, camera_cycle: bool }` argument replacing nothing else), and the camera from `CameraController` (default Chase). Terrain closure returns −1000 until Task 6.
- [ ] **Step 1: Write failing tests:** `raft_mesh_is_sane` (indices in range, no degenerate tris, unit normals, footprint ≈ 3×2.4 m, mast top 2.4–3.0 m, < 8000 tris, materials only wood/sail); `buoy_mesh_still_sane` (port of v1 test to MeshData, material BUOY); `mesh_append_offsets_indices`; shader library test still validates every shader (new/renamed files registered).
- [ ] **Step 2:** FAIL. **Step 3: Implement** ObjectPass (one vertex/index buffer per mesh, one model uniform slot per instance via dynamic offsets, up to 512 instances), migrate the buoy, wire raft + controller + key handling (`main.rs` feeds `KeyState`; C cycles; raft keys inactive in FreeFly).
- [ ] **Step 4: Verify (main session):** `cargo run --release -- --set show_overlay=false` — raft visible in chase view, bobbing; W/Space make it move (manual check by the user is fine; screenshot with `--freeze-time` for still framing). Bench defaults still ≤ 12 ms.
- [ ] **Step 5:** commit + push `feat: driftwood raft on the water with generic object rendering`.

---

### Task 5: Settings, help, camera/raft UI polish

**Files:** Modify `src/ui.rs`, `src/params.rs`, `README.md`.

**Interfaces:** `OceanParams` gains `camera_mode` is NOT a param (controller state); the panel gets a read-only "Raft" section (speed knots/m·s⁻¹, heading, sail up/down, grounded) via `OverlayInfo.raft: Option<RaftInfo>` (`pub struct RaftInfo { speed: f32, heading_deg: f32, sail_up: bool, grounded: bool, camera: &'static str }`) and the help line lists all controls.
- [ ] **Step 1: Failing test:** `raft_info_formats_heading_and_speed` (pure fn `raft_info(body, controls, grounded, mode) -> RaftInfo`: heading 0–360, speed ≥ 0).
- [ ] **Step 2–3:** FAIL → implement → PASS. **Step 4:** screenshot with `--panel`. **Step 5:** commit + push `feat: raft status and controls help`.

---

### Task 6: Islands on screen (terrain meshes, palms, streaming, sand)

**Files:** Create `src/world/terrain_mesh.rs`, `src/world/streaming.rs`; Modify `src/render/frame.rs`, `shaders/materials.wgsl`, `shaders/shade.wgsl`, `src/scene/raft.rs` callers (terrain closure).

**Interfaces — Consumes:** `IslandDesc`, `island_height`, `ObjectPass`. **Produces:**
- `pub fn island_mesh(d: &IslandDesc, lod: u32 /* 0 near, 1 far */) -> MeshData` — grid over the island's 3·radius square in *island-local* coordinates (centre at 0), LOD0 cell ≤ 0.75 m (cap 160² quads), LOD1 cell ≈ 4 m; vertices below SEABED_M + 2 dropped beyond the beach; material SAND, param = height (wet-sand darkening below +0.3 m).
- `pub fn palm_mesh() -> MeshData` (curved 6–9 m trunk, 8 fronds; trunk/frond materials) and `pub fn palm_transforms(d: &IslandDesc) -> Vec<Mat4>` (island-local, on dune ground ≥ 1.5 m, random lean/yaw).
- `pub struct StreamPlan { pub load: Vec<(IslandDesc, u32)>, pub unload: Vec<u64>, pub relod: Vec<(u64, u32)> }`, `pub fn plan_streaming(loaded: &HashMap<u64, u32>, visible: &[IslandDesc], focus: DVec2, max_new: usize) -> StreamPlan` (hysteresis: load < 4 km, unload > 4.5 km, LOD0 < 1.5 km, LOD1 > 1.7 km; at most `max_new` loads per frame, nearest first).
- `pub struct IslandStreamer` — owns a worker thread; `update(&mut self, focus_world: DVec2)`; `take_ready(&mut self) -> Vec<ReadyIsland>` (`ReadyIsland { id, desc, lod, mesh: MeshData, palms: Vec<Mat4> }`), `terrain_height(&self, world: DVec2) -> f32` (CPU `seabed_height` over loaded descs).
- Renderer: uploads ready meshes into `ObjectPass` (≤ 2 per frame), instances = island transform from `WorldOrigin::to_local(desc.center)`; raft terrain closure = `streamer.terrain_height`; camera ground fn uses it too.
- [ ] **Step 1: Failing tests:** `island_mesh_matches_height_fn` (every vertex y within 1 cm of `island_height`), `lod1_is_much_smaller` (< 1/10 the triangles), `palms_stand_on_dunes`, `streaming_loads_nearest_first_and_unloads_far`, `streaming_plan_limits_work_per_frame` (an archipelago of 6 entering at once: ≤ max_new loads per call, rest on later calls), `hysteresis_prevents_thrash` (focus oscillating ±50 m around 4 km: no load/unload churn).
- [ ] **Step 2–3:** FAIL → implement → PASS.
- [ ] **Step 4: Shading:** sand albedo (0.78, 0.66, 0.48) diffuse with slight sheen, wet darkening near the waterline; trunk/frond materials; aerial perspective as everything else. **Verify (main session):** start island visible ahead under the glitter path; free-fly over it; no hitch when an archipelago streams in (watch overlay max ms with `--set` wind defaults while free-flying to a known archipelago cell — find one via a small `examples/find_archipelago.rs`).
- [ ] **Step 5:** commit + push `feat: procedural desert islands with palms and streaming`.

---

### Task 7: Shallow water (seabed map, shoaling colour, shore foam, wave attenuation)

**Files:** Create `src/render/seabed.rs`, `shaders/seabed.wgsl`, `shaders/seabed_common.wgsl`, `tests/gpu_seabed.rs`; Modify `shaders/ocean_gbuffer.wgsl`, `shaders/rt_patch.wgsl`, `shaders/wave_query.wgsl`, `shaders/foam.wgsl`, `shaders/shade.wgsl`, `shaders/particles_sim.wgsl`, `src/waves/query.rs`, `src/render/frame.rs`.

**Interfaces — Produces:**
- `pub struct SeabedMap` — R32Float 1024² covering 4096 m (4 m texels) snapped to 64 m around the camera; `update(enc, islands: &[(Vec2 /*local centre*/, IslandDesc)])` recomputes only when the snapped centre or island set changes; `view()`, `params()` (origin, texel) for shaders.
- WGSL `seabed_height(p_local)` (bilinear) and `wave_attenuation(depth) = smoothstep(0.0, 10.0, depth)` in `seabed_common.wgsl`; every displacement sample (ocean mesh, RT patch, wave query, foam Jacobian, particle emission/landing) multiplies horizontal and vertical displacement by it; shade uses depth for a visible sandy bottom (sand albedo lit by in-water light through `exp(−(σa+σs)·2·depth)`), turquoise shoaling, and a shore foam band `smoothstep(1.5, 0.2, depth)` broken by the foam noise.
- CPU mirror: `pub fn wave_attenuation(depth: f32) -> f32` in `world::islands` (same formula).
- [ ] **Step 1: Failing tests:** `gpu_seabed_matches_cpu` (headless: SeabedMap vs `seabed_height` at 200 points within 5 cm, island set incl. an archipelago); `query_attenuates_like_the_surface` (headless: WaveQuery height over a 1 m-deep shoal ≤ 15 % of the open-water height at the same wave phase, and equals CPU `invert_horizontal` of attenuated displacement within 1 cm); `attenuation_is_monotonic_and_bounded` (pure).
- [ ] **Step 2–3:** FAIL → implement → PASS.
- [ ] **Step 4: Verify (main session):** turquoise shallows with visible sand around the start island, a soft foam ring at the beach, no waves riding up over the dunes, raft beaches without floating/sinking. Screenshot near the island (chase + high).
- [ ] **Step 5:** commit + push `feat: shoaling shallows, shore foam and wave attenuation near islands`.

---

### Task 8: Ray tracing for raft, islands and palms

**Files:** Modify `src/render/rt_scene.rs`, `shaders/reflect_rt.wgsl`, `src/render/frame.rs`.

**Interfaces:** `RtScene` gains `set_static_meshes(&mut self, meshes: &[(MeshHandle, (&Buffer,&Buffer,u32,u32))])` building a BLAS per mesh once (raft, buoy, palm, each island mesh as it streams in; freed on unload) and `record(enc, patch_origin, instances: &[(MeshHandle, Mat4, u32 /*material id*/)])` with `max_instances = 1024`. Shader: hit → material id via `instance_custom_data`; shading per id using `materials.wgsl` (sand normal ≈ up blended toward the triangle normal; wood/sail/palm approximations). Wave patch query unchanged (CULL_FRONT); objects queried two-sided by mask 0x02 as the buoy is now.
- [ ] **Step 1: Failing tests:** `instance_list_respects_capacity` (pure helper truncating by distance), `blas_registry_frees_unloaded_islands` (pure registry: add/remove ids → live set).
- [ ] **Step 2–3:** FAIL → implement → PASS.
- [ ] **Step 4: Verify (main session):** RT vs SSR A/B with `--freeze-time` near the start island: island, palms and raft reflected; bench ≤ 12 ms near the island.
- [ ] **Step 5:** commit + push `feat: ray-traced reflections of raft, islands and palms`.

---

### Task 9: Spray, wake, world-fixed buoy, origin rebasing wired

**Files:** Modify `src/render/particles.rs`, `shaders/particles_sim.wgsl`, `shaders/particles_draw.wgsl`, `shaders/foam.wgsl`, `src/render/frame.rs`, `src/scene/raft.rs`.

**Interfaces:**
- `pub const KIND_FOAM_MARK: u32 = 2;` — a particle that deposits foam where it is on its first update and dies (never drawn); deposit grid becomes **1024² at 1 m** (`DEPOSIT_RES`/`DEPOSIT_CELL_M` changed in both shaders).
- `pub fn raft_emissions(body: &RaftBody, water: &[WaterSample; 9], dt: f32) -> Vec<SpawnRequest>` (pure): bow spray when forward speed > 0.6 m/s and the bow's vertical velocity relative to the water is < −0.4 m/s (count ∝ impact), wake foam marks along the stern every 0.5 m of travel.
- `ParticleSystem::shift(&mut self, enc, shift: Vec2)` (compute pass adding −shift to every live particle) and the buoy becomes world-fixed at world (−25, 0, 40) relative to the start (local via `WorldOrigin`).
- Rebase in `Renderer`: when `origin.rebase_for(raft.xz)` is Some(s): shift raft, camera controller, buoy, particles, pending wave-query bookkeeping (results are relative to the point they were requested at, so re-key by request id), patch origin, invalidate reflection history and `prev_view_proj` for one frame; foam/wave textures unchanged (period 1024).
- [ ] **Step 1: Failing tests:** `bow_spray_only_when_plunging_at_speed`, `wake_marks_every_half_metre`, `raft_state_survives_rebase` (pure: rebasing raft+camera+buoy by s and stepping equals stepping then rebasing, within 1e-4), `emissions_respect_request_budget` (through `pack_requests`).
- [ ] **Step 2–3:** FAIL → implement → PASS.
- [ ] **Step 4: Verify (main session):** bow spray and a foam wake behind the raft at 10 m/s wind; teleport test via `--set`-free dev flag `--start-at x,z` (world, e.g. 9000,0) to sail across a rebase boundary: no visible jump.
- [ ] **Step 5:** commit + push `feat: raft spray and wake, world-fixed buoy, origin rebasing`.

---

### Task 10: Bench course, budget gate, docs, release

**Files:** Modify `src/main.rs`, `src/bench.rs`, `README.md`, `Cargo.toml` (version 0.2.0).

- `--bench N` now runs a scripted course: raft starts 150 m from the start island, sail up, steering to pass ~40 m off its beach, chase camera; plus `--bench-camera high|deck|chase` (default chase).
- [ ] **Step 1: Failing test:** `bench_course_controls_are_scripted` (pure fn `course_controls(t: f32) -> RaftControls` deterministic, sail up, steer within ±0.3).
- [ ] **Step 2–3:** FAIL → implement → PASS.
- [ ] **Step 4:** `cargo fmt`, clippy, full `cargo test` (record the total, not the doc-test line); release build; dist; foreground bench for chase and high: **p95 ≤ 12 ms** else reduce (island LOD0 cell → 1 m, palm count, RT instance distance) and report.
- [ ] **Step 5:** README: raft controls, cameras, islands; commit, push, tag **v0.2.0** (with the user's go-ahead on tagging).

---

### Task 11: Water look polish (user request, 2026-10-22)

User feedback on v1/v2 screenshots: the water "looks too opaque and sort of matte". After Task 10, diagnose by A/B screenshots against real sunset-sea references and fix — candidates to investigate, not a fixed recipe:

- Reflection strength/sharpness: is the sky reflection too weak or blurred (roughness from unresolved slope variance too high; footprint fade too aggressive; half-res reflection upsample softening it)?
- Transparency: near-camera water should read as clear — deep-water single scattering (`deep_water_upwelling`) may be too bright/flat; add view-dependent transmission (light paths through wave crests, Fresnel-weighted refraction of the dark depths) and contrast between faces facing the sky vs away.
- Specular: sharper, brighter sun glints at the near field (micro-facet variance per pixel), HDR bloom on glints.
- Normal detail: finer capillary detail near the camera (a detail normal map or the finest cascade not faded so early).

Verification: side-by-side screenshots before/after shown to the user; bench still ≤ 12 ms p95. Tests for any pure functions changed (e.g. roughness/footprint functions).
