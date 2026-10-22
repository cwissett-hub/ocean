// One iWave step of the raft's own waves (mirrors render::wake::iwave_step). The grid is
// addressed toroidally: texel = world cell mod RES, so the window follows the raft without
// copying; cells that were outside last step's window start flat. The hull is a pressure
// patch (m of water) pushed into the vertical derivative; a sponge near the window's edge
// lets waves leave instead of reflecting.

struct WakeParams {
    origin: vec2<i32>,       // window's lower-left cell, this step
    prev_origin: vec2<i32>,  // and last step's
    raft: vec2<f32>,         // hull centre, local m
    heading: f32,            // compass rad
    dt: f32,
    pressure: f32,           // head under the hull, m
    damping: f32,            // 1/s
    half_size: vec2<f32>,    // hull half beam (x) and half length (z), m
};

@group(0) @binding(0) var state_in: texture_2d<f32>;
@group(0) @binding(1) var state_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> wp: WakeParams;

const RES: i32 = 256;
const CELL_M: f32 = 0.25;
const RADIUS: i32 = 6;
const G: f32 = 9.81;
const SPONGE_CELLS: f32 = 24.0;

fn texel(cell: vec2<i32>) -> vec2<i32> {
    return ((cell % RES) + RES) % RES;
}

fn in_window(cell: vec2<i32>, origin: vec2<i32>) -> bool {
    let r = cell - origin;
    return all(r >= vec2<i32>(0)) && all(r < vec2<i32>(RES));
}

// Last step's (height, previous height) at a world cell; flat outside last step's window.
fn state(cell: vec2<i32>) -> vec2<f32> {
    if (!in_window(cell, wp.prev_origin)) {
        return vec2<f32>(0.0);
    }
    return textureLoad(state_in, texel(cell), 0).rg;
}

// Pressure head of the hull at a world cell: a soft-edged box the shape of the raft.
fn hull(cell: vec2<i32>) -> f32 {
    let p = (vec2<f32>(cell) + 0.5) * CELL_M - wp.raft;
    let c = cos(wp.heading);
    let s = sin(wp.heading);
    // Into the hull frame (+z forward = (sin h, cos h)).
    let local = vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
    let edge = 0.6;
    let bx = smoothstep(wp.half_size.x, wp.half_size.x - edge, abs(local.x));
    // Sharp entry at the bow, long run-out to the stern (strong bow wave, weak stern wave).
    let l = wp.half_size.y;
    let bz = smoothstep(l, l - edge, local.y) * smoothstep(-l, 0.3 * l, local.y);
    return wp.pressure * bx * bz;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = vec2<i32>(id.xy);
    if (t.x >= RES || t.y >= RES) {
        return;
    }
    // The world cell this texel holds in the current window.
    let cell = wp.origin + texel(t - wp.origin);
    var vd = 0.0;
    let side = 2 * RADIUS + 1;
    for (var j = -RADIUS; j <= RADIUS; j++) {
        for (var i = -RADIUS; i <= RADIUS; i++) {
            let n = cell + vec2<i32>(i, j);
            vd += kernel[(j + RADIUS) * side + i + RADIUS] * (state(n).x + hull(n));
        }
    }
    let here = state(cell);
    let rel = vec2<f32>(cell - wp.origin) + 0.5;
    let edge = min(min(rel.x, rel.y), min(f32(RES) - rel.x, f32(RES) - rel.y));
    let a = (wp.damping + 6.0 * smoothstep(SPONGE_CELLS, 0.0, edge)) * wp.dt;
    // Verlet with damping on the velocity (h − prev), not the height.
    let next = here.x * (2.0 - a) - here.y * (1.0 - a) - G * wp.dt * wp.dt * vd / CELL_M;
    textureStore(state_out, t, vec4<f32>(next, here.x, 0.0, 0.0));
}
