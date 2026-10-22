// Per-frame uniforms shared by the G-buffer and shading passes (group 0, binding 0).
// Must match FrameUniforms in src/render/frame.rs.

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,     // xyz, time
    sun_dir: vec4<f32>,     // xyz
    sizes: vec4<f32>,       // cascade patch sizes, m
    lambda_min: vec4<f32>,  // shortest wavelength in each cascade's band, m
    grid: vec4<f32>,        // base cell m, levels, grid quads per side, 0
    slope_var: vec4<f32>,   // mean-square slope of each cascade
    misc: vec4<f32>,        // unresolved slope variance, significant wave height m, exposure, wind direction rad
    resolution: vec4<f32>,  // w, h, 1/w, 1/h of the internal render targets
    seabed: vec4<f32>,      // seabed map: local corner x, z, texel m, resolution (seabed_common.wgsl)
    jitter: vec4<f32>,      // TAA sub-pixel offset of this frame's projection, pixels (x right, y down)
    wake: vec4<f32>,        // raft wake grid: origin cell x, z, cell m, resolution (wake_common.wgsl)
    raft_inv: mat4x4<f32>,  // world → raft frame (wood grain, wet line)
    raft_wet: array<vec4<f32>, 3>, // wet line per raft sample column (scene::wetness), raft-frame y
};

@group(0) @binding(0) var<uniform> frame: Frame;

const N_TEXELS: f32 = 256.0;

// Texel (x, z) of a cascade holds world point (x, z)·L/N; texel centres sit at +0.5 texel.
fn cascade_uv(p: vec2<f32>, c: u32) -> vec2<f32> {
    return p / frame.sizes[c] + vec2<f32>(0.5 / N_TEXELS);
}

fn world_from_depth(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(pixel.x * frame.resolution.z * 2.0 - 1.0, 1.0 - pixel.y * frame.resolution.w * 2.0);
    let h = frame.inv_view_proj * vec4<f32>(ndc, depth, 1.0);
    return h.xyz / h.w;
}

fn view_dir_for_pixel(pixel: vec2<f32>) -> vec3<f32> {
    return normalize(world_from_depth(pixel, 0.5) - frame.cam_pos.xyz);
}

// ---- The swash (mirrors world::swash) ----------------------------------------------------
// Water running up the beach and back: amp·(½ + ½·sin θ) above the still waterline, θ = ω·t
// plus a phase that wanders along the shore. Analytic, so sand knows when it was last wet.
const SWASH_OMEGA: f32 = 0.9110619;
const SWASH_DRY_TAU: f32 = 4.0;
const SWASH_DAMP: f32 = 0.35;
const SWASH_DAMP_BAND: f32 = 0.25;
const SWASH_PI: f32 = 3.14159265;

fn swash_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    var r = array<f32, 4>();
    for (var k = 0; k < 4; k++) {
        let c = i + vec2<f32>(f32(k & 1), f32(k >> 1));
        let q = fract(c * vec2<f32>(0.1031, 0.1030));
        let s = q + dot(q, q.yx + 33.33);
        r[k] = fract((s.x + s.y) * s.x);
    }
    return mix(mix(r[0], r[1], u.x), mix(r[2], r[3], u.x), u.y);
}

fn swash_phase(p: vec2<f32>) -> f32 {
    return frame.cam_pos.w * SWASH_OMEGA + swash_noise(p * 0.04) * 6.0;
}

fn swash_amplitude() -> f32 {
    return 0.08 + 0.06 * clamp(frame.misc.y, 0.0, 2.0);
}

fn swash_height(p: vec2<f32>) -> f32 {
    return swash_amplitude() * (0.5 + 0.5 * sin(swash_phase(p)));
}

// Wetness (0 dry .. 1 soaked) of beach `ground` m above still water at swash phase `theta`.
fn sand_wetness(ground: f32, theta: f32, amp: f32) -> f32 {
    if (ground <= 0.0) {
        return 1.0;
    }
    if (ground >= amp) {
        return SWASH_DAMP * smoothstep(amp + SWASH_DAMP_BAND, amp, ground);
    }
    let a = asin(clamp(2.0 * ground / amp - 1.0, -1.0, 1.0));
    let tau = 2.0 * SWASH_PI;
    let th = theta - floor(theta / tau) * tau;
    let start = a - floor(a / tau) * tau;
    let end = SWASH_PI - a;
    var covered = th >= start && th <= end;
    if (start > end) {
        covered = th >= start || th <= end;
    }
    if (covered) {
        return 1.0;
    }
    let d = th - end;
    let since = (d - floor(d / tau) * tau) / SWASH_OMEGA;
    return max(exp(-since / SWASH_DRY_TAU), SWASH_DAMP);
}
