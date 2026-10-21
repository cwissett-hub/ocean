// Ocean surface: nested camera-centred grids displaced by the four FFT cascades.
// Task 3: forward-shaded placeholder (sky gradient reflection, sun glint, deep-water colour).

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,     // xyz, time
    sun_dir: vec4<f32>,     // xyz
    sizes: vec4<f32>,       // cascade patch sizes, m
    lambda_min: vec4<f32>,  // shortest wavelength in each cascade's band, m
    grid: vec4<f32>,        // base cell m, levels, grid quads per side, 0
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var displacement: texture_2d_array<f32>;
@group(0) @binding(2) var derivatives: texture_2d_array<f32>;
@group(0) @binding(3) var samp: sampler;

const N: f32 = 256.0;

// --- shared helpers -----------------------------------------------------------------------

fn level_center(cam_xz: vec2<f32>, cell: f32) -> vec2<f32> {
    let snap = 2.0 * cell;
    return round(cam_xz / snap) * snap;
}

// Texel (x, z) of a cascade holds world point (x, z)·L/N; texel centres sit at +0.5 texel.
fn cascade_uv(p: vec2<f32>, c: u32) -> vec2<f32> {
    return p / frame.sizes[c] + vec2<f32>(0.5 / N);
}

// Cascades whose shortest waves are below the vertex spacing would alias; fade them out.
fn vertex_weight(c: u32, cell: f32) -> f32 {
    let l = frame.lambda_min[c];
    return 1.0 - smoothstep(0.25 * l, 0.5 * l, cell);
}

fn displacement_at(p: vec2<f32>, cell: f32) -> vec3<f32> {
    var d = vec3<f32>(0.0);
    for (var c = 0u; c < 4u; c++) {
        let w = vertex_weight(c, cell);
        if (w > 0.0) {
            d += w * textureSampleLevel(displacement, samp, cascade_uv(p, c), c, 0.0).xyz;
        }
    }
    return d;
}

fn sky(dir: vec3<f32>, sun_weight: f32) -> vec3<f32> {
    let sun = frame.sun_dir.xyz;
    let y = max(dir.y, 0.0);
    let horizon = vec3<f32>(1.0, 0.42, 0.16);
    let zenith = vec3<f32>(0.10, 0.09, 0.26);
    var c = mix(horizon, zenith, pow(y, 0.45));
    let sd = max(dot(dir, sun), 0.0);
    c += sun_weight * vec3<f32>(1.0, 0.5, 0.2) * pow(sd, 8.0) * 0.6;
    c += sun_weight * vec3<f32>(1.0, 0.85, 0.6) * pow(sd, 3000.0) * 30.0;
    return c;
}

// --- sky background -----------------------------------------------------------------------

struct SkyOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) i: u32) -> SkyOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: SkyOut;
    o.ndc = uv * 2.0 - 1.0;
    o.pos = vec4<f32>(o.ndc, 0.0, 1.0);
    return o;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    let h = frame.inv_view_proj * vec4<f32>(in.ndc, 0.5, 1.0);
    let dir = normalize(h.xyz / h.w - frame.cam_pos.xyz);
    return vec4<f32>(sky(dir, 1.0), 1.0);
}

// --- ocean surface ------------------------------------------------------------------------

struct OceanOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) grid_xz: vec2<f32>,
    @location(2) @interpolate(flat) level: u32,
};

@vertex
fn vs_ocean(@builtin(vertex_index) vi: u32, @builtin(instance_index) level: u32) -> OceanOut {
    let g = u32(frame.grid.z);
    let half = f32(g / 2u);
    let i = vi % (g + 1u);
    let j = vi / (g + 1u);
    let cell = frame.grid.x * exp2(f32(level));
    let ij = vec2<f32>(f32(i), f32(j)) - half;
    let p = level_center(frame.cam_pos.xz, cell) + ij * cell;

    // Outer band of each level morphs onto the next coarser grid (positions and cascade
    // weights), so its edge matches that level exactly.
    let m = smoothstep(0.75, 0.95, max(abs(ij.x), abs(ij.y)) / half);
    let cell_eff = cell * (1.0 + m);
    var d = displacement_at(p, cell_eff);
    let odd = vec2<f32>(f32(i & 1u), f32(j & 1u));
    if (m > 0.0 && odd.x + odd.y > 0.0) {
        let mid = 0.5 * (displacement_at(p - odd * cell, cell_eff) + displacement_at(p + odd * cell, cell_eff));
        d = mix(d, mid, m);
    }

    var o: OceanOut;
    o.world = vec3<f32>(p.x + d.x, d.y, p.y + d.z);
    o.pos = frame.view_proj * vec4<f32>(o.world, 1.0);
    o.grid_xz = p;
    o.level = level;
    return o;
}

@fragment
fn fs_ocean(in: OceanOut) -> @location(0) vec4<f32> {
    // Footprint first: derivatives must be taken before any discard.
    let footprint = max(length(dpdx(in.grid_xz)), length(dpdy(in.grid_xz)));
    if (in.level > 0u) {
        let fine = frame.grid.x * exp2(f32(in.level) - 1.0);
        let c = level_center(frame.cam_pos.xz, fine);
        let half = 0.5 * frame.grid.z * fine;
        if (all(abs(in.grid_xz - c) < vec2<f32>(half))) {
            discard;
        }
    }

    var der = vec4<f32>(0.0);
    for (var c = 0u; c < 4u; c++) {
        let texel = frame.sizes[c] / N;
        let w = 1.0 - smoothstep(texel, 4.0 * texel, footprint);
        if (w > 0.0) {
            der += w * textureSampleLevel(derivatives, samp, cascade_uv(in.grid_xz, c), c, 0.0);
        }
    }
    let n = normalize(vec3<f32>(-der.x / (1.0 + der.z), 1.0, -der.y / (1.0 + der.w)));

    let to_eye = frame.cam_pos.xyz - in.world;
    let dist = length(to_eye);
    let v = to_eye / dist;
    let ndv = max(dot(n, v), 0.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    let r0 = reflect(-v, n);
    // Rays reflected below the horizon would hit other waves (ray tracing handles that
    // later); here they see the horizon and cannot see the sun.
    let r = normalize(vec3<f32>(r0.x, max(r0.y, 0.0), r0.z));
    let sun = frame.sun_dir.xyz;
    let above = smoothstep(0.0, 0.03, r0.y);
    let glint = vec3<f32>(1.0, 0.75, 0.45) * pow(max(dot(r, sun), 0.0), 900.0) * 8.0 * above;
    let deep = vec3<f32>(0.008, 0.035, 0.05) + vec3<f32>(0.05, 0.03, 0.01) * max(sun.y, 0.0);
    var col = mix(deep, sky(r, above), fresnel) + glint;
    let fog = 1.0 - exp(-dist / 6000.0);
    col = mix(col, sky(normalize(vec3<f32>(-v.x, 0.02, -v.z)), 1.0), fog);
    return vec4<f32>(col, 1.0);
}
