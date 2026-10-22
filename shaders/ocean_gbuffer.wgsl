// #include frame_common.wgsl
// #include seabed_common.wgsl
// #include wake_common.wgsl
// Ocean surface into the G-buffer: nested camera-centred grids displaced by the four FFT
// cascades, calming over the shallows. Needs frame_common.wgsl, seabed_common.wgsl.

@group(0) @binding(1) var displacement: texture_2d_array<f32>;
@group(0) @binding(2) var derivatives: texture_2d_array<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var foam_tex: texture_2d<f32>;
@group(0) @binding(5) var seabed: texture_2d<f32>;
@group(0) @binding(6) var wake: texture_2d<f32>;

fn water_depth(p: vec2<f32>) -> f32 {
    return -seabed_height(seabed, frame.seabed, p);
}

fn hash2(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(0.1031, 0.1030));
    let r = q + dot(q, q.yx + 33.33);
    return fract((r.x + r.y) * r.x);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Cellular noise: distance to the nearest and second-nearest feature point (F1, F2).
fn worley(p: vec2<f32>) -> vec2<f32> {
    let i = floor(p);
    let f = fract(p);
    var d1 = 8.0;
    var d2 = 8.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = vec2<f32>(f32(x), f32(y));
            let o = vec2<f32>(hash2(i + c), hash2(i + c + vec2<f32>(19.1, 7.3)));
            let d = length(c + o - f);
            if (d < d1) {
                d2 = d1;
                d1 = d;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }
    return vec2<f32>(d1, d2);
}

// Foam bubble network: bright walls where two cells meet, holes in the middle.
fn foam_cells(p: vec2<f32>) -> f32 {
    let w = worley(p);
    // Soft, uneven walls: real foam networks are blotchy, not drawn in lines.
    return (1.0 - smoothstep(0.0, 0.45, w.y - w.x)) * (0.6 + 0.4 * value_noise(p * 0.7));
}

// Thick fresh foam reads as solid white; as it decays it opens into a network of bubble walls
// (cellular, at two scales) streaked along the wind, then thins to lace.
fn foam_coverage(foam: f32, p: vec2<f32>, footprint: f32) -> f32 {
    let a = frame.misc.w;
    let along = vec2<f32>(sin(a), cos(a));
    let q = vec2<f32>(dot(p, vec2<f32>(along.y, -along.x)), dot(p, along));
    let streak = value_noise(q * vec2<f32>(0.8, 0.18));
    let cells = foam_cells(q * vec2<f32>(0.9, 0.6)) * 0.6 + foam_cells(q * 2.3 + 11.0) * 0.4;
    let mottle = value_noise(q * 0.35 + 3.0);
    // Far away the pattern is sub-pixel: use its average instead of aliasing.
    let detail = mix(0.4 * streak + 0.3 * cells + 0.3 * mottle, 0.45, smoothstep(0.08, 0.5, footprint));
    // The foam texture has 0.5 m texels and no mips: fade it before it aliases into specks.
    let distance_fade = 1.0 - smoothstep(0.75, 3.0, footprint);
    return smoothstep(0.0, 1.0, clamp(foam * 2.2 - (1.0 - detail) * 1.1, 0.0, 1.0)) * distance_fade;
}

// The surf line: foam in the thin water of the swash, thickest at its leading edge while it
// runs up the beach, broken into lace like the open-sea foam.
fn surf(p: vec2<f32>, depth: f32, footprint: f32) -> f32 {
    let t = frame.cam_pos.w;
    // Water over this spot of sand (or sea bottom), m.
    let thick = swash_height(p) + depth;
    let band = smoothstep(0.0, 0.02, thick) * smoothstep(0.22, 0.0, thick);
    let running_up = max(cos(swash_phase(p)), 0.0);
    let edge = smoothstep(0.25, 0.0, thick);
    let wash = mix(0.3, 1.0, running_up * edge + 0.4 * (1.0 - edge));
    let lace = value_noise(p * 1.7 + vec2<f32>(t * 0.15, 0.0)) * 0.6 + value_noise(p * 5.3) * 0.4;
    let detail = mix(lace, 0.6, smoothstep(0.2, 1.0, footprint));
    return clamp(band * mix(0.4, 1.1, wash) * 1.8 - (1.0 - detail) * 1.4, 0.0, 1.0);
}

fn level_center(cam_xz: vec2<f32>, cell: f32) -> vec2<f32> {
    let snap = 2.0 * cell;
    return round(cam_xz / snap) * snap;
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
    // The swell dies out over the last 10 m to the shore and never climbs the beach.
    // The raft's own waves (bow wave, the dip along the hull, its Kelvin wake) ride on top.
    d.y += wake_height(wake, frame.wake, p);
    let depth_here = water_depth(p);
    d *= wave_attenuation(depth_here);
    // At the shore the swash runs up the beach and back (the sand shading knows when it was wet).
    d.y += swash_height(p) * smoothstep(1.5, 0.2, depth_here);

    var o: OceanOut;
    o.world = vec3<f32>(p.x + d.x, d.y, p.y + d.z);
    o.pos = frame.view_proj * vec4<f32>(o.world, 1.0);
    o.grid_xz = p;
    o.level = level;
    return o;
}

struct GBufferOut {
    @location(0) normal_var: vec4<f32>,   // world normal, total slope variance
    @location(1) material: vec4<f32>,     // foam, material id (0 water), 0, 0
    @location(2) motion: vec2<f32>,       // screen-space motion to the previous frame, pixels
};

@fragment
fn fs_ocean(in: OceanOut) -> GBufferOut {
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

    // Resolved cascades shape the normal; unresolved ones become roughness, so the total
    // slope variance (and so the width of the sun's glitter path) is preserved.
    // In the shallows the long waves flatten with the surface (as displaced above), while the
    // short ripples, which barely feel the bottom, last until the last half metre.
    let depth = water_depth(in.grid_xz);
    let att = wave_attenuation(depth);
    let ripple_att = smoothstep(0.0, 0.5, depth);
    var der = vec4<f32>(0.0);
    var variance = frame.misc.x;
    for (var c = 0u; c < 4u; c++) {
        let texel = frame.sizes[c] / N_TEXELS;
        let w = 1.0 - smoothstep(texel, 4.0 * texel, footprint);
        let a = select(att, ripple_att, c >= 2u);
        if (w > 0.0) {
            der += a * w * textureSampleLevel(derivatives, samp, cascade_uv(in.grid_xz, c), c, 0.0);
        }
        variance += a * a * (1.0 - w) * frame.slope_var[c];
    }
    // Where a crest folds (1 + ∂D/∂x → 0) the normal would blow up; clamp the stretch.
    let wake_s = att * wake_slope(wake, frame.wake, in.grid_xz);
    let n = normalize(vec3<f32>(-der.x / max(1.0 + der.z, 0.2) - wake_s.x, 1.0, -der.y / max(1.0 + der.w, 0.2) - wake_s.y));

    let prev = frame.prev_view_proj * vec4<f32>(in.world, 1.0);
    let prev_px = (vec2<f32>(prev.x, -prev.y) / prev.w * 0.5 + 0.5) * frame.resolution.xy;

    var o: GBufferOut;
    o.normal_var = vec4<f32>(n, variance);
    // Breaking-crest foam fades with the waves that make it (the foam texture tiles every
    // 1 km, so the shallows are applied here rather than when it is accumulated).
    let foam = textureSampleLevel(foam_tex, samp, in.grid_xz / frame.sizes.x, 0.0).r * att;
    // Water the raft has broken: steep wake slopes (the bow wave, the wake's first crests).
    // Only close to the hull: further out the wake is unbroken waves.
    let hull_xz = (frame.wake.xy + 0.5 * frame.wake.w) * frame.wake.z;
    let churn = 0.4 * smoothstep(0.3, 0.6, length(wake_s)) * smoothstep(5.0, 2.5, distance(in.grid_xz, hull_xz));
    o.material = vec4<f32>(max(foam_coverage(foam + churn, in.grid_xz, footprint), surf(in.grid_xz, depth, footprint)), 0.0, 0.0, 0.0);
    // Motion of the unjittered image (TAA jitters the projection, not the scene).
    o.motion = prev_px - (in.pos.xy - frame.jitter.xy);
    return o;
}
