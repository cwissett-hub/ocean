// Ocean surface into the G-buffer: nested camera-centred grids displaced by the four FFT
// cascades. Needs frame_common.wgsl.

@group(0) @binding(1) var displacement: texture_2d_array<f32>;
@group(0) @binding(2) var derivatives: texture_2d_array<f32>;
@group(0) @binding(3) var samp: sampler;

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
    var der = vec4<f32>(0.0);
    var variance = frame.misc.x;
    for (var c = 0u; c < 4u; c++) {
        let texel = frame.sizes[c] / N_TEXELS;
        let w = 1.0 - smoothstep(texel, 4.0 * texel, footprint);
        if (w > 0.0) {
            der += w * textureSampleLevel(derivatives, samp, cascade_uv(in.grid_xz, c), c, 0.0);
        }
        variance += (1.0 - w) * frame.slope_var[c];
    }
    // Where a crest folds (1 + ∂D/∂x → 0) the normal would blow up; clamp the stretch.
    let n = normalize(vec3<f32>(-der.x / max(1.0 + der.z, 0.2), 1.0, -der.y / max(1.0 + der.w, 0.2)));

    let prev = frame.prev_view_proj * vec4<f32>(in.world, 1.0);
    let prev_px = (vec2<f32>(prev.x, -prev.y) / prev.w * 0.5 + 0.5) * frame.resolution.xy;

    var o: GBufferOut;
    o.normal_var = vec4<f32>(n, variance);
    o.material = vec4<f32>(0.0);
    o.motion = prev_px - in.pos.xy;
    return o;
}
