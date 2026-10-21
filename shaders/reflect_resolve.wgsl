// #include frame_common.wgsl
// Temporal accumulation of the half-res reflection (SSR or ray traced) with a 3x3
// neighbourhood clamp against ghosting. Blend weight mirrors render::reflect::temporal_blend.
// Needs frame_common.wgsl.

struct ReflectParams {
    jitter: vec2<u32>,
    history_valid: u32,
    _pad: u32,
};

@group(0) @binding(1) var current: texture_2d<f32>;
@group(0) @binding(2) var history: texture_2d<f32>;
@group(0) @binding(3) var motion_tex: texture_2d<f32>;
@group(0) @binding(4) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> rp: ReflectParams;

fn temporal_blend(valid: bool, motion_px: f32) -> f32 {
    if (!valid || !(motion_px == motion_px) || motion_px > 1e30) {
        return 0.0;
    }
    return 0.5 + 0.4 * (1.0 - clamp(motion_px / 4.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(out_tex));
    let p = vec2<i32>(id.xy);
    if (p.x >= size.x || p.y >= size.y) {
        return;
    }
    let cur = textureLoad(current, p, 0);
    var lo = cur;
    var hi = cur;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let s = textureLoad(current, clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), size - 1), 0);
            lo = min(lo, s);
            hi = max(hi, s);
        }
    }
    let full = min(p * 2, vec2<i32>(frame.resolution.xy) - 1);
    let motion = textureLoad(motion_tex, full, 0).xy;
    // Motion is in full-res pixels; history lives at half res.
    let prev_p = clamp(vec2<i32>(vec2<f32>(p) + motion * 0.5 + 0.5), vec2<i32>(0), size - 1);
    let hist = clamp(textureLoad(history, prev_p, 0), lo, hi);
    let w = temporal_blend(rp.history_valid != 0u, length(motion));
    textureStore(out_tex, p, mix(cur, hist, w));
}
