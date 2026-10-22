// #include frame_common.wgsl
// Temporal anti-aliasing resolve (see src/render/taa.rs). Each frame's jittered HDR image is
// blended into the history reprojected by the G-buffer motion vectors (the sky by the camera
// rotation alone). The history is clipped to this frame's local colour distribution (YCoCg
// variance box) so moving things don't ghost, and samples are weighted by 1 / (1 + luma) so a
// single bright glint can't dominate a pixel — glints shimmer instead of popping.

struct TaaParams {
    history_valid: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(1) var current: texture_2d<f32>;
@group(0) @binding(2) var history: texture_2d<f32>;
@group(0) @binding(3) var motion_tex: texture_2d<f32>;
@group(0) @binding(4) var depth_tex: texture_depth_2d;
@group(0) @binding(5) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var lin: sampler;
@group(0) @binding(7) var<uniform> tp: TaaParams;

fn to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

fn from_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Mirrors render::taa::history_weight.
fn history_weight(valid: bool, motion_px: f32, on_screen: bool) -> f32 {
    if (!valid || !on_screen || !(motion_px == motion_px) || motion_px > 1e30) {
        return 0.0;
    }
    return 0.9 - 0.35 * clamp(motion_px / 16.0, 0.0, 1.0);
}

fn luma_weight(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)) * frame.misc.z);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(out_tex));
    let p = vec2<i32>(id.xy);
    if (p.x >= size.x || p.y >= size.y) {
        return;
    }
    // Neighbourhood statistics (YCoCg), for clipping the history.
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let s = to_ycocg(textureLoad(current, clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), size - 1), 0).rgb);
            m1 += s;
            m2 += s * s;
        }
    }
    m1 /= 9.0;
    let sigma = sqrt(max(m2 / 9.0 - m1 * m1, vec3<f32>(0.0)));
    let lo = m1 - 1.25 * sigma;
    let hi = m1 + 1.25 * sigma;

    let cur = textureLoad(current, p, 0).rgb;
    let px = vec2<f32>(p) + 0.5;
    var motion = textureLoad(motion_tex, p, 0).xy;
    if (textureLoad(depth_tex, p, 0) <= 0.0) {
        // Sky: nothing wrote motion; it moves with the camera's rotation only.
        let dir = view_dir_for_pixel(px + frame.jitter.xy);
        let prev = frame.prev_view_proj * vec4<f32>(dir, 0.0);
        let prev_px = (vec2<f32>(prev.x, -prev.y) / prev.w * 0.5 + 0.5) * frame.resolution.xy;
        motion = select(vec2<f32>(1e31), prev_px - px, prev.w > 0.0);
    }
    let prev_uv = (px + motion) / vec2<f32>(size);
    let on_screen = all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0));
    let w = history_weight(tp.history_valid != 0u, length(motion), on_screen);
    var out = cur;
    if (w > 0.0) {
        let hist_raw = to_ycocg(textureSampleLevel(history, lin, prev_uv, 0.0).rgb);
        let hist = from_ycocg(clamp(hist_raw, lo, hi));
        let wc = (1.0 - w) * luma_weight(cur);
        let wh = w * luma_weight(hist);
        out = (cur * wc + hist * wh) / max(wc + wh, 1e-6);
    }
    textureStore(out_tex, p, vec4<f32>(max(out, vec3<f32>(0.0)), 1.0));
}
