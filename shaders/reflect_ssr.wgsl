// #include frame_common.wgsl
// Half-resolution screen-space reflections. Marches the reflected ray in world space with
// growing steps, tests it against the depth buffer and returns last frame's lit colour at the
// hit (alpha = confidence; 0 = missed, shading falls back to the sky). Needs frame_common.wgsl.

struct ReflectParams {
    jitter: vec2<u32>,   // which full-res pixel of each 2x2 block this frame samples
    history_valid: u32,
    _pad: u32,
};

@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var prev_hdr: texture_2d<f32>;
@group(0) @binding(4) var lin: sampler;
@group(0) @binding(5) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var<uniform> rp: ReflectParams;

const STEPS: u32 = 48u;
const REFINE: u32 = 5u;
const FIRST_STEP_M: f32 = 0.15;
const GROWTH: f32 = 1.12;
const MIN_HIT_PX: f32 = 3.0;

fn project(p: vec3<f32>) -> vec3<f32> {
    let c = frame.view_proj * vec4<f32>(p, 1.0);
    let ndc = c.xyz / c.w;
    return vec3<f32>((ndc.x * 0.5 + 0.5) * frame.resolution.x, (0.5 - ndc.y * 0.5) * frame.resolution.y, ndc.z);
}

fn scene_depth(px: vec2<f32>) -> f32 {
    let i = vec2<i32>(clamp(px, vec2<f32>(0.0), frame.resolution.xy - 1.0));
    return textureLoad(depth_tex, i, 0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let half = textureDimensions(out_tex);
    if (id.x >= half.x || id.y >= half.y) {
        return;
    }
    let full = min(id.xy * 2u + rp.jitter, vec2<u32>(frame.resolution.xy) - 1u);
    let d = textureLoad(depth_tex, vec2<i32>(full), 0);
    if (d <= 0.0) {
        textureStore(out_tex, vec2<i32>(id.xy), vec4<f32>(0.0));
        return;
    }
    let px = vec2<f32>(full) + 0.5;
    let p = world_from_depth(px, d);
    let n = textureLoad(normal_tex, vec2<i32>(full), 0).xyz;
    let v = normalize(frame.cam_pos.xyz - p);
    let r = reflect(-v, n);

    var t = FIRST_STEP_M;
    var step = FIRST_STEP_M;
    var prev_t = 0.0;
    var hit = false;
    var hit_px = vec2<f32>(0.0);
    for (var i = 0u; i < STEPS; i++) {
        let s = project(p + r * t);
        if (s.x < 0.0 || s.y < 0.0 || s.x >= frame.resolution.x || s.y >= frame.resolution.y || s.z <= 0.0) {
            break;
        }
        let sd = scene_depth(s.xy);
        // Reversed-Z: the ray is behind the surface when the stored depth is larger.
        if (sd > s.z) {
            // Reject hits through thin geometry: the surface must be near the ray in depth.
            let surf = world_from_depth(s.xy, sd);
            if (length(surf - (p + r * t)) < 0.5 + step * 2.0) {
                // Bisect between the last miss and this hit.
                var lo = prev_t;
                var hi = t;
                for (var k = 0u; k < REFINE; k++) {
                    let mid = 0.5 * (lo + hi);
                    let m = project(p + r * mid);
                    if (scene_depth(m.xy) > m.z) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                hit_px = project(p + r * hi).xy;
                hit = true;
            }
            break;
        }
        prev_t = t;
        step *= GROWTH;
        t += step;
    }
    if (!hit) {
        textureStore(out_tex, vec2<i32>(id.xy), vec4<f32>(0.0));
        return;
    }
    // A hit on (or beside) the source pixel is the surface finding itself: feeding last
    // frame's colour back in would darken it a little more every frame.
    if (length(hit_px - px) < MIN_HIT_PX) {
        textureStore(out_tex, vec2<i32>(id.xy), vec4<f32>(0.0));
        return;
    }
    // Near the sides rays run off-screen and fall back to the sky; fade by the hit's and the
    // source's distance from the side edges so that seam becomes a gradient.
    let edge_hit = min(hit_px.x, frame.resolution.x - hit_px.x);
    let edge_src = min(px.x, frame.resolution.x - px.x);
    let ramp = 0.08 * frame.resolution.x;
    // Beyond ~100 m waves are a few pixels wide and SSR only finds itself; wave-on-wave
    // occlusion there is statistical and lives in the BRDF's masking term instead.
    let dist = length(frame.cam_pos.xyz - p);
    let conf = smoothstep(0.0, ramp, edge_hit) * smoothstep(0.0, ramp, edge_src) * (1.0 - smoothstep(80.0, 250.0, dist));
    let col = textureSampleLevel(prev_hdr, lin, hit_px * frame.resolution.zw, 0.0).rgb;
    textureStore(out_tex, vec2<i32>(id.xy), vec4<f32>(col, conf));
}
