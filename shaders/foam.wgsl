// Foam accumulation over cascade 0's 1024 m period at 0.5 m per texel, in the surface's
// undisplaced (parameter) coordinates. Mirrors src/waves/foam.rs: the combined Jacobian of
// cascades 0–2 drives injection; foam decays exponentially.

struct Params {
    time: f32,
    choppiness: f32,
    dt: f32,
    threshold: f32,
    inject: f32,
    decay_per_s: f32,
    _pad: vec2<f32>,
    sizes: vec4<f32>,
};

@group(0) @binding(0) var displacement: texture_2d_array<f32>;
@group(0) @binding(1) var derivatives: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var prev: texture_2d<f32>;
@group(0) @binding(4) var next: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> params: Params;

const FOAM_RES: u32 = 2048u;
const FOAM_CASCADES: u32 = 3u;
const FOAM_MAX: f32 = 4.0;
const N: f32 = 256.0;

fn foam_step(prev_foam: f32, jacobian: f32, dt: f32) -> f32 {
    let decayed = max(prev_foam * exp(-params.decay_per_s * dt), 0.0);
    return clamp(decayed + params.inject * clamp(params.threshold - jacobian, 0.0, 1.0) * dt, 0.0, FOAM_MAX);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= FOAM_RES || id.y >= FOAM_RES) {
        return;
    }
    let p = (vec2<f32>(id.xy) + 0.5) * (params.sizes.x / f32(FOAM_RES));
    var d = vec3<f32>(0.0);
    for (var c = 0u; c < FOAM_CASCADES; c++) {
        let uv = p / params.sizes[c] + vec2<f32>(0.5 / N);
        let der = textureSampleLevel(derivatives, samp, uv, c, 0.0);
        let jxz = textureSampleLevel(displacement, samp, uv, c, 0.0).w;
        d += vec3<f32>(der.z, der.w, jxz);
    }
    let jacobian = (1.0 + d.x) * (1.0 + d.y) - d.z * d.z;
    let old = textureLoad(prev, vec2<i32>(id.xy), 0).r;
    textureStore(next, vec2<i32>(id.xy), vec4<f32>(foam_step(old, jacobian, params.dt), 0.0, 0.0, 0.0));
}
