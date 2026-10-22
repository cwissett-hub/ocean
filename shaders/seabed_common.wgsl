// The seabed heightmap around the camera (baked on the CPU by src/render/seabed.rs from the
// same height functions the raft physics uses) and what shallow water does to the waves.
// `sp` is (local corner x, z, texel m, resolution); resolution 0 means no map yet.

const SEA_DEPTH_M: f32 = 25.0;

// Bilinear between texel centres; open sea depth off the map.
fn seabed_height(map: texture_2d<f32>, sp: vec4<f32>, p: vec2<f32>) -> f32 {
    let res = sp.w;
    let t = (p - sp.xy) / max(sp.z, 1e-6) - 0.5;
    if (res < 1.0 || any(t < vec2<f32>(0.0)) || any(t > vec2<f32>(res - 1.0))) {
        return -SEA_DEPTH_M;
    }
    let i = vec2<i32>(floor(t));
    let f = t - floor(t);
    let j = min(i + 1, vec2<i32>(i32(res) - 1));
    let a = textureLoad(map, i, 0).r;
    let b = textureLoad(map, vec2<i32>(j.x, i.y), 0).r;
    let c = textureLoad(map, vec2<i32>(i.x, j.y), 0).r;
    let d = textureLoad(map, j, 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Fraction of the open-sea wave displacement that survives over water `depth` m deep
// (world::islands::wave_attenuation is the same formula).
fn wave_attenuation(depth: f32) -> f32 {
    return smoothstep(0.0, 10.0, depth);
}
