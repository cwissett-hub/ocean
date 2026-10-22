// Sampling the raft's wake (src/render/wake.rs, wake.wgsl). `wp` is (window origin cell x,
// z, cell size m, resolution); resolution 0 means no wake yet. The grid is toroidal: a world
// cell lives in texel cell mod resolution.

fn wake_texel(cell: vec2<f32>, res: f32) -> vec2<i32> {
    let r = i32(res);
    let c = vec2<i32>(cell);
    return ((c % r) + r) % r;
}

// Wake height at local point p (bilinear between cell centres); flat outside the window.
fn wake_height(map: texture_2d<f32>, wp: vec4<f32>, p: vec2<f32>) -> f32 {
    let res = wp.w;
    if (res < 1.0) {
        return 0.0;
    }
    let c = p / wp.z - 0.5;
    let rel = c - wp.xy;
    if (any(rel < vec2<f32>(0.0)) || any(rel > vec2<f32>(res - 1.0))) {
        return 0.0;
    }
    let i = floor(c);
    let f = c - i;
    let a = textureLoad(map, wake_texel(i, res), 0).r;
    let b = textureLoad(map, wake_texel(i + vec2<f32>(1.0, 0.0), res), 0).r;
    let d = textureLoad(map, wake_texel(i + vec2<f32>(0.0, 1.0), res), 0).r;
    let e = textureLoad(map, wake_texel(i + vec2<f32>(1.0, 1.0), res), 0).r;
    return mix(mix(a, b, f.x), mix(d, e, f.x), f.y);
}

// Wake slope (∂h/∂x, ∂h/∂z) by central differences.
fn wake_slope(map: texture_2d<f32>, wp: vec4<f32>, p: vec2<f32>) -> vec2<f32> {
    let e = wp.z;
    return vec2<f32>(
        wake_height(map, wp, p + vec2<f32>(e, 0.0)) - wake_height(map, wp, p - vec2<f32>(e, 0.0)),
        wake_height(map, wp, p + vec2<f32>(0.0, e)) - wake_height(map, wp, p - vec2<f32>(0.0, e)),
    ) / (2.0 * e);
}
