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

// Height of one cell (by world cell index), flat outside the window.
fn wake_cell(map: texture_2d<f32>, wp: vec4<f32>, cell: vec2<f32>) -> f32 {
    let rel = cell - wp.xy;
    if (any(rel < vec2<f32>(0.0)) || any(rel > vec2<f32>(wp.w - 1.0))) {
        return 0.0;
    }
    return textureLoad(map, wake_texel(cell, wp.w), 0).r;
}

// Wake slope (∂h/∂x, ∂h/∂z): central-difference gradients at the four surrounding cell
// centres, interpolated bilinearly — smooth, where differencing the bilinear height would be
// constant per cell and shade the wake in facets.
fn wake_slope(map: texture_2d<f32>, wp: vec4<f32>, p: vec2<f32>) -> vec2<f32> {
    if (wp.w < 1.0) {
        return vec2<f32>(0.0);
    }
    let c = p / wp.z - 0.5;
    let i = floor(c);
    let f = c - i;
    var g: array<vec2<f32>, 4>;
    for (var k = 0; k < 4; k++) {
        let o = i + vec2<f32>(f32(k & 1), f32(k >> 1));
        g[k] = vec2<f32>(
            wake_cell(map, wp, o + vec2<f32>(1.0, 0.0)) - wake_cell(map, wp, o - vec2<f32>(1.0, 0.0)),
            wake_cell(map, wp, o + vec2<f32>(0.0, 1.0)) - wake_cell(map, wp, o - vec2<f32>(0.0, 1.0)),
        ) / (2.0 * wp.z);
    }
    return mix(mix(g[0], g[1], f.x), mix(g[2], g[3], f.x), f.y);
}
