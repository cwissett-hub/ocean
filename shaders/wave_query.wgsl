// Water height and normal at arbitrary world points (mirrors waves::query::invert_horizontal).
// Samples the same displacement textures the renderer draws.

struct QueryParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    sizes: vec4<f32>,
};

@group(0) @binding(0) var displacement: texture_2d_array<f32>;
@group(0) @binding(1) var derivatives: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> results: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> qp: QueryParams;

// The sub-metre cascade only makes ripples; nothing that floats feels it.
const QUERY_CASCADES: u32 = 3u;
const ITERATIONS: u32 = 4u;
const N: f32 = 256.0;

fn uv(p: vec2<f32>, c: u32) -> vec2<f32> {
    return p / qp.sizes[c] + vec2<f32>(0.5 / N);
}

fn disp(p: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(0.0);
    for (var c = 0u; c < QUERY_CASCADES; c++) {
        d += textureSampleLevel(displacement, samp, uv(p, c), c, 0.0).xyz;
    }
    return d;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= qp.count) {
        return;
    }
    let p = points[id.x];
    var x = p;
    for (var i = 0u; i < ITERATIONS; i++) {
        x = p - disp(x).xz;
    }
    var der = vec4<f32>(0.0);
    for (var c = 0u; c < QUERY_CASCADES; c++) {
        der += textureSampleLevel(derivatives, samp, uv(x, c), c, 0.0);
    }
    let n = normalize(vec3<f32>(-der.x / max(1.0 + der.z, 0.2), 1.0, -der.y / max(1.0 + der.w, 0.2)));
    results[id.x] = vec4<f32>(disp(x).y, n);
}
