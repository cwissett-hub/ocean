// 256-point inverse Stockham radix-2 FFT along rows (HORIZONTAL) or columns, over two packed
// rgba32float array textures (4 complex channels per texel). One workgroup per line; the
// whole line lives in workgroup memory for all eight stages.

override HORIZONTAL: bool = true;

@group(0) @binding(0) var src0: texture_2d_array<f32>;
@group(0) @binding(1) var src1: texture_2d_array<f32>;
@group(0) @binding(2) var dst0: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(3) var dst1: texture_storage_2d_array<rgba32float, write>;

const N: u32 = 256u;
const HALF: u32 = 128u;
const STAGES: u32 = 8u;
const PI: f32 = 3.14159265;

var<workgroup> a0: array<vec4<f32>, 256>;
var<workgroup> a1: array<vec4<f32>, 256>;
var<workgroup> b0: array<vec4<f32>, 256>;
var<workgroup> b1: array<vec4<f32>, 256>;

fn texel(line: u32, i: u32) -> vec2<i32> {
    if (HORIZONTAL) {
        return vec2<i32>(i32(i), i32(line));
    }
    return vec2<i32>(i32(line), i32(i));
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Multiplies both complex numbers packed in v by w.
fn cmul2(v: vec4<f32>, w: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(cmul(v.xy, w), cmul(v.zw, w));
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let j = lid.x;
    let line = wg.y;
    let layer = wg.z;
    a0[j] = textureLoad(src0, texel(line, j), layer, 0);
    a1[j] = textureLoad(src1, texel(line, j), layer, 0);
    a0[j + HALF] = textureLoad(src0, texel(line, j + HALF), layer, 0);
    a1[j + HALF] = textureLoad(src1, texel(line, j + HALF), layer, 0);
    workgroupBarrier();

    var ns = 1u;
    for (var s = 0u; s < STAGES; s++) {
        let k = j % ns;
        let angle = PI * f32(k) / f32(ns); // +: inverse transform
        let w = vec2<f32>(cos(angle), sin(angle));
        let out = (j / ns) * ns * 2u + k;
        if (s % 2u == 0u) {
            let u0 = a0[j];
            let u1 = a1[j];
            let v0 = cmul2(a0[j + HALF], w);
            let v1 = cmul2(a1[j + HALF], w);
            b0[out] = u0 + v0;
            b1[out] = u1 + v1;
            b0[out + ns] = u0 - v0;
            b1[out + ns] = u1 - v1;
        } else {
            let u0 = b0[j];
            let u1 = b1[j];
            let v0 = cmul2(b0[j + HALF], w);
            let v1 = cmul2(b1[j + HALF], w);
            a0[out] = u0 + v0;
            a1[out] = u1 + v1;
            a0[out + ns] = u0 - v0;
            a1[out + ns] = u1 - v1;
        }
        ns *= 2u;
        workgroupBarrier();
    }

    // Eight stages: the result ends in the a buffers.
    textureStore(dst0, texel(line, j), layer, a0[j]);
    textureStore(dst1, texel(line, j), layer, a1[j]);
    textureStore(dst0, texel(line, j + HALF), layer, a0[j + HALF]);
    textureStore(dst1, texel(line, j + HALF), layer, a1[j + HALF]);
}
