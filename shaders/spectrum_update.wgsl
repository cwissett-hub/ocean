// Evolves h0(k) to time t and writes the eight spectra the FFT turns into displacement,
// slope and displacement-gradient fields. Real fields are packed in pairs as F1 + i·F2
// (each is Hermitian, so the inverse FFT returns f1 + i·f2):
//   out0 = (η + i·Dx, Dz + i·sx)   out1 = (sz + i·Jxx, Jzz + i·Jxz)

struct Params {
    time: f32,
    choppiness: f32,
    _pad: vec2<f32>,
    sizes: vec4<f32>,
};

@group(0) @binding(0) var h0_tex: texture_2d_array<f32>;
@group(0) @binding(1) var omega_tex: texture_2d_array<f32>;
@group(0) @binding(2) var out0: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(3) var out1: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(4) var<uniform> params: Params;

const N: u32 = 256u;
const TAU: f32 = 6.2831855;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// i·z
fn ci(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-z.y, z.x);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= N || id.y >= N) {
        return;
    }
    let layer = id.z;
    let texel = vec2<i32>(id.xy);
    let dk = TAU / params.sizes[layer];
    let kx = (f32(id.x) - f32(N / 2u)) * dk;
    let kz = (f32(id.y) - f32(N / 2u)) * dk;
    let k = sqrt(kx * kx + kz * kz);
    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    if (k > 0.0) {
        let h0 = textureLoad(h0_tex, texel, layer, 0);
        let w = textureLoad(omega_tex, texel, layer, 0).x;
        let phase = w * params.time;
        let e = vec2<f32>(cos(phase), sin(phase));
        let ec = vec2<f32>(e.x, -e.y);
        let h = cmul(h0.xy, e) + cmul(h0.zw, ec);
        let lam = params.choppiness;
        let dx = -ci(h) * (kx / k) * lam;
        let dz = -ci(h) * (kz / k) * lam;
        let sx = ci(h) * kx;
        let sz = ci(h) * kz;
        let jxx = h * (kx * kx / k) * lam;
        let jzz = h * (kz * kz / k) * lam;
        let jxz = h * (kx * kz / k) * lam;
        o0 = vec4<f32>(h + ci(dx), dz + ci(sx));
        o1 = vec4<f32>(sz + ci(jxx), jzz + ci(jxz));
    }
    textureStore(out0, texel, layer, o0);
    textureStore(out1, texel, layer, o1);
}
