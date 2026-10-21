// Unpacks the inverse-FFT result into the textures the renderer samples:
//   displacement = (Dx, height, Dz, ∂Dx/∂z)   derivatives = (∂h/∂x, ∂h/∂z, ∂Dx/∂x, ∂Dz/∂z)

@group(0) @binding(0) var src0: texture_2d_array<f32>;
@group(0) @binding(1) var src1: texture_2d_array<f32>;
@group(0) @binding(2) var displacement: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var derivatives: texture_storage_2d_array<rgba16float, write>;

const N: u32 = 256u;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= N || id.y >= N) {
        return;
    }
    let texel = vec2<i32>(id.xy);
    let layer = id.z;
    // The centred k grid shifts every output texel by (−1)^(x+z).
    let sign = select(-1.0, 1.0, ((id.x + id.y) & 1u) == 0u);
    let p0 = textureLoad(src0, texel, layer, 0) * sign; // (η, Dx, Dz, sx)
    let p1 = textureLoad(src1, texel, layer, 0) * sign; // (sz, Jxx, Jzz, Jxz)
    textureStore(displacement, texel, layer, vec4<f32>(p0.y, p0.x, p0.z, p1.w));
    textureStore(derivatives, texel, layer, vec4<f32>(p0.w, p1.x, p1.y, p1.z));
}
