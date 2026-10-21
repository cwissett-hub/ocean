// Per-frame uniforms shared by the G-buffer and shading passes (group 0, binding 0).
// Must match FrameUniforms in src/render/frame.rs.

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,     // xyz, time
    sun_dir: vec4<f32>,     // xyz
    sizes: vec4<f32>,       // cascade patch sizes, m
    lambda_min: vec4<f32>,  // shortest wavelength in each cascade's band, m
    grid: vec4<f32>,        // base cell m, levels, grid quads per side, 0
    slope_var: vec4<f32>,   // mean-square slope of each cascade
    misc: vec4<f32>,        // unresolved slope variance, significant wave height m, exposure, wind direction rad
    resolution: vec4<f32>,  // w, h, 1/w, 1/h of the internal render targets
};

@group(0) @binding(0) var<uniform> frame: Frame;

const N_TEXELS: f32 = 256.0;

// Texel (x, z) of a cascade holds world point (x, z)·L/N; texel centres sit at +0.5 texel.
fn cascade_uv(p: vec2<f32>, c: u32) -> vec2<f32> {
    return p / frame.sizes[c] + vec2<f32>(0.5 / N_TEXELS);
}

fn world_from_depth(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(pixel.x * frame.resolution.z * 2.0 - 1.0, 1.0 - pixel.y * frame.resolution.w * 2.0);
    let h = frame.inv_view_proj * vec4<f32>(ndc, depth, 1.0);
    return h.xyz / h.w;
}

fn view_dir_for_pixel(pixel: vec2<f32>) -> vec3<f32> {
    return normalize(world_from_depth(pixel, 0.5) - frame.cam_pos.xyz);
}
