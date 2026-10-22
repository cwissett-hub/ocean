// #include seabed_common.wgsl
// Writes the displaced near-field wave patch the ray-tracing BLAS is built from: a
// (grid+1)² vertex grid of world positions, sampled exactly like the render mesh.

struct PatchParams {
    origin: vec2<f32>,
    cell: f32,
    grid: u32,
    sizes: vec4<f32>,
    lambda_min: vec4<f32>,
    seabed: vec4<f32>,
};

@group(0) @binding(0) var displacement: texture_2d_array<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read_write> verts: array<f32>;
@group(0) @binding(3) var<uniform> pp: PatchParams;
@group(0) @binding(4) var seabed: texture_2d<f32>;

const N: f32 = 256.0;

fn vertex_weight(c: u32, cell: f32) -> f32 {
    let l = pp.lambda_min[c];
    return 1.0 - smoothstep(0.25 * l, 0.5 * l, cell);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x > pp.grid || id.y > pp.grid) {
        return;
    }
    let p = pp.origin + vec2<f32>(id.xy) * pp.cell;
    var d = vec3<f32>(0.0);
    for (var c = 0u; c < 4u; c++) {
        let w = vertex_weight(c, pp.cell);
        if (w > 0.0) {
            d += w * textureSampleLevel(displacement, samp, p / pp.sizes[c] + vec2<f32>(0.5 / N), c, 0.0).xyz;
        }
    }
    d *= wave_attenuation(-seabed_height(seabed, pp.seabed, p));
    let i = (id.y * (pp.grid + 1u) + id.x) * 3u;
    verts[i] = p.x + d.x;
    verts[i + 1u] = d.y;
    verts[i + 2u] = p.y + d.z;
}
