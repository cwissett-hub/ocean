// #include frame_common.wgsl
// #include sky_common.wgsl
// #include sky_sample.wgsl
// Camera-facing billboards for spray and mist, lit by the sun (strong forward scattering, so
// spray glows when backlit by a low sun) and the sky; soft-faded against the scene depth.

struct Particle {
    pos: vec3<f32>,
    life: f32,
    vel: vec3<f32>,
    kind: u32,
    size: f32,
    max_life: f32,
    seed: f32,
    _pad: f32,
};

@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world: vec3<f32>,
    @location(2) @interpolate(flat) kind: u32,
    @location(3) fade: f32,
    @location(4) radius: f32,
};

const CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let q = particles[ii];
    var o: VsOut;
    if (q.life <= 0.0) {
        o.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    let corner = CORNERS[vi];
    let to_cam = normalize(frame.cam_pos.xyz - q.pos);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_cam));
    let up = cross(to_cam, right);
    let world = q.pos + (corner.x * right + corner.y * up) * q.size * 0.5;
    o.clip = frame.view_proj * vec4<f32>(world, 1.0);
    o.uv = corner;
    o.world = world;
    o.kind = q.kind;
    let age = 1.0 - q.life / q.max_life;
    // Fade in quickly, out slowly.
    o.fade = smoothstep(0.0, 0.08, age) * (1.0 - smoothstep(0.6, 1.0, age));
    o.radius = q.size * 0.5;
    return o;
}

fn henyey_greenstein(c: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * c, 1.5));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }
    let spray = in.kind == 0u;
    var alpha = pow(1.0 - r2, select(3.0, 2.0, spray)) * in.fade * select(0.018, 0.35, spray);
    // Mist right in front of the camera would read as a flat disc.
    if (!spray) {
        alpha *= smoothstep(15.0, 40.0, length(in.world - frame.cam_pos.xyz));
    }
    // Soft contact with whatever is behind (water, buoy).
    let d = textureLoad(depth_tex, vec2<i32>(in.clip.xy), 0);
    if (d > 0.0) {
        let scene = world_from_depth(in.clip.xy, d);
        let gap = length(scene - frame.cam_pos.xyz) - length(in.world - frame.cam_pos.xyz);
        alpha *= smoothstep(0.0, in.radius, gap);
    }
    if (alpha < 0.002) {
        discard;
    }
    let sun = frame.sun_dir.xyz;
    let view = normalize(in.world - frame.cam_pos.xyz);
    let phase = henyey_greenstein(dot(view, sun), select(0.5, 0.75, spray)) * 4.0 * PI;
    let light = sun_transmittance() * phase * 0.25 + sky_irradiance() / PI;
    let col = light * select(0.85, 0.95, spray);
    return vec4<f32>(col * alpha, alpha);
}
