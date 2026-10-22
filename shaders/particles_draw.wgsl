// #include frame_common.wgsl
// #include sky_common.wgsl
// #include sky_sample.wgsl
// Billboards for spray and mist. Spray clumps are stretched along their motion, broken up into
// droplets by procedural noise and shaded as rough spheres (lit side, wrap, a bright rim and
// strong forward scattering when back-lit by a low sun); mist is soft and faint. Both fade
// softly against the scene depth.

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
    @location(5) right: vec3<f32>,
    @location(6) up: vec3<f32>,
    @location(7) @interpolate(flat) seed: f32,
};

const CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let q = particles[ii];
    var o: VsOut;
    // Dead particles and foam marks (which only deposit foam) draw nothing.
    if (q.life <= 0.0 || q.kind == 2u) {
        o.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return o;
    }
    let corner = CORNERS[vi];
    let to_cam = normalize(frame.cam_pos.xyz - q.pos);
    var right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_cam));
    var up = cross(to_cam, right);
    var stretch = 1.0;
    if (q.kind == 0u) {
        // Spray streaks along its motion (as seen on screen), up to 2.5× long.
        let v = q.vel - to_cam * dot(q.vel, to_cam);
        let speed = length(v);
        if (speed > 0.5) {
            up = v / speed;
            right = normalize(cross(up, to_cam));
            stretch = 1.0 + min(speed * 0.12, 1.5);
        }
    }
    let half = q.size * 0.5;
    let world = q.pos + (corner.x * right + corner.y * up * stretch) * half;
    o.clip = frame.view_proj * vec4<f32>(world, 1.0);
    o.uv = corner;
    o.world = world;
    o.kind = q.kind;
    let age = 1.0 - q.life / q.max_life;
    // Fade in quickly, out slowly.
    o.fade = smoothstep(0.0, 0.08, age) * (1.0 - smoothstep(0.6, 1.0, age));
    o.radius = half;
    o.right = right;
    o.up = up;
    o.seed = q.seed;
    return o;
}

fn henyey_greenstein(c: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * c, 1.5));
}

fn hash2(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(0.1031, 0.1030));
    let r = q + dot(q, q.yx + 33.33);
    return fract((r.x + r.y) * r.x);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x), mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }
    let spray = in.kind == 0u;
    var alpha: f32;
    if (spray) {
        // A clump of droplets: a soft core broken into blobs by two octaves of noise.
        let s = in.uv * 2.2 + in.seed * 37.0;
        let blobs = value_noise(s) * 0.65 + value_noise(s * 2.7 + 5.0) * 0.35;
        let body = pow(1.0 - r2, 1.5);
        alpha = smoothstep(0.35, 0.75, blobs * 0.6 + body * 0.7) * body * 0.55;
    } else {
        alpha = pow(1.0 - r2, 3.0) * 0.018;
        // Mist right in front of the camera would read as a flat disc.
        alpha *= smoothstep(15.0, 40.0, length(in.world - frame.cam_pos.xyz));
    }
    alpha *= in.fade;
    // Soft contact with whatever is behind (water, raft).
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
    let e_sun = sun_transmittance();
    var col: vec3<f32>;
    if (spray) {
        // Rough-sphere shading in the billboard frame.
        let n = normalize(in.uv.x * in.right + in.uv.y * in.up - view * sqrt(max(1.0 - r2, 0.0)));
        let wrap = clamp((dot(n, sun) + 0.4) / 1.4, 0.0, 1.0);
        let forward = henyey_greenstein(dot(view, sun), 0.7) * 4.0 * PI;
        let rim = pow(r2, 2.0) * forward;
        let shadow_tint = vec3<f32>(0.75, 0.88, 1.0);
        col = e_sun * (0.85 * wrap + 0.12 * forward + 0.35 * rim) / PI + sky_irradiance() * shadow_tint / PI;
    } else {
        let phase = henyey_greenstein(dot(view, sun), 0.5) * 4.0 * PI;
        col = (e_sun * phase * 0.25 + sky_irradiance() / PI) * 0.85;
    }
    return vec4<f32>(col * alpha, alpha);
}
