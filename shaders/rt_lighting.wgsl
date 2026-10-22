// #include frame_common.wgsl
enable wgpu_ray_query;

// Ray-traced sun shadow and ambient occlusion from the objects (raft, buoy, islands, palms),
// one ray each per pixel. The shadow ray is jittered across the sun's disc and the AO ray
// cosine-distributed over the hemisphere, both re-seeded every frame, so TAA resolves soft
// penumbrae and contact shading from the noise. Output: r = sun visibility, g = ambient
// visibility (the shade pass multiplies its lighting by them).

@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var tlas: acceleration_structure;

const OBJECT_MASK: u32 = 0x02u;
// A little wider than the real sun (0.27°) for soft, readable penumbrae.
const SUN_CONE: f32 = 0.012;
const AO_RANGE_M: f32 = 1.6;
const AO_STRENGTH: f32 = 0.85;

fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

fn basis(n: vec3<f32>) -> mat3x3<f32> {
    let a = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t = normalize(cross(a, n));
    return mat3x3<f32>(t, cross(n, t), n);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out_tex);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let px = vec2<i32>(id.xy);
    let d = textureLoad(depth_tex, px, 0);
    if (d <= 0.0) {
        textureStore(out_tex, px, vec4<f32>(1.0, 1.0, 0.0, 1.0));
        return;
    }
    let p = world_from_depth(vec2<f32>(id.xy) + 0.5, d);
    let n = normalize(textureLoad(normal_tex, px, 0).xyz);
    let origin = p + n * 0.03;
    // Per-pixel, per-frame random numbers (interleaved gradient noise, decorrelated).
    let f = frame.cam_pos.w * 60.0;
    let r1 = ign(vec2<f32>(id.xy) + vec2<f32>(5.588238 * f, 0.0));
    let r2 = ign(vec2<f32>(id.yx) + vec2<f32>(0.0, 7.1137 * f) + 13.0);
    let r3 = ign(vec2<f32>(id.xy) * 1.37 + vec2<f32>(3.31 * f, 9.7 * f));

    let sun = frame.sun_dir.xyz;
    var shadow = 1.0;
    if (sun.y > -0.05 && dot(n, sun) > -0.2) {
        let b = basis(sun);
        let a = r1 * 6.2831853;
        let rr = SUN_CONE * sqrt(r2);
        let dir = normalize(b * vec3<f32>(cos(a) * rr, sin(a) * rr, 1.0));
        var q: ray_query;
        rayQueryInitialize(&q, tlas, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, OBJECT_MASK, 0.02, 2000.0, origin, dir));
        rayQueryProceed(&q);
        if (rayQueryGetCommittedIntersection(&q).kind != RAY_QUERY_INTERSECTION_NONE) {
            shadow = 0.0;
        }
    }

    // Cosine-weighted hemisphere ray; nearer hits occlude more.
    let a2 = r3 * 6.2831853;
    let s = sqrt(r1);
    let ao_dir = normalize(basis(n) * vec3<f32>(cos(a2) * s, sin(a2) * s, sqrt(max(1.0 - r1, 0.0))));
    var q2: ray_query;
    rayQueryInitialize(&q2, tlas, RayDesc(RAY_FLAG_NONE, OBJECT_MASK, 0.02, AO_RANGE_M, origin, ao_dir));
    rayQueryProceed(&q2);
    let hit = rayQueryGetCommittedIntersection(&q2);
    var ao = 1.0;
    if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
        ao = 1.0 - AO_STRENGTH * (1.0 - hit.t / AO_RANGE_M);
    }
    textureStore(out_tex, px, vec4<f32>(shadow, ao, 0.0, 1.0));
}
