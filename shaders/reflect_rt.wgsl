// #include frame_common.wgsl
// #include sky_common.wgsl
// #include sky_sample.wgsl
// #include materials.wgsl
// #include surfaces.wgsl
// #include seabed_common.wgsl
enable wgpu_ray_query;

// Half-resolution hardware ray-traced reflections: one mirror ray per pixel from the
// G-buffer surface into the TLAS (near-field wave patch + buoy, raft, islands, palms). Hits are shaded with a
// simplified one-bounce BRDF; misses see the sky. Output goes through the same temporal
// resolve as screen-space reflections.

struct ReflectParams {
    jitter: vec2<u32>,
    history_valid: u32,
    patch_grid: u32,
};

@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> rp: ReflectParams;
@group(0) @binding(5) var tlas: acceleration_structure;
@group(0) @binding(6) var<storage, read> patch_verts: array<f32>;
@group(0) @binding(7) var derivatives: texture_2d_array<f32>;
@group(0) @binding(8) var wave_samp: sampler;
@group(0) @binding(9) var seabed: texture_2d<f32>;

// Smooth normal of the waves at a hit, from the same derivative field the main shading uses
// (the flat BLAS facets would mirror the sky in hard-edged patches). The finest resolved
// cascade fades with distance; the hit xz stands in for the undisplaced coordinate.
fn wave_normal(h: vec3<f32>) -> vec3<f32> {
    let dist = length(h - frame.cam_pos.xyz);
    var der = vec4<f32>(0.0);
    for (var c = 0u; c < 3u; c++) {
        var w = 1.0;
        if (c == 2u) {
            w = 1.0 - smoothstep(20.0, 80.0, dist);
        }
        der += w * textureSampleLevel(derivatives, wave_samp, cascade_uv(h.xz, c), c, 0.0);
    }
    return normalize(vec3<f32>(-der.x / max(1.0 + der.z, 0.2), 1.0, -der.y / max(1.0 + der.w, 0.2)));
}

// The ray tracing patch is coarser than the render mesh near the camera; start past the
// mismatch so a ray does not hit the surface it leaves.
const T_MIN: f32 = 0.3;
const T_MAX: f32 = 3000.0;
const PATCH_MASK: u32 = 0x01u;
const OBJECT_MASK: u32 = 0x02u;

fn patch_vertex(i: u32) -> vec3<f32> {
    return vec3<f32>(patch_verts[i * 3u], patch_verts[i * 3u + 1u], patch_verts[i * 3u + 2u]);
}

// Geometric normal of patch triangle `prim` (quad q = prim / 2), oriented upwards.
fn patch_normal(prim: u32) -> vec3<f32> {
    let g = rp.patch_grid;
    let q = prim / 2u;
    let a = (q / g) * (g + 1u) + q % g;
    let v = g + 1u;
    var i1 = a + 1u;
    var i2 = a + v + 1u;
    if (prim % 2u == 1u) {
        i1 = a + v + 1u;
        i2 = a + v;
    }
    let p0 = patch_vertex(a);
    let n = normalize(cross(patch_vertex(i1) - p0, patch_vertex(i2) - p0));
    return select(n, -n, n.y < 0.0);
}

// Water seen in a reflection: its own sky reflection plus the deep-water glow.
fn shade_reflected_water(h: vec3<f32>, n: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let v = -dir;
    let f = fresnel(max(dot(n, v), 0.0));
    let r0 = reflect(dir, n);
    let r = normalize(vec3<f32>(r0.x, max(r0.y, 0.0), r0.z));
    let col = sky_radiance(r) * f + (1.0 - f) * deep_water_upwelling();
    return aerial(col, dir, length(h - frame.cam_pos.xyz));
}

// An object seen in a reflection. Ray queries give no vertex data, so each mesh's normal and
// material parameter are rebuilt from the object-space hit, knowing its shape
// (src/scene/buoy.rs, raft_mesh.rs, terrain_mesh.rs); islands use the seabed map's slope.
fn shade_object_hit(h: vec3<f32>, local: vec3<f32>, r: vec3<f32>, id: u32, o2w: mat4x3<f32>) -> vec3<f32> {
    var ln = vec3<f32>(0.0, 1.0, 0.0);
    var mat = id;
    var param = 0.5;
    switch (id) {
        case MAT_BUOY: {
            // Cylinder-ish from the local position (sides) or ±y (caps).
            ln = normalize(vec3<f32>(local.x, 0.0, local.z));
            if (abs(local.y - 0.47) < 0.02 || abs(local.y - 3.26) < 0.02 || abs(local.y - 3.5) < 0.02) {
                ln = vec3<f32>(0.0, 1.0, 0.0);
            }
            param = (local.y + 0.5) / 4.0;
        }
        case MAT_RAFT_WOOD: {
            if (local.y > 0.85 && local.y < 2.52 && abs(local.x) > 0.08 && abs(local.x) < 0.92) {
                // The sail hangs across the raft, billowing forward.
                mat = MAT_SAIL;
                ln = vec3<f32>(0.0, 0.0, 1.0);
                param = (2.5 - local.y) / 1.6;
            } else if (abs(local.x) < 0.1 && local.y > 0.15) {
                ln = normalize(vec3<f32>(local.x, 0.0, local.z - 0.2));
                param = 0.6;
            } else {
                // Seven logs along z, 0.36 m apart, axes at y = -0.03.
                let axis_x = round((local.x + 1.08) / 0.36) * 0.36 - 1.08;
                ln = normalize(vec3<f32>(local.x - axis_x, local.y + 0.03, 0.0));
                param = fract(axis_x * 3.7 + 0.5);
            }
        }
        case MAT_PALM_TRUNK: {
            if (local.y < 6.6) {
                // The trunk curves as x = 0.9·t², t = y / 7.
                let t = local.y / 7.0;
                ln = normalize(vec3<f32>(local.x - 0.9 * t * t, 0.0, local.z));
                param = t;
            } else {
                mat = MAT_PALM_FROND;
            }
        }
        case MAT_SAND: {
            let e = 2.0;
            let gx = seabed_height(seabed, frame.seabed, h.xz + vec2<f32>(e, 0.0)) - seabed_height(seabed, frame.seabed, h.xz - vec2<f32>(e, 0.0));
            let gz = seabed_height(seabed, frame.seabed, h.xz + vec2<f32>(0.0, e)) - seabed_height(seabed, frame.seabed, h.xz - vec2<f32>(0.0, e));
            let wn = normalize(vec3<f32>(-gx / (2.0 * e), 1.0, -gz / (2.0 * e)));
            return shade_object(h, wn, MAT_SAND, clamp(h.y / 8.0, 0.0, 1.0));
        }
        default: {}
    }
    var wn = normalize((o2w * vec4<f32>(ln, 0.0)).xyz);
    // Two-sided cloth and leaves: shade the side the ray arrived at.
    if (dot(wn, r) > 0.0 && (mat == MAT_SAIL || mat == MAT_PALM_FROND)) {
        wn = -wn;
    }
    return shade_object(h, wn, mat, param);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let half = textureDimensions(out_tex);
    if (id.x >= half.x || id.y >= half.y) {
        return;
    }
    let full = min(id.xy * 2u + rp.jitter, vec2<u32>(frame.resolution.xy) - 1u);
    let d = textureLoad(depth_tex, vec2<i32>(full), 0);
    if (d <= 0.0) {
        textureStore(out_tex, vec2<i32>(id.xy), vec4<f32>(0.0));
        return;
    }
    let p = world_from_depth(vec2<f32>(full) + 0.5, d);
    let n = textureLoad(normal_tex, vec2<i32>(full), 0).xyz;
    let v = normalize(frame.cam_pos.xyz - p);
    let r = reflect(-v, n);
    let origin = p + n * 0.05;

    // The patch is a heightfield seen from above: culling its (front-wound) underside lets rays
    // that start just below it, where it is coarser than the render mesh, pass through instead
    // of hitting it at once. Objects are traced two-sided in their own query; nearer hit wins.
    var rq_patch: ray_query;
    rayQueryInitialize(&rq_patch, tlas, RayDesc(RAY_FLAG_CULL_FRONT_FACING, PATCH_MASK, T_MIN, T_MAX, origin, r));
    rayQueryProceed(&rq_patch);
    var hit = rayQueryGetCommittedIntersection(&rq_patch);
    var rq_obj: ray_query;
    rayQueryInitialize(&rq_obj, tlas, RayDesc(RAY_FLAG_NONE, OBJECT_MASK, T_MIN, T_MAX, origin, r));
    rayQueryProceed(&rq_obj);
    let obj_hit = rayQueryGetCommittedIntersection(&rq_obj);
    if (obj_hit.kind != RAY_QUERY_INTERSECTION_NONE && (hit.kind == RAY_QUERY_INTERSECTION_NONE || obj_hit.t < hit.t)) {
        hit = obj_hit;
    }

    var col: vec3<f32>;
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        col = sky_radiance(normalize(vec3<f32>(r.x, max(r.y, 0.0), r.z)));
    } else {
        let h = origin + r * hit.t;
        if (hit.instance_custom_data != MAT_WATER) {
            let local = hit.world_to_object * vec4<f32>(h, 1.0);
            col = shade_object_hit(h, local, r, hit.instance_custom_data, hit.object_to_world);
        } else {
            // Facet normal says which side was hit; the smooth normal shades it.
            let facet = patch_normal(hit.primitive_index);
            var n_hit = wave_normal(h);
            if (dot(n_hit, facet) < 0.0) {
                n_hit = facet;
            }
            col = shade_reflected_water(h, n_hit, r);
        }
    }
    textureStore(out_tex, vec2<i32>(id.xy), vec4<f32>(col, 1.0));
}
