// #include frame_common.wgsl
// #include sky_common.wgsl
// #include sky_sample.wgsl
// #include materials.wgsl
// #include surfaces.wgsl
enable wgpu_ray_query;

// Half-resolution hardware ray-traced reflections: one mirror ray per pixel from the
// G-buffer surface into the TLAS (near-field wave patch + buoy). Hits are shaded with a
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
const BUOY_ID: u32 = 1u;
const PATCH_MASK: u32 = 0x01u;
const BUOY_MASK: u32 = 0x02u;

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
    // of hitting it at once. The buoy is traced two-sided in its own query; nearer hit wins.
    var rq_patch: ray_query;
    rayQueryInitialize(&rq_patch, tlas, RayDesc(RAY_FLAG_CULL_FRONT_FACING, PATCH_MASK, T_MIN, T_MAX, origin, r));
    rayQueryProceed(&rq_patch);
    var hit = rayQueryGetCommittedIntersection(&rq_patch);
    var rq_buoy: ray_query;
    rayQueryInitialize(&rq_buoy, tlas, RayDesc(RAY_FLAG_NONE, BUOY_MASK, T_MIN, T_MAX, origin, r));
    rayQueryProceed(&rq_buoy);
    let buoy_hit = rayQueryGetCommittedIntersection(&rq_buoy);
    if (buoy_hit.kind != RAY_QUERY_INTERSECTION_NONE && (hit.kind == RAY_QUERY_INTERSECTION_NONE || buoy_hit.t < hit.t)) {
        hit = buoy_hit;
    }

    var col: vec3<f32>;
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        col = sky_radiance(normalize(vec3<f32>(r.x, max(r.y, 0.0), r.z)));
    } else {
        let h = origin + r * hit.t;
        if (hit.instance_custom_data == BUOY_ID) {
            let local = hit.world_to_object * vec4<f32>(h, 1.0);
            // Cylinder-ish normal from the local position (sides) or ±y (caps).
            var ln = normalize(vec3<f32>(local.x, 0.0, local.z));
            let near_cap = abs(local.y - 0.47) < 0.02 || abs(local.y - 3.26) < 0.02 || abs(local.y - 3.5) < 0.02;
            if (near_cap) {
                ln = vec3<f32>(0.0, 1.0, 0.0);
            }
            let wn = normalize((hit.object_to_world * vec4<f32>(ln, 0.0)).xyz);
            col = shade_object(h, wn, MAT_BUOY, (local.y + 0.5) / 4.0);
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
