// #include frame_common.wgsl
// #include sky_common.wgsl
// #include sky_sample.wgsl
// #include buoy_common.wgsl
// #include surfaces.wgsl
// Full-screen composite into the HDR target: sky where nothing was drawn, physically based
// water elsewhere. Needs frame_common.wgsl, sky_common.wgsl, sky_sample.wgsl.

@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var material_tex: texture_2d<f32>;
@group(0) @binding(4) var reflection_tex: texture_2d<f32>;
@group(0) @binding(5) var lin: sampler;
@group(0) @binding(6) var opaque_depth: texture_depth_2d;
@group(0) @binding(7) var opaque_albedo: texture_2d<f32>;


struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}


fn shade_water(px: vec2<f32>, p: vec3<f32>, n_in: vec3<f32>, variance: f32, foam: f32, traced: vec4<f32>) -> vec3<f32> {
    let to_eye = frame.cam_pos.xyz - p;
    let dist = length(to_eye);
    let v = to_eye / dist;
    // Back-facing normals (steep waves seen edge-on) are bent towards the viewer.
    var n = n_in;
    if (dot(n, v) < 0.02) {
        n = normalize(n + v * (0.02 - dot(n, v)));
    }
    let n_v = max(dot(n, v), 1e-4);
    let sun = frame.sun_dir.xyz;
    let e_sun = sun_transmittance();

    // Reflection: whatever the reflection pass found (other waves, objects), else the sky.
    // Rays reflected below the horizon that found nothing see the horizon.
    let r0 = reflect(-v, n);
    let r = normalize(vec3<f32>(r0.x, max(r0.y, 0.0), r0.z));
    let f = fresnel(n_v);
    let reflected = mix(sky_radiance(r), traced.rgb, traced.a) * f;

    // Sun specular: microfacet GGX over the unresolved slopes.
    let alpha2 = clamp(2.0 * variance, 1e-4, 1.0);
    let h = normalize(v + sun);
    let n_l = dot(n, sun);
    var spec = vec3<f32>(0.0);
    if (n_l > 0.0) {
        let g = smith_g1(n_v, alpha2) * smith_g1(n_l, alpha2);
        spec = e_sun * ggx(max(dot(n, h), 0.0), alpha2) * g * fresnel(dot(v, h)) / (4.0 * n_v);
    }

    // Light scattered back up out of the deep water column (single scattering, deep water).
    let albedo = WATER_SCATTER / (WATER_ABSORB + WATER_SCATTER);
    let e_in = e_sun * max(sun.y, 0.0) * (1.0 - fresnel(max(sun.y, 0.0))) + sky_irradiance() * 0.94;
    var body = (1.0 - f) * e_in * albedo / (4.0 * PI) * 1.3;

    // Refraction: anything opaque under this bit of surface (the buoy's hull) shows through,
    // dimmed and tinted by the water it is seen through. Screen-space offset by the normal.
    let rp = clamp(px + n.xz * (40.0 / max(dist, 1.0)) * frame.resolution.x / 1280.0, vec2<f32>(0.0), frame.resolution.xy - 1.0);
    let od = textureLoad(opaque_depth, vec2<i32>(rp), 0);
    if (od > 0.0) {
        let q = world_from_depth(rp, od);
        let path = length(q - p);
        if (q.y < p.y && length(q - frame.cam_pos.xyz) > dist) {
            let t = exp(-(WATER_ABSORB + WATER_SCATTER) * path);
            let lit = textureLoad(opaque_albedo, vec2<i32>(rp), 0).rgb / PI * e_in * t;
            body = (1.0 - f) * lit + body * (vec3<f32>(1.0) - t);
        }
    }

    // Sunlight through thin wave crests toward the eye: the green-gold glow when looking
    // into a low sun.
    let crest = clamp(p.y / max(frame.misc.y * 0.5, 0.05), 0.0, 1.0);
    let toward_sun = pow(clamp(dot(-v, normalize(vec3<f32>(sun.x, 0.0, sun.z))), 0.0, 1.0), 4.0);
    let sss = e_sun * CREST_TINT * crest * crest * toward_sun * (1.0 - f) * 0.12 * max(sun.y + 0.1, 0.0);

    // Foam: a bright diffuse layer that hides the reflection and the water beneath it.
    let foam_light = FOAM_ALBEDO / PI * (e_sun * max(n_l, 0.0) + sky_irradiance());
    let water = reflected + spec + body + sss;
    return aerial(mix(water, foam_light, foam), -v, dist);
}

@fragment
fn fs_shade(in: VsOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let depth = textureLoad(depth_tex, px, 0);
    if (depth <= 0.0) {
        let dir = view_dir_for_pixel(in.pos.xy);
        if (dir.y < 0.0) {
            // Past the end of the mesh (or through a sub-pixel crack) below the horizon: the
            // sea there is so far away that only the aerial-perspective horizon light remains.
            return vec4<f32>(sky_radiance(normalize(vec3<f32>(dir.x, 0.01, dir.z))), 1.0);
        }
        return vec4<f32>(sky_radiance(dir) + sun_disc(dir), 1.0);
    }
    let p = world_from_depth(in.pos.xy, depth);
    let nv = textureLoad(normal_tex, px, 0);
    let foam = textureLoad(material_tex, px, 0).r;
    let traced = textureSampleLevel(reflection_tex, lin, in.pos.xy * frame.resolution.zw, 0.0);
    let mat = textureLoad(material_tex, px, 0);
    if (mat.g > 0.5) {
        return vec4<f32>(shade_buoy(p, nv.xyz, mat.b * 4.0 - 0.5), 1.0);
    }
    return vec4<f32>(shade_water(in.pos.xy, p, nv.xyz, nv.w, foam, traced), 1.0);
}

