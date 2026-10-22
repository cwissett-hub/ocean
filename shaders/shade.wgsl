// #include frame_common.wgsl
// #include sky_common.wgsl
// #include sky_sample.wgsl
// #include materials.wgsl
// #include surfaces.wgsl
// #include seabed_common.wgsl
// Full-screen composite into the HDR target: sky where nothing was drawn, physically based
// water elsewhere. Needs frame_common.wgsl, sky_common.wgsl, sky_sample.wgsl.

@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var material_tex: texture_2d<f32>;
@group(0) @binding(4) var reflection_tex: texture_2d<f32>;
@group(0) @binding(5) var lin: sampler;
@group(0) @binding(6) var opaque_depth: texture_depth_2d;
@group(0) @binding(7) var opaque_albedo: texture_2d<f32>;
@group(0) @binding(8) var seabed: texture_2d<f32>;


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

    // Sun specular: microfacet Beckmann over the unresolved slopes.
    let alpha2 = clamp(2.0 * variance, 1e-4, 1.0);
    let h = normalize(v + sun);
    let n_l = dot(n, sun);
    var spec = vec3<f32>(0.0);
    if (n_l > 0.0) {
        let g = smith_g1(n_v, alpha2) * smith_g1(n_l, alpha2);
        spec = e_sun * beckmann(max(dot(n, h), 0.0), max(variance, 5e-5)) * g * fresnel(dot(v, h)) / (4.0 * n_v);
    }

    // Light scattered back up out of the water column (infinitely deep), and the light that
    // gets into the water to reach anything below.
    let e_in = underwater_irradiance();
    let column = e_in * water_column_reflectance();

    // The bottom: the sandy seabed where the refracted view ray meets it (one step along the
    // ray from the depth straight below), unless the opaque pass drew something nearer under
    // this bit of surface (a raft log, the buoy's hull, an island's beach), screen-space
    // offset by the normal. Light reaches it down through the water and comes back up along
    // the view ray, so the shallows go from sand through turquoise to the deep blue.
    let ext = WATER_ABSORB + WATER_SCATTER;
    let down = refract(-v, n, 1.0 / 1.333);
    let cos_down = max(-down.y, 0.15);
    let below = max(p.y - seabed_height(seabed, frame.seabed, p.xz), 0.0);
    let bed_xz = p.xz + down.xz * (below / cos_down);
    var depth = max(p.y - seabed_height(seabed, frame.seabed, bed_xz), 0.0);
    var view_path = depth / cos_down;
    var bottom = material_albedo(MAT_SAND, 0.0);
    let rp = clamp(px + n.xz * (40.0 / max(dist, 1.0)) * frame.resolution.x / 1280.0, vec2<f32>(0.0), frame.resolution.xy - 1.0);
    let od = textureLoad(opaque_depth, vec2<i32>(rp), 0);
    if (od > 0.0) {
        let q = world_from_depth(rp, od);
        if (q.y < p.y && length(q - frame.cam_pos.xyz) > dist && length(q - p) < view_path + 1.0) {
            bottom = textureLoad(opaque_albedo, vec2<i32>(rp), 0).rgb;
            depth = p.y - q.y;
            view_path = length(q - p);
        }
    }
    let seen = exp(-ext * view_path);
    // Radiance seen through the surface is divided by n², like the column's.
    let bottom_lit = bottom / PI * e_in * exp(-ext * depth) * seen / WATER_N2;
    let body = (1.0 - f) * (bottom_lit + column * (vec3<f32>(1.0) - seen));

    // Subsurface scattering (the stylised-physical model of the Atlas / Sea of Thieves water
    // talks): light scattered inside the wave bodies. Back-lit crests glow when looking toward
    // the sun — stronger on tall crests and on faces turned away from it — the water facing the
    // eye carries a softer teal glow, and sunlit faces get a little diffuse lift.
    let height = clamp(p.y / max(frame.misc.y, 0.1) + 0.5, 0.0, 1.5);
    let back = height * height * pow(max(dot(sun, -v), 0.0), 4.0) * pow(0.5 - 0.5 * dot(sun, n), 3.0);
    let front = pow(max(dot(v, n), 0.0), 2.0);
    let sss = (1.0 - f) * SSS_TINT * (e_sun * (SSS_BACK * back + SSS_SUN * max(n_l, 0.0)) + e_in * SSS_FRONT * front);

    // Foam: a thick bubbly layer that hides the reflection and the water beneath it. Light
    // wraps round it, the side away from the sun is lit blue-white by the sky, and back-lit
    // foam glows a little as light scatters through its bubbles.
    let wrap = clamp((n_l + 0.35) / 1.35, 0.0, 1.0);
    let through = pow(max(dot(-v, sun), 0.0), 3.0) * 0.35;
    let sky_tint = vec3<f32>(0.82, 0.92, 1.0);
    let foam_light = FOAM_ALBEDO / PI * (e_sun * (wrap + through) + sky_irradiance() * sky_tint);
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
    let id = decode_material(mat.g);
    if (id != MAT_WATER) {
        return vec4<f32>(shade_object(p, nv.xyz, id, mat.b), 1.0);
    }
    return vec4<f32>(shade_water(in.pos.xy, p, nv.xyz, nv.w, foam, traced), 1.0);
}

