// Full-screen composite into the HDR target: sky where nothing was drawn, physically based
// water elsewhere. Needs frame_common.wgsl, sky_common.wgsl, sky_sample.wgsl.

@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var material_tex: texture_2d<f32>;
@group(0) @binding(4) var reflection_tex: texture_2d<f32>;
@group(0) @binding(5) var lin: sampler;

const WATER_F0: f32 = 0.02;
// Seawater absorption and scattering, 1/m (pure-water absorption + a little particulate).
const WATER_ABSORB: vec3<f32> = vec3<f32>(0.45, 0.065, 0.025);
const WATER_SCATTER: vec3<f32> = vec3<f32>(0.012, 0.018, 0.026);
const CREST_TINT: vec3<f32> = vec3<f32>(0.08, 0.55, 0.42);
const FOAM_ALBEDO: f32 = 0.8;

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

fn fresnel(cos_t: f32) -> f32 {
    return WATER_F0 + (1.0 - WATER_F0) * pow(1.0 - clamp(cos_t, 0.0, 1.0), 5.0);
}

// GGX distribution with α² = 2σ² (matching Beckmann slope variance σ²).
fn ggx(n_h: f32, alpha2: f32) -> f32 {
    let d = n_h * n_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d);
}

fn smith_g1(n_x: f32, alpha2: f32) -> f32 {
    return 2.0 * n_x / (n_x + sqrt(alpha2 + (1.0 - alpha2) * n_x * n_x));
}

// Light lost and gained along the view path near sea level (aerial perspective).
fn aerial(col: vec3<f32>, view: vec3<f32>, dist_m: f32) -> vec3<f32> {
    let m = medium(0.0);
    let t = exp(-m.extinction * dist_m * 0.001);
    let horizon = sky_radiance(normalize(vec3<f32>(view.x, 0.01, view.z)));
    return col * t + horizon * (vec3<f32>(1.0) - t);
}

fn shade_water(p: vec3<f32>, n_in: vec3<f32>, variance: f32, foam: f32, traced: vec4<f32>) -> vec3<f32> {
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
    let body = (1.0 - f) * e_in * albedo / (4.0 * PI) * 1.3;

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
    return vec4<f32>(shade_water(p, nv.xyz, nv.w, foam, traced), 1.0);
}
