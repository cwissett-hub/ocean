// Surface shading shared by the main composite and ray-traced reflection hits.
// Needs frame_common, sky_common, sky_sample, materials.

const WATER_F0: f32 = 0.02;
// Seawater absorption and scattering, 1/m (pure-water absorption + a little particulate).
const WATER_ABSORB: vec3<f32> = vec3<f32>(0.45, 0.065, 0.025);
const WATER_SCATTER: vec3<f32> = vec3<f32>(0.012, 0.018, 0.026);
// n² for seawater: radiance leaving the water is divided by it.
const WATER_N2: f32 = 1.78;
const CREST_TINT: vec3<f32> = vec3<f32>(0.08, 0.55, 0.42);
const FOAM_ALBEDO: f32 = 0.8;

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

// Upwelling radiance just above the surface per unit downwelling irradiance just below it,
// for an infinitely deep column, 1/sr (Gordon et al. 1988; mirrors
// render::water_optics::water_column_reflectance). Clear seawater is dim and blue: most light
// that enters is absorbed, which is what makes the sea dark and its reflections stand out.
fn water_column_reflectance() -> vec3<f32> {
    let bb = WATER_SCATTER * 0.5;
    let u = bb / max(WATER_ABSORB + bb, vec3<f32>(1e-6));
    return (0.0949 + 0.0794 * u) * u / WATER_N2;
}

// Light scattered back up out of deep water, before the (1 − F) factor.
fn deep_water_upwelling() -> vec3<f32> {
    return underwater_irradiance() * water_column_reflectance();
}

// Downwelling light just below the surface (sun through the surface plus sky).
fn underwater_irradiance() -> vec3<f32> {
    let sun = frame.sun_dir.xyz;
    return sun_transmittance() * max(sun.y, 0.0) * (1.0 - fresnel(max(sun.y, 0.0))) + sky_irradiance() * 0.94;
}

// Opaque objects (buoy, raft, sand, palms): diffuse sun and sky, a GGX highlight, the sky in
// their sheen, sunlight glowing through the sail when it is between the eye and a low sun, and
// the buoy's lantern at dusk.
fn shade_object(p: vec3<f32>, n: vec3<f32>, id: u32, param: f32) -> vec3<f32> {
    let to_eye = frame.cam_pos.xyz - p;
    let dist = length(to_eye);
    let v = to_eye / dist;
    let sun = frame.sun_dir.xyz;
    let e_sun = sun_transmittance();
    let albedo = material_albedo(id, param);
    let n_l = max(dot(n, sun), 0.0);
    let n_v = max(dot(n, v), 1e-4);
    let ambient = sky_irradiance() * (0.6 + 0.4 * n.y) + sky_radiance(normalize(vec3<f32>(-sun.x, 0.1, -sun.z))) * PI * 0.15 * max(-n.y, 0.0);
    var col = albedo / PI * (e_sun * n_l + ambient);
    let alpha2 = clamp(2.0 * material_variance(id), 1e-3, 1.0);
    let h = normalize(v + sun);
    if (n_l > 0.0) {
        let g = smith_g1(n_v, alpha2) * smith_g1(n_l, alpha2);
        let f0 = 0.04 + (1.0 - 0.04) * pow(1.0 - clamp(dot(v, h), 0.0, 1.0), 5.0);
        col += e_sun * ggx(max(dot(n, h), 0.0), alpha2) * g * f0 / (4.0 * n_v);
    }
    let f = 0.04 + 0.96 * pow(1.0 - n_v, 5.0);
    col += sky_radiance(reflect(-v, n)) * f * select(0.5, 0.15, alpha2 > 0.3);
    if (id == MAT_SAIL || id == MAT_PALM_FROND) {
        // Thin cloth / leaves: light through them toward the eye.
        let through = pow(max(dot(-v, sun), 0.0), 4.0);
        col += albedo * e_sun * through * select(0.25, 0.6, id == MAT_SAIL);
    }
    col += material_emission(id, param) * 0.4 * (1.0 - smoothstep(-6.0, 20.0, degrees(asin(sun.y))));
    return aerial(col, -v, dist);
}
