// Surface shading shared by the main composite and ray-traced reflection hits.
// Needs frame_common, sky_common, sky_sample, buoy_common.

const WATER_F0: f32 = 0.02;
// Seawater absorption and scattering, 1/m (pure-water absorption + a little particulate).
const WATER_ABSORB: vec3<f32> = vec3<f32>(0.45, 0.065, 0.025);
const WATER_SCATTER: vec3<f32> = vec3<f32>(0.012, 0.018, 0.026);
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

// Light scattered back up out of deep water (single scattering), before the (1 − F) factor.
fn deep_water_upwelling() -> vec3<f32> {
    let sun = frame.sun_dir.xyz;
    let albedo = WATER_SCATTER / (WATER_ABSORB + WATER_SCATTER);
    let e_in = sun_transmittance() * max(sun.y, 0.0) * (1.0 - fresnel(max(sun.y, 0.0))) + sky_irradiance() * 0.94;
    return e_in * albedo / (4.0 * PI) * 1.3;
}

// Downwelling light just below the surface (sun through the surface plus sky).
fn underwater_irradiance() -> vec3<f32> {
    let sun = frame.sun_dir.xyz;
    return sun_transmittance() * max(sun.y, 0.0) * (1.0 - fresnel(max(sun.y, 0.0))) + sky_irradiance() * 0.94;
}

// Painted steel: diffuse sun and sky, a soft GGX highlight and the sky in its sheen.
fn shade_buoy(p: vec3<f32>, n: vec3<f32>, local_y: f32) -> vec3<f32> {
    let to_eye = frame.cam_pos.xyz - p;
    let dist = length(to_eye);
    let v = to_eye / dist;
    let sun = frame.sun_dir.xyz;
    let e_sun = sun_transmittance();
    let albedo = buoy_albedo(local_y);
    let n_l = max(dot(n, sun), 0.0);
    let n_v = max(dot(n, v), 1e-4);
    let ambient = sky_irradiance() * (0.6 + 0.4 * n.y) + sky_radiance(normalize(vec3<f32>(-sun.x, 0.1, -sun.z))) * PI * 0.15 * max(-n.y, 0.0);
    var col = albedo / PI * (e_sun * n_l + ambient);
    let alpha2 = 2.0 * BUOY_PAINT_VARIANCE;
    let h = normalize(v + sun);
    if (n_l > 0.0) {
        let g = smith_g1(n_v, alpha2) * smith_g1(n_l, alpha2);
        let f0 = 0.04 + (1.0 - 0.04) * pow(1.0 - clamp(dot(v, h), 0.0, 1.0), 5.0);
        col += e_sun * ggx(max(dot(n, h), 0.0), alpha2) * g * f0 / (4.0 * n_v);
    }
    let f = 0.04 + 0.96 * pow(1.0 - n_v, 5.0);
    col += sky_radiance(reflect(-v, n)) * f * 0.5;
    // The lantern, bright enough to bloom at dusk.
    col += buoy_emission(local_y) * 0.4 * (1.0 - smoothstep(-6.0, 20.0, degrees(asin(sun.y))));
    return aerial(col, -v, dist);
}
