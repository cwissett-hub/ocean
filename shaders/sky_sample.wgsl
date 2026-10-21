// Group-1 sky bindings and lookups for any pass that needs sky light. Needs sky_common.wgsl.

struct SkyUniforms {
    sun_dir: vec4<f32>,
};

@group(1) @binding(0) var<uniform> sky_u: SkyUniforms;
@group(1) @binding(1) var sky_transmittance: texture_2d<f32>;
@group(1) @binding(2) var sky_view_lut: texture_2d<f32>;
@group(1) @binding(3) var sky_clamp: sampler;
@group(1) @binding(4) var sky_wrap: sampler;

const SUN_ANGULAR_RADIUS: f32 = 0.0087; // 0.5°, a little larger than the real 0.27°

// Sky radiance (no sun disc) seen from the camera along dir, for unit solar illuminance.
fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let d = normalize(vec3<f32>(dir.x, max(dir.y, -0.02), dir.z));
    return textureSampleLevel(sky_view_lut, sky_wrap, sky_view_uv(d, sky_u.sun_dir.xyz), 0.0).rgb;
}

// Fraction of sunlight reaching sea level (camera altitude).
fn sun_transmittance() -> vec3<f32> {
    let r = R_GROUND + CAMERA_ALT_KM;
    let t = textureSampleLevel(sky_transmittance, sky_clamp, transmittance_uv(r, sky_u.sun_dir.y), 0.0).rgb;
    // Below the horizon the sun is hidden by the Earth itself.
    return t * smoothstep(-SUN_ANGULAR_RADIUS, SUN_ANGULAR_RADIUS, sky_u.sun_dir.y);
}

// Sun disc radiance with limb darkening (illuminance 1 spread over the disc).
fn sun_disc(dir: vec3<f32>) -> vec3<f32> {
    let c = dot(dir, sky_u.sun_dir.xyz);
    let cos_r = cos(SUN_ANGULAR_RADIUS);
    if (c < cos_r) {
        return vec3<f32>(0.0);
    }
    let x = sqrt(max(1.0 - (1.0 - c) / (1.0 - cos_r), 0.0));
    let limb = 0.4 + 0.6 * x;
    let solid = 2.0 * PI * (1.0 - cos_r);
    return sun_transmittance() * limb / (solid * 0.8);
}

// Approximate downwelling sky irradiance on a horizontal surface.
fn sky_irradiance() -> vec3<f32> {
    let s = sky_u.sun_dir.xyz;
    let toward = normalize(vec3<f32>(s.x, 0.15, s.z));
    let away = normalize(vec3<f32>(-s.x, 0.15, -s.z));
    let side = normalize(vec3<f32>(s.z, 0.15, -s.x));
    let l = 0.4 * sky_radiance(vec3<f32>(0.0, 1.0, 0.0)) + 0.2 * sky_radiance(toward)
        + 0.2 * sky_radiance(away) + 0.2 * sky_radiance(side);
    return PI * l;
}
