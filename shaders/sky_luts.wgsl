// #include sky_common.wgsl
// Hillaire 2020, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique".
// Distances in km. Sky radiance is for unit solar illuminance (exposure scales it later).
// Entry points: transmittance (256x64), multiscatter (32x32), sky_view (192x108).
// Prepended with sky_common.wgsl at load time.


struct SkyParams {
    sun_dir: vec4<f32>,
};

@group(0) @binding(0) var<uniform> sky: SkyParams;
@group(0) @binding(1) var transmittance_in: texture_2d<f32>;
@group(0) @binding(2) var multiscatter_in: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var lut_out: texture_storage_2d<rgba16float, write>;

fn transmittance_to_sun(r: f32, mu: f32) -> vec3<f32> {
    return textureSampleLevel(transmittance_in, lut_sampler, transmittance_uv(r, mu), 0.0).rgb;
}

fn multiscatter(r: f32, mu_s: f32) -> vec3<f32> {
    let uv = vec2<f32>(mu_s * 0.5 + 0.5, (r - R_GROUND) / (R_TOP - R_GROUND));
    return textureSampleLevel(multiscatter_in, lut_sampler, uv, 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn transmittance(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(lut_out);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let h = sqrt(R_TOP * R_TOP - R_GROUND * R_GROUND);
    let rho = h * uv.y;
    let r = sqrt(rho * rho + R_GROUND * R_GROUND);
    let d_min = R_TOP - r;
    let d_max = rho + h;
    let d = d_min + uv.x * (d_max - d_min);
    var mu = 1.0;
    if (d > 0.0) {
        mu = clamp((h * h - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    }
    let steps = 40;
    let dt = d / f32(steps);
    var depth = vec3<f32>(0.0);
    for (var i = 0; i < steps; i++) {
        let t = (f32(i) + 0.5) * dt;
        let alt = sqrt(r * r + t * t + 2.0 * r * mu * t) - R_GROUND;
        depth += medium(alt).extinction * dt;
    }
    textureStore(lut_out, vec2<i32>(id.xy), vec4<f32>(exp(-depth), 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn multiscatter_lut(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(lut_out);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let mu_s = uv.x * 2.0 - 1.0;
    let r = R_GROUND + uv.y * (R_TOP - R_GROUND);
    let sun = vec3<f32>(sqrt(max(1.0 - mu_s * mu_s, 0.0)), mu_s, 0.0);
    var l2 = vec3<f32>(0.0);
    var fms = vec3<f32>(0.0);
    let n = 8;
    for (var a = 0; a < n; a++) {
        for (var b = 0; b < n; b++) {
            let cos_t = 1.0 - 2.0 * (f32(a) + 0.5) / f32(n);
            let phi = 2.0 * PI * (f32(b) + 0.5) / f32(n);
            let sin_t = sqrt(1.0 - cos_t * cos_t);
            let dir = vec3<f32>(sin_t * cos(phi), cos_t, sin_t * sin(phi));
            let ground = ray_sphere(r, dir.y, R_GROUND);
            var t_max = ray_sphere(r, dir.y, R_TOP);
            if (ground > 0.0) {
                t_max = ground;
            }
            let steps = 20;
            let dt = t_max / f32(steps);
            var trans = vec3<f32>(1.0);
            var lum = vec3<f32>(0.0);
            var f = vec3<f32>(0.0);
            for (var i = 0; i < steps; i++) {
                let t = (f32(i) + 0.5) * dt;
                let p = vec3<f32>(0.0, r, 0.0) + dir * t;
                let pr = length(p);
                let m = medium(pr - R_GROUND);
                let scatter = m.rayleigh + vec3<f32>(m.mie);
                let step_t = exp(-m.extinction * dt);
                let integ = (vec3<f32>(1.0) - step_t) / max(m.extinction, vec3<f32>(1e-7));
                let mu = dot(p / pr, sun);
                let ts = transmittance_to_sun(pr, mu);
                let shadow = select(1.0, 0.0, ray_sphere(pr, mu, R_GROUND) > 0.0);
                lum += trans * integ * scatter * ts * shadow / (4.0 * PI);
                f += trans * integ * scatter;
                trans *= step_t;
            }
            l2 += lum;
            fms += f;
        }
    }
    let count = f32(n * n);
    l2 /= count;
    fms /= count;
    let psi = l2 / (vec3<f32>(1.0) - fms);
    textureStore(lut_out, vec2<i32>(id.xy), vec4<f32>(psi, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn sky_view(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(lut_out);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let sun = sky.sun_dir.xyz;
    let l = uv.y * 2.0 - 1.0;
    let elev = sign(l) * l * l * 0.5 * PI;
    let az = uv.x * 2.0 * PI + atan2(sun.x, sun.z);
    let dir = vec3<f32>(sin(az) * cos(elev), sin(elev), cos(az) * cos(elev));
    let r = R_GROUND + CAMERA_ALT_KM;
    let ground = ray_sphere(r, dir.y, R_GROUND);
    var t_max = ray_sphere(r, dir.y, R_TOP);
    if (ground > 0.0) {
        t_max = ground;
    }
    let cos_vs = dot(dir, sun);
    let pr_r = rayleigh_phase(cos_vs);
    let pr_m = mie_phase(cos_vs);
    let steps = 32;
    var trans = vec3<f32>(1.0);
    var lum = vec3<f32>(0.0);
    var t0 = 0.0;
    for (var i = 0; i < steps; i++) {
        // Quadratic step distribution: dense near the eye where the air is thick.
        let t1 = t_max * pow((f32(i) + 1.0) / f32(steps), 2.0);
        let dt = t1 - t0;
        let t = 0.5 * (t0 + t1);
        t0 = t1;
        let p = vec3<f32>(0.0, r, 0.0) + dir * t;
        let pr = length(p);
        let m = medium(pr - R_GROUND);
        let mu = dot(p / pr, sun);
        let ts = transmittance_to_sun(pr, mu);
        let shadow = select(1.0, 0.0, ray_sphere(pr, mu, R_GROUND) > 0.0);
        let single = (m.rayleigh * pr_r + vec3<f32>(m.mie * pr_m)) * ts * shadow;
        let multi = (m.rayleigh + vec3<f32>(m.mie)) * multiscatter(pr, mu);
        let step_t = exp(-m.extinction * dt);
        let integ = (vec3<f32>(1.0) - step_t) / max(m.extinction, vec3<f32>(1e-7));
        lum += trans * integ * (single + multi);
        trans *= step_t;
    }
    textureStore(lut_out, vec2<i32>(id.xy), vec4<f32>(lum, 1.0));
}
