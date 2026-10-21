// ---- sky_common --------------------------------------------------------------------------
const PI: f32 = 3.14159265;
const R_GROUND: f32 = 6360.0;
const R_TOP: f32 = 6460.0;
const RAYLEIGH_SCATTER: vec3<f32> = vec3<f32>(5.802e-3, 13.558e-3, 33.1e-3);
const RAYLEIGH_H: f32 = 8.0;
const MIE_SCATTER: f32 = 3.996e-3;
const MIE_ABSORB: f32 = 4.40e-3;
const MIE_H: f32 = 1.2;
const MIE_G: f32 = 0.8;
const OZONE_ABSORB: vec3<f32> = vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3);
const CAMERA_ALT_KM: f32 = 0.02;

struct Medium {
    rayleigh: vec3<f32>,
    mie: f32,
    extinction: vec3<f32>,
};

fn medium(alt: f32) -> Medium {
    let r = exp(-alt / RAYLEIGH_H);
    let m = exp(-alt / MIE_H);
    let o = max(0.0, 1.0 - abs(alt - 25.0) / 15.0);
    var out: Medium;
    out.rayleigh = RAYLEIGH_SCATTER * r;
    out.mie = MIE_SCATTER * m;
    out.extinction = out.rayleigh + vec3<f32>((MIE_SCATTER + MIE_ABSORB) * m) + OZONE_ABSORB * o;
    return out;
}

// Distance along a ray from radius r with cosine-of-zenith mu to the sphere of radius R
// (positive root), or -1 if missed.
fn ray_sphere(r: f32, mu: f32, radius: f32) -> f32 {
    let b = r * mu;
    let c = r * r - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    let s = sqrt(disc);
    let t1 = -b - s;
    let t2 = -b + s;
    if (t1 > 0.0) {
        return t1;
    }
    if (t2 > 0.0) {
        return t2;
    }
    return -1.0;
}

// Bruneton's transmittance parameterisation.
fn transmittance_uv(r: f32, mu: f32) -> vec2<f32> {
    let h = sqrt(R_TOP * R_TOP - R_GROUND * R_GROUND);
    let rho = sqrt(max(r * r - R_GROUND * R_GROUND, 0.0));
    let disc = r * r * (mu * mu - 1.0) + R_TOP * R_TOP;
    let d = max(-r * mu + sqrt(max(disc, 0.0)), 0.0);
    let d_min = R_TOP - r;
    let d_max = rho + h;
    return vec2<f32>((d - d_min) / (d_max - d_min), rho / h);
}

fn rayleigh_phase(c: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + c * c);
}

fn mie_phase(c: f32) -> f32 {
    // Cornette–Shanks.
    let g2 = MIE_G * MIE_G;
    let k = 3.0 / (8.0 * PI) * (1.0 - g2) / (2.0 + g2);
    return k * (1.0 + c * c) / pow(1.0 + g2 - 2.0 * MIE_G * c, 1.5);
}

// Sky-view LUT mapping: u = azimuth from the sun / 2π, v puts the horizon at 0.5 and packs
// texels near it (elevation ∝ (2v − 1)²).
fn sky_view_uv(dir: vec3<f32>, sun: vec3<f32>) -> vec2<f32> {
    let elev = asin(clamp(dir.y, -1.0, 1.0));
    let az_dir = atan2(dir.x, dir.z);
    let az_sun = atan2(sun.x, sun.z);
    var rel = az_dir - az_sun;
    rel = rel - floor(rel / (2.0 * PI)) * 2.0 * PI;
    let v = 0.5 + 0.5 * sign(elev) * sqrt(abs(elev) / (0.5 * PI));
    return vec2<f32>(rel / (2.0 * PI), v);
}
// ---- end sky_common ----------------------------------------------------------------------
