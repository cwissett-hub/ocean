// Surface shading shared by the main composite and ray-traced reflection hits.
// Needs frame_common, sky_common, sky_sample, materials.

const WATER_F0: f32 = 0.02;
// Seawater absorption and scattering, 1/m (pure-water absorption + a little particulate).
const WATER_ABSORB: vec3<f32> = vec3<f32>(0.45, 0.065, 0.025);
const WATER_SCATTER: vec3<f32> = vec3<f32>(0.012, 0.018, 0.026);
// n² for seawater: radiance leaving the water is divided by it.
const WATER_N2: f32 = 1.78;
// Subsurface scattering (shade.wgsl): tint of light that has travelled through wave bodies,
// and the strength of its back-lit, sunlit and view-facing parts.
const SSS_TINT: vec3<f32> = vec3<f32>(0.03, 0.42, 0.52);
const SSS_BACK: f32 = 0.8;
const SSS_SUN: f32 = 0.015;
const SSS_FRONT: f32 = 0.09;
const FOAM_ALBEDO: f32 = 0.8;

fn fresnel(cos_t: f32) -> f32 {
    return WATER_F0 + (1.0 - WATER_F0) * pow(1.0 - clamp(cos_t, 0.0, 1.0), 5.0);
}

// Beckmann distribution for mean-square slope m2: Gaussian slopes, as Cox–Munk measured on
// the sea. Same width as GGX but without its long tails, which spread the sun into a milky
// haze around the glitter path.
fn beckmann(n_h: f32, m2: f32) -> f32 {
    let c2 = max(n_h * n_h, 1e-4);
    return exp((c2 - 1.0) / (m2 * c2)) / (PI * m2 * c2 * c2);
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

fn wood_hash(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(0.1031, 0.1030));
    let r = q + dot(q, q.yx + 33.33);
    return fract((r.x + r.y) * r.x);
}

fn wood_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(wood_hash(i), wood_hash(i + vec2<f32>(1.0, 0.0)), u.x), mix(wood_hash(i + vec2<f32>(0.0, 1.0)), wood_hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

struct Surface {
    albedo: vec3<f32>,
    variance: f32,
};

// The raft's driftwood (raft frame: seven logs along z, axes 0.36 m apart at y = −0.03;
// mast and yard above). Sun-bleached silver-brown with grain streaking along each log,
// cracks, knots and growth rings on the cut ends; below the wet line (scene::wetness, per
// sample column) it is darker, richer and glossy.
fn raft_wood(p: vec3<f32>, n: vec3<f32>, param: f32) -> Surface {
    let l = (frame.raft_inv * vec4<f32>(p, 1.0)).xyz;
    let ln = normalize((frame.raft_inv * vec4<f32>(n, 0.0)).xyz);
    var radial: vec2<f32>;
    var along: f32;
    if (abs(l.x) < 0.12 && l.y > 0.12) {
        // Mast (vertical) and yard (across).
        radial = vec2<f32>(l.x, l.z - 0.2);
        along = select(l.y, l.x, l.y > 2.5);
    } else {
        let axis = round((l.x + 1.08) / 0.36) * 0.36 - 1.08;
        radial = vec2<f32>(l.x - axis, l.y + 0.03);
        along = l.z;
    }
    let around = atan2(radial.y, radial.x);
    let seed = param * 17.0;
    let grain = wood_noise(vec2<f32>(along * 1.2, around * 4.0 + seed)) * 0.55 + wood_noise(vec2<f32>(along * 5.0, around * 14.0 + seed)) * 0.45;
    let crack = smoothstep(0.86, 0.95, wood_noise(vec2<f32>(along * 0.5, around * 10.0 + seed * 3.0)));
    let knot = smoothstep(0.88, 0.97, wood_noise(vec2<f32>(along * 2.2 + seed, around * 1.5)));
    let bleached = mix(vec3<f32>(0.40, 0.36, 0.31), vec3<f32>(0.66, 0.62, 0.56), grain) * mix(0.9, 1.1, fract(param * 7.3));
    var albedo = bleached * (1.0 - 0.55 * crack) * (1.0 - 0.35 * knot);
    // Cut ends show their growth rings.
    let end = smoothstep(0.7, 0.9, abs(ln.z)) * select(1.0, 0.0, abs(l.x) < 0.12 && l.y > 0.12);
    let rings = 0.5 + 0.5 * sin(length(radial) * 160.0 + wood_noise(radial * 30.0) * 3.0);
    albedo = mix(albedo, vec3<f32>(0.55, 0.44, 0.31) * (0.8 + 0.2 * rings), end);
    // Wet line: bilinear over the 3×3 water-sample columns (x −1.2..1.2, z −1.5..1.5).
    let u = clamp((l.x + 1.2) / 1.2, 0.0, 2.0);
    let w = clamp((l.z + 1.5) / 1.5, 0.0, 2.0);
    let i = min(vec2<u32>(u32(u), u32(w)), vec2<u32>(1u));
    let f = vec2<f32>(u, w) - vec2<f32>(i);
    let k = i.y * 3u + i.x;
    let line = mix(mix(raft_wet_line(k), raft_wet_line(k + 1u), f.x), mix(raft_wet_line(k + 3u), raft_wet_line(k + 4u), f.x), f.y);
    let wet = smoothstep(line + 0.02, line - 0.04, l.y);
    let wet_albedo = pow(albedo, vec3<f32>(1.4)) * 0.45;
    return Surface(mix(albedo, wet_albedo, wet), mix(0.35, 0.06, wet));
}

fn raft_wet_line(k: u32) -> f32 {
    return frame.raft_wet[k / 4u][k % 4u];
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
    var albedo = material_albedo(id, param);
    var variance = material_variance(id);
    if (id == MAT_RAFT_WOOD) {
        let s = raft_wood(p, n, param);
        albedo = s.albedo;
        variance = s.variance;
    }
    let n_l = max(dot(n, sun), 0.0);
    let n_v = max(dot(n, v), 1e-4);
    let ambient = sky_irradiance() * (0.6 + 0.4 * n.y) + sky_radiance(normalize(vec3<f32>(-sun.x, 0.1, -sun.z))) * PI * 0.15 * max(-n.y, 0.0);
    var col = albedo / PI * (e_sun * n_l + ambient);
    let alpha2 = clamp(2.0 * variance, 1e-3, 1.0);
    let h = normalize(v + sun);
    if (n_l > 0.0) {
        let g = smith_g1(n_v, alpha2) * smith_g1(n_l, alpha2);
        let f0 = 0.04 + (1.0 - 0.04) * pow(1.0 - clamp(dot(v, h), 0.0, 1.0), 5.0);
        col += e_sun * ggx(max(dot(n, h), 0.0), alpha2) * g * f0 / (4.0 * n_v);
    }
    let f = 0.04 + 0.96 * pow(1.0 - n_v, 5.0);
    // Reflected surroundings: sky above the horizon, the darker sea below it. Rough surfaces
    // blur the two across a wide band — a sharp mirrored horizon drew a hard-edged dome across
    // the sailcloth.
    let rr = reflect(-v, n);
    let sky_seen = sky_radiance(normalize(vec3<f32>(rr.x, max(rr.y, 0.02), rr.z)));
    let blur = mix(0.03, 0.6, clamp(alpha2, 0.0, 1.0));
    let env = sky_seen * mix(0.3, 1.0, smoothstep(-blur, blur, rr.y));
    col += env * f * select(0.5, 0.15, alpha2 > 0.3);
    if (id == MAT_SAIL || id == MAT_PALM_FROND) {
        // Thin cloth / leaves: light through them toward the eye.
        let through = pow(max(dot(-v, sun), 0.0), 4.0);
        col += albedo * e_sun * through * select(0.25, 0.6, id == MAT_SAIL);
    }
    col += material_emission(id, param) * 0.4 * (1.0 - smoothstep(-6.0, 20.0, degrees(asin(sun.y))));
    return aerial(col, -v, dist);
}
