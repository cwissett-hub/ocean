// Albedo, roughness and emission per material id (see src/scene/mesh.rs::material).
// `param` is the per-vertex 0..1 value each mesh defines (buoy: height, wood: log tint,
// sail: height down the sail, sand: height above water).

const MAT_WATER: u32 = 0u;
const MAT_BUOY: u32 = 1u;
const MAT_RAFT_WOOD: u32 = 2u;
const MAT_SAIL: u32 = 3u;
const MAT_SAND: u32 = 4u;
const MAT_PALM_TRUNK: u32 = 5u;
const MAT_PALM_FROND: u32 = 6u;

fn decode_material(g: f32) -> u32 {
    return u32(round(g * 8.0));
}

fn buoy_albedo(local_y: f32) -> vec3<f32> {
    let red = vec3<f32>(0.50, 0.035, 0.03);
    let white = vec3<f32>(0.78, 0.76, 0.72);
    let steel = vec3<f32>(0.07, 0.07, 0.075);
    let yellow = vec3<f32>(0.75, 0.55, 0.05);
    if (local_y < 0.4) {
        let band = select(red, white, local_y > 0.05 && local_y < 0.22);
        return band * select(1.0, 0.55, local_y < -0.05);
    }
    if (local_y < 0.47) {
        return steel;
    }
    if (local_y < 3.2) {
        return select(steel, red, local_y > 1.6 && local_y < 2.4);
    }
    return yellow;
}

fn material_albedo(id: u32, param: f32) -> vec3<f32> {
    switch (id) {
        case 1u: {
            return buoy_albedo(param * 4.0 - 0.5);
        }
        case 2u: {
            // Sun-bleached driftwood, each log a slightly different grey-brown.
            return mix(vec3<f32>(0.32, 0.27, 0.21), vec3<f32>(0.52, 0.47, 0.40), param);
        }
        case 3u: {
            return vec3<f32>(0.80, 0.74, 0.62);
        }
        case 4u: {
            // Dry sand, darker where wet near the waterline (param = height / 8 m).
            let wet = 1.0 - smoothstep(0.0, 0.05, param);
            return mix(vec3<f32>(0.78, 0.66, 0.48), vec3<f32>(0.45, 0.37, 0.27), wet);
        }
        case 5u: {
            return vec3<f32>(0.30, 0.22, 0.15);
        }
        case 6u: {
            return mix(vec3<f32>(0.10, 0.20, 0.05), vec3<f32>(0.22, 0.30, 0.08), param);
        }
        default: {
            return vec3<f32>(0.5);
        }
    }
}

// Beckmann slope variance (roughness) of each surface.
fn material_variance(id: u32) -> f32 {
    switch (id) {
        case 1u: { return 0.06; }
        case 2u: { return 0.35; }
        case 3u: { return 0.5; }
        case 4u: { return 0.6; }
        default: { return 0.4; }
    }
}

fn material_emission(id: u32, param: f32) -> vec3<f32> {
    if (id == 1u) {
        let y = param * 4.0 - 0.5;
        return select(vec3<f32>(0.0), vec3<f32>(1.0, 0.75, 0.3), y > 3.27 && y < 3.49);
    }
    return vec3<f32>(0.0);
}
