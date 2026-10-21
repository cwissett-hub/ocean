// Buoy paint scheme by height above the waterline (metres, y = 0 at the waterline).

const BUOY_PAINT_VARIANCE: f32 = 0.06;

fn buoy_albedo(local_y: f32) -> vec3<f32> {
    let red = vec3<f32>(0.50, 0.035, 0.03);
    let white = vec3<f32>(0.78, 0.76, 0.72);
    let steel = vec3<f32>(0.07, 0.07, 0.075);
    let yellow = vec3<f32>(0.75, 0.55, 0.05);
    if (local_y < 0.4) {
        // Drum: red with one white band, fouled and darker below the waterline.
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

// The lantern at the top glows at dusk.
fn buoy_emission(local_y: f32) -> vec3<f32> {
    return select(vec3<f32>(0.0), vec3<f32>(1.0, 0.75, 0.3), local_y > 3.27 && local_y < 3.49);
}
