//! How much light the water column sends back up (mirrors `water_column_reflectance` in
//! surfaces.wgsl). Gordon et al. (1988): just below the surface, upwelling radiance over
//! downwelling irradiance is rrs = (0.0949 + 0.0794·u)·u with u = bb / (a + bb); crossing the
//! surface divides radiance by n².

use glam::Vec3;

/// Seawater absorption and scattering, 1/m (surfaces.wgsl WATER_ABSORB / WATER_SCATTER).
pub const WATER_ABSORB: Vec3 = Vec3::new(0.45, 0.065, 0.025);
pub const WATER_SCATTER: Vec3 = Vec3::new(0.012, 0.018, 0.026);
/// Half of pure water's scattering is backwards (its phase function is symmetric).
const BACKSCATTER_FRACTION: f32 = 0.5;
/// n² for seawater: radiance leaving the water is divided by this.
pub const WATER_N2: f32 = 1.78;

/// Upwelling radiance just above the surface per unit downwelling irradiance just below it,
/// for an infinitely deep column, 1/sr (before the view-angle Fresnel factor).
pub fn water_column_reflectance(absorb: Vec3, scatter: Vec3) -> Vec3 {
    let bb = scatter * BACKSCATTER_FRACTION;
    let u = bb / (absorb + bb).max(Vec3::splat(1e-6));
    (Vec3::splat(0.0949) + 0.0794 * u) * u / WATER_N2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_sea_is_blue_and_dim() {
        let r = water_column_reflectance(WATER_ABSORB, WATER_SCATTER);
        assert!(r.z > r.y && r.y > r.x, "blue > green > red: {r}");
        // Clear open-ocean remote-sensing reflectance: ~0.01–0.03 /sr in blue, a few 1e-3
        // in green, under 1e-3 in red.
        assert!((0.01..0.03).contains(&r.z), "blue {}", r.z);
        assert!((0.002..0.01).contains(&r.y), "green {}", r.y);
        assert!(r.x < 1e-3, "red {}", r.x);
    }

    #[test]
    fn shader_uses_the_same_water() {
        let src = include_str!("../../shaders/surfaces.wgsl");
        let v = |c: Vec3| format!("vec3<f32>({:?}, {:?}, {:?})", c.x, c.y, c.z);
        assert!(src.contains(&format!("WATER_ABSORB: vec3<f32> = {};", v(WATER_ABSORB))));
        assert!(src.contains(&format!("WATER_SCATTER: vec3<f32> = {};", v(WATER_SCATTER))));
        assert!(src.contains(&format!("WATER_N2: f32 = {:?};", WATER_N2)));
    }

    #[test]
    fn murkier_water_reflects_more_and_black_water_none() {
        let clear = water_column_reflectance(WATER_ABSORB, WATER_SCATTER);
        let murky = water_column_reflectance(WATER_ABSORB, WATER_SCATTER * 5.0);
        assert!(murky.cmpgt(clear).all());
        assert_eq!(water_column_reflectance(WATER_ABSORB, Vec3::ZERO), Vec3::ZERO);
    }
}
