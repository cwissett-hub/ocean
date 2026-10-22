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

/// Glint contrast: sun glitter is made of separate sparkles off capillary facets about this
/// big; a pixel covering many of them sees their average.
pub const GLINT_FACET_M: f32 = 0.02;
/// Exponent of the sparkle distribution (higher = rarer, brighter glints).
pub const GLINT_POWER: f32 = 8.0;

/// Multiplier on the sun's specular for one sparkle cell with uniform random `v` (0..1), at a
/// pixel footprint of `footprint_m` (mirrors `glint` in shade.wgsl). Its mean over v is
/// exactly 1, so the glitter path keeps its brightness and width; its spread shrinks as a pixel
/// covers more facets.
pub fn glint(v: f32, footprint_m: f32) -> f32 {
    let amp = (GLINT_FACET_M / footprint_m.max(1e-6)).clamp(0.0, 1.0);
    let sparkle = (GLINT_POWER + 1.0) * v.clamp(0.0, 1.0).powf(GLINT_POWER);
    1.0 + amp * (sparkle - 1.0)
}

/// Radius of a glint dot, in cells.
pub const GLINT_RADIUS: f32 = 0.35;

/// A glint's brightness `d` cells from its centre, scaled so its mean over the cell is 1
/// (mirrors `glint_shape` in shade.wgsl).
pub fn glint_shape(d: f32) -> f32 {
    let x = (1.0 - (d / GLINT_RADIUS).powi(2)).max(0.0);
    // ∫ (1 − ρ²/r²)² over the disc is π r² / 3.
    x * x / (std::f32::consts::PI * GLINT_RADIUS * GLINT_RADIUS / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glint_dots_are_round_and_keep_the_mean() {
        // Integrate over a unit cell with the dot centred anywhere it can be (r..1 − r, so it
        // never overhangs its cell).
        let n = 400;
        for c in [
            glam::Vec2::splat(0.5),
            glam::Vec2::new(GLINT_RADIUS, 1.0 - GLINT_RADIUS),
        ] {
            let mut sum = 0.0;
            for j in 0..n {
                for i in 0..n {
                    let p = (glam::Vec2::new(i as f32, j as f32) + 0.5) / n as f32;
                    sum += glint_shape(p.distance(c));
                }
            }
            let mean = sum / (n * n) as f32;
            assert!((mean - 1.0).abs() < 0.01, "mean {mean} for a dot at {c}");
        }
        assert_eq!(glint_shape(GLINT_RADIUS), 0.0);
        assert!(glint_shape(0.0) > glint_shape(0.2 * GLINT_RADIUS));
    }

    fn mean_glint(footprint: f32) -> (f32, f32, f32) {
        let n = 100_000;
        let g: Vec<f32> = (0..n).map(|i| glint((i as f32 + 0.5) / n as f32, footprint)).collect();
        let mean = g.iter().sum::<f32>() / n as f32;
        let lo = g.iter().copied().fold(f32::MAX, f32::min);
        let hi = g.iter().copied().fold(f32::MIN, f32::max);
        (mean, lo, hi)
    }

    #[test]
    fn glints_conserve_the_glitter_and_fade_with_distance() {
        for fp in [0.002, 0.01, 0.05, 0.3, 5.0] {
            let (mean, lo, _) = mean_glint(fp);
            assert!((mean - 1.0).abs() < 1e-3, "mean {mean} at footprint {fp}");
            assert!(lo >= 0.0, "negative light at {fp}");
        }
        let (_, near_lo, near_hi) = mean_glint(0.005);
        assert!(near_hi > 6.0 && near_lo < 0.05, "near: sparkles {near_lo}..{near_hi}");
        let (_, far_lo, far_hi) = mean_glint(2.0);
        assert!(far_hi - far_lo < 0.2, "far: almost uniform {far_lo}..{far_hi}");
    }

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
        let shade = include_str!("../../shaders/shade.wgsl");
        assert!(shade.contains(&format!("GLINT_FACET_M: f32 = {:?};", GLINT_FACET_M)));
        assert!(shade.contains(&format!("GLINT_POWER: f32 = {:?};", GLINT_POWER)));
        assert!(shade.contains(&format!("GLINT_RADIUS: f32 = {:?};", GLINT_RADIUS)));
    }

    #[test]
    fn murkier_water_reflects_more_and_black_water_none() {
        let clear = water_column_reflectance(WATER_ABSORB, WATER_SCATTER);
        let murky = water_column_reflectance(WATER_ABSORB, WATER_SCATTER * 5.0);
        assert!(murky.cmpgt(clear).all());
        assert_eq!(water_column_reflectance(WATER_ABSORB, Vec3::ZERO), Vec3::ZERO);
    }
}
