/// Which technique produces water reflections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionMode {
    Rt,
    ScreenSpace,
}

/// Every user-tunable setting. The settings panel mutates this and nothing else.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OceanParams {
    pub wind_speed: f32,
    pub wind_dir_deg: f32,
    pub fetch_km: f32,
    pub swell_amount: f32,
    pub choppiness: f32,
    pub foam_amount: f32,
    pub sun_elevation_deg: f32,
    pub sun_azimuth_deg: f32,
    pub reflection_mode: ReflectionMode,
    pub particle_amount: f32,
    pub render_scale: f32,
    pub show_overlay: bool,
    pub vsync: bool,
}

impl Default for OceanParams {
    fn default() -> Self {
        Self {
            wind_speed: 5.0,
            wind_dir_deg: 30.0,
            fetch_km: 100.0,
            swell_amount: 0.6,
            choppiness: 0.9,
            foam_amount: 1.0,
            sun_elevation_deg: 4.0,
            sun_azimuth_deg: 30.0,
            reflection_mode: ReflectionMode::Rt,
            particle_amount: 1.0,
            render_scale: 1.0,
            show_overlay: true,
            vsync: true,
        }
    }
}

pub const WIND_SPEED: (f32, f32) = (0.5, 25.0);
pub const WIND_DIR_DEG: (f32, f32) = (0.0, 360.0);
pub const FETCH_KM: (f32, f32) = (10.0, 500.0);
pub const SWELL_AMOUNT: (f32, f32) = (0.0, 1.0);
pub const CHOPPINESS: (f32, f32) = (0.0, 1.5);
pub const FOAM_AMOUNT: (f32, f32) = (0.0, 2.0);
pub const SUN_ELEVATION_DEG: (f32, f32) = (-6.0, 90.0);
pub const SUN_AZIMUTH_DEG: (f32, f32) = (0.0, 360.0);
pub const PARTICLE_AMOUNT: (f32, f32) = (0.0, 2.0);
pub const RENDER_SCALE: (f32, f32) = (0.5, 1.5);

fn clamp(v: f32, range: (f32, f32), default: f32) -> f32 {
    if v.is_finite() { v.clamp(range.0, range.1) } else { default }
}

impl OceanParams {
    /// Returns a copy with every numeric field inside its range; non-finite values reset to default.
    pub fn clamped(self) -> Self {
        let d = Self::default();
        Self {
            wind_speed: clamp(self.wind_speed, WIND_SPEED, d.wind_speed),
            wind_dir_deg: clamp(self.wind_dir_deg, WIND_DIR_DEG, d.wind_dir_deg),
            fetch_km: clamp(self.fetch_km, FETCH_KM, d.fetch_km),
            swell_amount: clamp(self.swell_amount, SWELL_AMOUNT, d.swell_amount),
            choppiness: clamp(self.choppiness, CHOPPINESS, d.choppiness),
            foam_amount: clamp(self.foam_amount, FOAM_AMOUNT, d.foam_amount),
            sun_elevation_deg: clamp(self.sun_elevation_deg, SUN_ELEVATION_DEG, d.sun_elevation_deg),
            sun_azimuth_deg: clamp(self.sun_azimuth_deg, SUN_AZIMUTH_DEG, d.sun_azimuth_deg),
            particle_amount: clamp(self.particle_amount, PARTICLE_AMOUNT, d.particle_amount),
            render_scale: clamp(self.render_scale, RENDER_SCALE, d.render_scale),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let p = OceanParams::default();
        assert_eq!(p.wind_speed, 5.0);
        assert_eq!(p.sun_elevation_deg, 4.0);
        assert_eq!(p.reflection_mode, ReflectionMode::Rt);
        assert_eq!(p.render_scale, 1.0);
    }

    #[test]
    fn clamp_enforces_ranges() {
        let p = OceanParams {
            sun_elevation_deg: -30.0,
            wind_speed: 99.0,
            render_scale: 0.0,
            ..Default::default()
        }
        .clamped();
        assert_eq!(p.sun_elevation_deg, -6.0);
        assert_eq!(p.wind_speed, 25.0);
        assert_eq!(p.render_scale, 0.5);
    }

    #[test]
    fn clamp_replaces_nan_with_default() {
        let p = OceanParams { wind_speed: f32::NAN, ..Default::default() }.clamped();
        assert_eq!(p.wind_speed, 5.0);
    }
}
