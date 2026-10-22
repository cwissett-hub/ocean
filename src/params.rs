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
    /// The controls hint in the bottom-left corner (H).
    pub show_controls: bool,
    /// Per-pass GPU timing table under the fps readout (for tuning).
    pub show_timings: bool,
    pub vsync: bool,
    /// Temporal anti-aliasing.
    pub taa: bool,
    /// Contrast-adaptive sharpening after TAA, 0 (off) ..1.
    pub sharpness: f32,
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
            show_controls: true,
            show_timings: false,
            vsync: true,
            taa: true,
            sharpness: 0.5,
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
pub const SHARPNESS: (f32, f32) = (0.0, 1.0);

fn clamp(v: f32, range: (f32, f32), default: f32) -> f32 {
    if v.is_finite() {
        v.clamp(range.0, range.1)
    } else {
        default
    }
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
            sharpness: clamp(self.sharpness, SHARPNESS, d.sharpness),
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
        let p = OceanParams {
            wind_speed: f32::NAN,
            ..Default::default()
        }
        .clamped();
        assert_eq!(p.wind_speed, 5.0);
    }
}

#[cfg(test)]
mod override_tests {
    use super::*;

    #[test]
    fn sets_numeric_and_bool_fields() {
        let p = apply_overrides(
            OceanParams::default(),
            &["wind_speed=15", "vsync=false", "sun_elevation_deg=-2.5"],
        )
        .unwrap();
        assert_eq!(p.wind_speed, 15.0);
        assert!(!p.vsync);
        assert_eq!(p.sun_elevation_deg, -2.5);
    }

    #[test]
    fn sets_reflection_mode() {
        let p = apply_overrides(OceanParams::default(), &["reflection_mode=screenspace"]).unwrap();
        assert_eq!(p.reflection_mode, ReflectionMode::ScreenSpace);
    }

    #[test]
    fn rejects_unknown_keys_and_bad_values() {
        assert!(apply_overrides(OceanParams::default(), &["windspeed=3"]).is_err());
        assert!(apply_overrides(OceanParams::default(), &["wind_speed=fast"]).is_err());
        assert!(apply_overrides(OceanParams::default(), &["wind_speed"]).is_err());
    }

    #[test]
    fn result_is_clamped() {
        let p = apply_overrides(OceanParams::default(), &["wind_speed=400"]).unwrap();
        assert_eq!(p.wind_speed, WIND_SPEED.1);
    }
}

/// Applies `key=value` overrides (field names of `OceanParams`), e.g. from `--set` on the
/// command line, and clamps the result.
pub fn apply_overrides<S: AsRef<str>>(mut p: OceanParams, overrides: &[S]) -> Result<OceanParams, String> {
    for o in overrides {
        let o = o.as_ref();
        let (key, value) = o
            .split_once('=')
            .ok_or_else(|| format!("expected key=value, got {o:?}"))?;
        let num = || {
            value
                .parse::<f32>()
                .map_err(|_| format!("{key}: {value:?} is not a number"))
        };
        let flag = || {
            value
                .parse::<bool>()
                .map_err(|_| format!("{key}: {value:?} is not true/false"))
        };
        match key {
            "wind_speed" => p.wind_speed = num()?,
            "wind_dir_deg" => p.wind_dir_deg = num()?,
            "fetch_km" => p.fetch_km = num()?,
            "swell_amount" => p.swell_amount = num()?,
            "choppiness" => p.choppiness = num()?,
            "foam_amount" => p.foam_amount = num()?,
            "sun_elevation_deg" => p.sun_elevation_deg = num()?,
            "sun_azimuth_deg" => p.sun_azimuth_deg = num()?,
            "particle_amount" => p.particle_amount = num()?,
            "render_scale" => p.render_scale = num()?,
            "show_overlay" => p.show_overlay = flag()?,
            "show_controls" => p.show_controls = flag()?,
            "show_timings" => p.show_timings = flag()?,
            "vsync" => p.vsync = flag()?,
            "taa" => p.taa = flag()?,
            "sharpness" => p.sharpness = num()?,
            "reflection_mode" => {
                p.reflection_mode = match value.to_ascii_lowercase().as_str() {
                    "rt" => ReflectionMode::Rt,
                    "screenspace" | "ssr" => ReflectionMode::ScreenSpace,
                    _ => return Err(format!("reflection_mode: {value:?} is not rt/screenspace")),
                }
            }
            _ => return Err(format!("unknown setting {key:?}")),
        }
    }
    Ok(p.clamped())
}

#[cfg(test)]
mod overlay_tests {
    use super::*;

    #[test]
    fn detailed_timings_are_off_by_default_and_settable() {
        assert!(!OceanParams::default().show_timings);
        assert!(OceanParams::default().show_overlay, "the fps readout stays on");
        assert!(
            apply_overrides(OceanParams::default(), &["show_timings=true"])
                .unwrap()
                .show_timings
        );
    }
}
