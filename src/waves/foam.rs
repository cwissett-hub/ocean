//! Whitecaps: foam is injected wherever the combined horizontal displacement folds the
//! surface (Jacobian below a threshold) and decays exponentially. `foam.wgsl` mirrors
//! `foam_step` and `combined_jacobian` exactly.

use crate::params::OceanParams;

/// Cascades whose folding makes whitecaps; the finest (sub-metre) one only roughens.
pub const FOAM_CASCADES: usize = 3;
pub const FOAM_MAX: f32 = 4.0;
/// Jacobian threshold at foam_amount = 1, calibrated so the default 5 m/s sea breaks on
/// ~0.01 % of its surface and 15 m/s on ~0.5 % (persistence then spreads that into
/// Monahan-like coverage).
const BASE_THRESHOLD: f32 = 0.72;
const THRESHOLD_PER_AMOUNT: f32 = 0.08;
/// Foam per second per unit of Jacobian below threshold.
const INJECT: f32 = 30.0;
const HALF_LIFE_S: f32 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoamParams {
    pub threshold: f32,
    pub inject: f32,
    pub decay_per_s: f32,
}

pub fn foam_params(p: &OceanParams) -> FoamParams {
    FoamParams {
        threshold: BASE_THRESHOLD + THRESHOLD_PER_AMOUNT * (p.foam_amount - 1.0),
        inject: INJECT,
        decay_per_s: std::f32::consts::LN_2 / HALF_LIFE_S,
    }
}

/// Jacobian determinant from the summed gradient (∂Dx/∂x, ∂Dz/∂z, ∂Dx/∂z).
pub fn combined_jacobian(d: [f32; 3]) -> f32 {
    (1.0 + d[0]) * (1.0 + d[1]) - d[2] * d[2]
}

pub fn foam_step(prev: f32, jacobian: f32, f: FoamParams, dt: f32) -> f32 {
    let decayed = (prev * (-f.decay_per_s * dt).exp()).max(0.0);
    (decayed + f.inject * (f.threshold - jacobian).clamp(0.0, 1.0) * dt).clamp(0.0, FOAM_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::OceanParams;
    use crate::waves::cpu_ocean::CpuOcean;
    use crate::waves::spectrum::{CASCADE_SIZES_M, N, build_spectrum};

    /// Combined Jacobian of the foam cascades on a 1 m grid over a 1024 m square, sampled
    /// nearest-texel (statistics only).
    fn combined_jacobians(p: &OceanParams) -> Vec<f32> {
        let o = CpuOcean::new(&build_spectrum(p, 42), p.choppiness);
        let f: Vec<_> = (0..FOAM_CASCADES).map(|c| o.evaluate(c, 0.0)).collect();
        let mut out = Vec::with_capacity(1024 * 1024);
        for z in 0..1024usize {
            for x in 0..1024usize {
                let mut d = [0.0f32; 3];
                for (c, fc) in f.iter().enumerate() {
                    let texel = CASCADE_SIZES_M[c] / N as f32;
                    let (cx, cz) = ((x as f32 / texel) as usize % N, (z as f32 / texel) as usize % N);
                    let g = fc.derivs[cz * N + cx];
                    (0..3).for_each(|i| d[i] += g[i]);
                }
                out.push(combined_jacobian(d));
            }
        }
        out
    }

    fn breaking_fraction(p: &OceanParams) -> f32 {
        let t = foam_params(p).threshold;
        let js = combined_jacobians(p);
        js.iter().filter(|&&j| j < t).count() as f32 / js.len() as f32
    }

    #[test]
    fn no_injection_above_threshold() {
        let f = foam_params(&OceanParams::default());
        assert_eq!(foam_step(0.0, f.threshold, f, 0.1), 0.0);
        assert_eq!(foam_step(0.0, 1.2, f, 0.1), 0.0);
        assert!(foam_step(0.0, f.threshold - 0.1, f, 0.1) > 0.0);
    }

    #[test]
    fn decays_with_its_half_life() {
        let f = foam_params(&OceanParams::default());
        let half_life = std::f32::consts::LN_2 / f.decay_per_s;
        let v = foam_step(1.0, 1.0, f, half_life);
        assert!((v - 0.5).abs() < 1e-4, "{v}");
    }

    #[test]
    fn stays_within_bounds() {
        let f = foam_params(&OceanParams {
            foam_amount: 2.0,
            ..Default::default()
        });
        for (prev, j, dt) in [(0.0, -5.0, 10.0), (3.9, -1.0, 1.0), (-1.0, 1.0, 0.1), (100.0, 1.0, 0.0)] {
            let v = foam_step(prev, j, f, dt);
            assert!((0.0..=FOAM_MAX).contains(&v), "prev {prev} j {j} dt {dt} -> {v}");
        }
    }

    #[test]
    fn foam_amount_raises_threshold() {
        let lo = foam_params(&OceanParams {
            foam_amount: 0.0,
            ..Default::default()
        })
        .threshold;
        let hi = foam_params(&OceanParams {
            foam_amount: 2.0,
            ..Default::default()
        })
        .threshold;
        assert!(lo < hi);
    }

    #[test]
    fn gentle_default_sea_breaks_rarely_but_does_break() {
        let frac = breaking_fraction(&OceanParams::default());
        assert!(frac > 0.0 && frac < 0.02, "{frac}");
    }

    #[test]
    fn strong_wind_breaks_far_more() {
        let calm = breaking_fraction(&OceanParams::default());
        let rough = breaking_fraction(&OceanParams {
            wind_speed: 15.0,
            ..Default::default()
        });
        assert!(rough > 0.002 && rough > 50.0 * calm, "calm {calm} rough {rough}");
    }

    #[test]
    fn combined_jacobian_of_identity_is_one() {
        assert_eq!(combined_jacobian([0.0; 3]), 1.0);
        assert!((combined_jacobian([-0.5, -0.5, 0.0]) - 0.25).abs() < 1e-6);
    }
}
