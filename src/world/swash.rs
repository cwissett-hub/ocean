//! The swash: water running up the beach and back. Its height above the still waterline is
//! amp·(½ + ½·sin θ) with θ = ω·t + φ (φ varies along the shore). Because it is analytic, how
//! long ago the water last reached a spot of beach is known exactly, so the sand it leaves is
//! wet and dries over a few seconds. Mirrors `swash_*` in frame_common.wgsl.

use std::f32::consts::{PI, TAU};

/// Swash rhythm, rad/s: 29 run-ups per 200 s (~7 s each), so it repeats with the sim clock
/// (time wraps every REPEAT_PERIOD_S) without a jump.
pub const SWASH_OMEGA: f32 = std::f32::consts::TAU * 29.0 / 200.0;
/// Wet sand dries with this time constant, s.
pub const DRY_TAU_S: f32 = 4.0;
/// Above the swash's reach the sand is only damp, up to this far above it, m.
const DAMP_BAND_M: f32 = 0.25;
const DAMP: f32 = 0.35;

/// How high the swash runs above the still waterline for significant wave height `hs`, m.
pub fn swash_amplitude(hs: f32) -> f32 {
    0.08 + 0.06 * hs.clamp(0.0, 2.0)
}

/// Wetness (0 dry .. 1 soaked) of beach at height `ground` (above still water) when the swash
/// phase is `theta`, for swash amplitude `amp`.
pub fn sand_wetness(ground: f32, theta: f32, amp: f32) -> f32 {
    if ground <= 0.0 {
        return 1.0;
    }
    if ground >= amp {
        let t = ((ground - (amp + DAMP_BAND_M)) / -DAMP_BAND_M).clamp(0.0, 1.0);
        return DAMP * t * t * (3.0 - 2.0 * t);
    }
    // Covered while amp·(½ + ½ sin θ) ≥ ground, i.e. θ ∈ [a, π − a] with sin a = 2g/amp − 1.
    let a = (2.0 * ground / amp - 1.0).clamp(-1.0, 1.0).asin();
    let th = theta.rem_euclid(TAU);
    let start = a.rem_euclid(TAU);
    let end = PI - a;
    let covered = if start <= end {
        th >= start && th <= end
    } else {
        th >= start || th <= end
    };
    if covered {
        return 1.0;
    }
    let since = (th - end).rem_euclid(TAU) / SWASH_OMEGA;
    (-since / DRY_TAU_S).exp().max(DAMP)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_runs_the_same_swash() {
        let src = include_str!("../../shaders/frame_common.wgsl");
        for needle in [
            format!("SWASH_OMEGA: f32 = {SWASH_OMEGA:?};"),
            format!("SWASH_DRY_TAU: f32 = {DRY_TAU_S:?};"),
            format!("SWASH_DAMP: f32 = {DAMP:?};"),
            format!("SWASH_DAMP_BAND: f32 = {DAMP_BAND_M:?};"),
            "0.08 + 0.06 * clamp(frame.misc.y, 0.0, 2.0)".to_string(),
        ] {
            assert!(src.contains(&needle), "frame_common.wgsl lacks {needle:?}");
        }
        assert!((SWASH_OMEGA * crate::waves::spectrum::REPEAT_PERIOD_S / std::f32::consts::TAU - 29.0).abs() < 1e-4);
    }

    #[test]
    fn sand_the_swash_just_left_is_wet_and_dries() {
        let amp = 0.2;
        // Below the still waterline: always wet.
        assert_eq!(sand_wetness(-0.1, 1.0, amp), 1.0);
        // Halfway up the swash: covered while sin θ ≥ 0, i.e. θ ∈ [0, π].
        assert_eq!(sand_wetness(0.1, 0.5 * PI, amp), 1.0);
        // Just after the water left (θ = π + small): still almost soaked; later drier.
        let just = sand_wetness(0.1, PI + 0.05, amp);
        let later = sand_wetness(0.1, PI + 3.0, amp);
        assert!(just > 0.95, "{just}");
        assert!(later < just && later > 0.0, "{later}");
        let expect = (-(3.0 / SWASH_OMEGA) / DRY_TAU_S).exp();
        assert!((later - expect).abs() < 1e-3, "{later} vs {expect}");
        // Higher up it is wet for less of the cycle.
        let high = sand_wetness(0.19, 0.5 * PI + 0.6, amp);
        assert!(high < 1.0);
        // Above the reach: only damp, fading to dry.
        let damp = sand_wetness(amp + 0.05, 1.0, amp);
        assert!(damp > 0.0 && damp < 0.4, "{damp}");
        assert_eq!(sand_wetness(amp + 0.5, 1.0, amp), 0.0);
        // Periodic in θ.
        assert!((sand_wetness(0.1, PI + 1.0, amp) - sand_wetness(0.1, PI + 1.0 + TAU, amp)).abs() < 1e-5);
    }
}
