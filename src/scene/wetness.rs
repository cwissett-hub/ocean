//! How wet the raft's logs are: for each of the raft's 9 water-sample columns, the height (in
//! the raft's frame) below which the wood is wet. Water reaching higher soaks it at once;
//! once the water falls away the wet line sinks slowly as the wood dries, so a wave that
//! washes over the deck leaves it dark for a while. `shade` darkens wood below the line.

use crate::scene::raft::{RAFT_SAMPLE_OFFSETS, RaftBody};
use crate::waves::query::WaterSample;

/// How fast the wet line sinks once the water has gone, m/s (0.2 m takes ~10 s).
pub const DRY_RATE_M_PER_S: f32 = 0.02;
/// Water climbs a little up dry wood (capillarity, small splashes).
pub const WICK_M: f32 = 0.1;
/// Lowest wet line kept (well below the logs).
const FLOOR_M: f32 = -0.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaftWetness {
    /// Wet line per sample column (RAFT_SAMPLE_OFFSETS order), raft-frame y, m.
    pub line: [f32; 9],
}

impl Default for RaftWetness {
    fn default() -> Self {
        RaftWetness { line: [FLOOR_M; 9] }
    }
}

impl RaftWetness {
    /// Water heights at the sample columns relative to the hull there (raft frame).
    pub fn water_levels(body: &RaftBody, water: &[WaterSample; 9]) -> [f32; 9] {
        std::array::from_fn(|i| {
            let r = body.rotation * RAFT_SAMPLE_OFFSETS[i];
            water[i].height - (body.position.y + r.y)
        })
    }

    pub fn update(&mut self, levels: &[f32; 9], dt: f32) {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        for (line, &level) in self.line.iter_mut().zip(levels) {
            let dried = *line - DRY_RATE_M_PER_S * dt;
            let reach = if level.is_finite() { level + WICK_M } else { FLOOR_M };
            *line = dried.max(reach).max(FLOOR_M);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soaks_at_once_and_dries_slowly() {
        let mut w = RaftWetness::default();
        w.update(&[0.0; 9], 1.0 / 60.0);
        assert!(
            w.line.iter().all(|&l| (l - WICK_M).abs() < 1e-6),
            "wet to the waterline: {:?}",
            w.line
        );
        // A wave washes 0.25 m over the deck: soaked at once.
        w.update(&[0.25; 9], 1.0 / 60.0);
        assert!(w.line.iter().all(|&l| l >= 0.25), "{:?}", w.line);
        // The water drops back: after 5 s the line has only sunk by DRY_RATE·5.
        for _ in 0..300 {
            w.update(&[0.0; 9], 1.0 / 60.0);
        }
        let want = 0.25 + WICK_M - DRY_RATE_M_PER_S * 5.0;
        assert!(
            w.line.iter().all(|&l| (l - want).abs() < 0.01),
            "{:?} vs {want}",
            w.line
        );
        // ... and never below where the water is now.
        for _ in 0..6000 {
            w.update(&[0.0; 9], 1.0 / 60.0);
        }
        assert!(w.line.iter().all(|&l| (l - WICK_M).abs() < 1e-4), "{:?}", w.line);
    }

    #[test]
    fn columns_are_independent_and_ignore_bad_samples() {
        let mut w = RaftWetness::default();
        let mut levels = [0.0; 9];
        levels[7] = 0.3; // the bow dips into a wave
        levels[0] = f32::NAN;
        w.update(&levels, 1.0 / 60.0);
        assert!(w.line[7] > 0.3 && w.line[4] <= WICK_M + 1e-6);
        assert!(w.line[0].is_finite());
    }
}
