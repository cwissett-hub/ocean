//! Local (render) coordinates vs world coordinates. Everything the GPU sees is local f32,
//! relative to an origin that moves in whole wave/foam periods so the sea is unchanged when
//! it does, and no f32 ever grows large enough to jitter.

use glam::{DVec2, Vec2};

/// Rebase once the focus (the raft) is this far from the origin.
pub const REBASE_DISTANCE_M: f64 = 4096.0;
/// Origin moves in multiples of this: the largest wave cascade and the foam texture period.
pub const REBASE_STEP_M: f64 = 1024.0;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WorldOrigin {
    pub offset: DVec2,
}

impl WorldOrigin {
    pub fn to_world(&self, local: Vec2) -> DVec2 {
        self.offset + local.as_dvec2()
    }

    pub fn to_local(&self, world: DVec2) -> Vec2 {
        (world - self.offset).as_vec2()
    }

    /// The local shift to apply (subtract from every local position) when `focus` has drifted
    /// too far; a whole number of REBASE_STEP_M per axis.
    pub fn rebase_for(&self, focus: Vec2) -> Option<Vec2> {
        if (focus.length() as f64) <= REBASE_DISTANCE_M {
            return None;
        }
        let step = REBASE_STEP_M as f32;
        Some((focus / step).round() * step)
    }

    pub fn apply(&mut self, shift: Vec2) {
        self.offset += shift.as_dvec2();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec2, Vec2};

    #[test]
    fn local_and_world_round_trip() {
        let o = WorldOrigin {
            offset: DVec2::new(8192.0, -3072.0),
        };
        let w = DVec2::new(8200.5, -3000.25);
        assert_eq!(o.to_world(o.to_local(w)), w);
    }

    #[test]
    fn rebase_only_beyond_threshold_and_in_whole_periods() {
        let o = WorldOrigin::default();
        assert!(o.rebase_for(Vec2::new(4000.0, 0.0)).is_none());
        let f = Vec2::new(5000.0, -6000.0);
        let s = o.rebase_for(f).expect("rebase");
        for v in [s.x, s.y] {
            assert!((v / REBASE_STEP_M as f32).fract().abs() < 1e-6, "{s}");
        }
        assert!((f - s).abs().max_element() <= REBASE_STEP_M as f32, "{}", f - s);
    }

    #[test]
    fn apply_moves_the_offset_by_the_shift() {
        let mut o = WorldOrigin::default();
        o.apply(Vec2::new(5120.0, -6144.0));
        assert_eq!(o.offset, DVec2::new(5120.0, -6144.0));
    }

    #[test]
    fn rebase_step_is_a_whole_number_of_every_wave_and_foam_period() {
        // Shifting by a whole number of every texture period leaves the sea unchanged.
        for l in crate::waves::spectrum::CASCADE_SIZES_M {
            let k = REBASE_STEP_M / l as f64;
            assert!((k - k.round()).abs() < 1e-9, "cascade {l}");
        }
        assert_eq!(REBASE_STEP_M, crate::waves::spectrum::CASCADE_SIZES_M[0] as f64);
    }
}
