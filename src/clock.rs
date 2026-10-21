//! Wall-clock to simulation time. Elapsed time is kept in f64 and reduced modulo the wave
//! repeat period *before* it becomes f32, so frames stay evenly spaced after days of running.

use crate::waves::spectrum::REPEAT_PERIOD_S;

/// Simulation time in [0, REPEAT_PERIOD_S) for an elapsed wall-clock time in seconds.
pub fn sim_time(elapsed_s: f64) -> f32 {
    (elapsed_s.rem_euclid(REPEAT_PERIOD_S as f64) as f32).min(REPEAT_PERIOD_S.next_down())
}

/// Dragging a sea slider changes the spectrum every frame; rebuilding it costs milliseconds,
/// so rebuild at most this often.
pub const SPECTRUM_REBUILD_INTERVAL_S: f32 = 0.15;

pub fn should_rebuild_spectrum(changed: bool, since_last_rebuild_s: f32) -> bool {
    changed && since_last_rebuild_s >= SPECTRUM_REBUILD_INTERVAL_S
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_stay_smooth_after_days_of_running() {
        let frame = 1.0 / 60.0;
        for start in [12.0 * 3600.0, 3.0 * 86_400.0, 400.0 * 86_400.0] {
            let (a, b) = (sim_time(start), sim_time(start + frame));
            let step = if b >= a {
                b - a
            } else {
                b + crate::waves::spectrum::REPEAT_PERIOD_S - a
            };
            assert!((step as f64 - frame).abs() < 1e-4, "at {start} s step {step}");
        }
    }

    #[test]
    fn stays_inside_one_period() {
        for t in [0.0, 199.9, 200.0, 1e7] {
            let s = sim_time(t);
            assert!(
                (0.0..crate::waves::spectrum::REPEAT_PERIOD_S).contains(&s),
                "{t} -> {s}"
            );
        }
    }

    #[test]
    fn spectrum_rebuild_is_debounced() {
        assert!(!should_rebuild_spectrum(false, 10.0));
        assert!(!should_rebuild_spectrum(true, 0.05));
        assert!(should_rebuild_spectrum(true, SPECTRUM_REBUILD_INTERVAL_S));
    }
}
