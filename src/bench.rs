//! `--bench N`: render N frames uncapped after a warm-up and report GPU and CPU frame times.

use glam::DVec2;

use crate::gpu::timing::{FrameLog, rolling_p95};
use crate::scene::cameras::CameraMode;
use crate::scene::raft::RaftControls;
use crate::world::islands::start_island;

pub const WARMUP_FRAMES: u32 = 120;
/// The course starts this far from the start island and passes this far off its beach.
const COURSE_START_M: f64 = 150.0;
const COURSE_PASS_OFF_BEACH_M: f64 = 40.0;

/// Where the bench course starts (world) and its heading (degrees): 150 m from the start
/// island, on a straight track whose closest approach is 40 m off the beach, so the island,
/// its palms, the shallows and the surf all pass through the view.
pub fn course_start(seed: u64) -> (DVec2, f32) {
    let isle = start_island(seed);
    let heading_deg = 30.0f64;
    let h = heading_deg.to_radians();
    let dir = DVec2::new(h.sin(), h.cos());
    let side = DVec2::new(dir.y, -dir.x);
    let miss = isle.radius as f64 + COURSE_PASS_OFF_BEACH_M;
    let along = (COURSE_START_M * COURSE_START_M - miss * miss).sqrt();
    (isle.center - dir * along + side * miss, heading_deg as f32)
}

/// `--bench-camera` names.
pub fn bench_camera(name: &str) -> Option<CameraMode> {
    match name {
        "chase" => Some(CameraMode::Chase),
        "deck" => Some(CameraMode::Deck),
        "high" => Some(CameraMode::High),
        _ => None,
    }
}

/// The bench course's helm: sail up, a slow gentle weave (deterministic in `t`, seconds
/// since the bench started) so the view sweeps across the scene.
pub fn course_controls(t: f32) -> RaftControls {
    RaftControls {
        steer: 0.25 * (t * 0.21).sin(),
        paddle: 0.0,
        sail_up: true,
    }
}

fn stats(v: &[f32]) -> (f32, f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let avg = v.iter().sum::<f32>() / v.len() as f32;
    (avg, rolling_p95(v), v.iter().copied().fold(0.0, f32::max))
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// One-line JSON report.
pub fn report(adapter: &str, extent: (u32, u32), log: &FrameLog, cpu_frame_ms: &[f32]) -> String {
    let (g_avg, g_p95, g_max) = stats(&log.totals);
    let (c_avg, c_p95, c_max) = stats(cpu_frame_ms);
    let passes: Vec<String> = log
        .passes
        .iter()
        .map(|(name, v)| {
            let (a, p, _) = stats(v);
            format!("{}:{{\"avg_ms\":{a:.3},\"p95_ms\":{p:.3}}}", json_str(name))
        })
        .collect();
    format!(
        "{{\"adapter\":{},\"extent\":[{},{}],\"frames\":{},\"gpu_avg_ms\":{g_avg:.3},\"gpu_p95_ms\":{g_p95:.3},\"gpu_max_ms\":{g_max:.3},\"cpu_frame_avg_ms\":{c_avg:.3},\"cpu_frame_p95_ms\":{c_p95:.3},\"cpu_frame_max_ms\":{c_max:.3},\"passes\":{{{}}}}}",
        json_str(adapter),
        extent.0,
        extent.1,
        log.totals.len(),
        passes.join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_contains_the_numbers() {
        let log = FrameLog {
            totals: (1..=100).map(|v| v as f32 / 10.0).collect(),
            passes: vec![("shade", vec![1.0, 3.0])],
        };
        let r = report("GPU \"X\"", (1280, 720), &log, &[16.0, 17.0]);
        assert!(r.contains("\"adapter\":\"GPU \\\"X\\\"\""), "{r}");
        assert!(r.contains("\"extent\":[1280,720]"));
        assert!(r.contains("\"frames\":100"));
        assert!(r.contains("\"gpu_p95_ms\":9.500"), "{r}");
        assert!(r.contains("\"shade\":{\"avg_ms\":2.000"), "{r}");
        assert!(r.starts_with('{') && r.ends_with('}'));
    }

    #[test]
    fn empty_log_reports_zeros() {
        let r = report("a", (1, 1), &FrameLog::default(), &[]);
        assert!(r.contains("\"frames\":0") && r.contains("\"gpu_avg_ms\":0.000"));
    }

    #[test]
    fn bench_camera_names() {
        assert_eq!(bench_camera("high"), Some(CameraMode::High));
        assert_eq!(bench_camera("deck"), Some(CameraMode::Deck));
        assert_eq!(bench_camera("chase"), Some(CameraMode::Chase));
        assert_eq!(bench_camera("orbit"), None);
    }

    #[test]
    fn bench_course_controls_are_scripted() {
        for i in 0..6000 {
            let t = i as f32 * 0.1;
            let c = course_controls(t);
            assert_eq!(c, course_controls(t), "deterministic at {t}");
            assert!(c.sail_up, "sail up at {t}");
            assert!(c.steer.abs() <= 0.3, "steer {} at {t}", c.steer);
            assert_eq!(c.paddle, 0.0);
        }
        assert_ne!(
            course_controls(3.0).steer,
            course_controls(9.0).steer,
            "the course weaves"
        );
    }

    #[test]
    fn bench_course_passes_off_the_start_beach() {
        use crate::world::islands::{WORLD_SEED, start_island};
        let isle = start_island(WORLD_SEED);
        let (start, heading_deg) = course_start(WORLD_SEED);
        assert!(
            ((start - isle.center).length() - 150.0).abs() < 1.0,
            "starts 150 m away"
        );
        // The straight track's closest approach is 40 m off the beach (the dunes' edge).
        let h = (heading_deg as f64).to_radians();
        let dir = glam::DVec2::new(h.sin(), h.cos());
        let to = isle.center - start;
        assert!(to.dot(dir) > 100.0, "heads towards the island");
        let miss = (to - dir * to.dot(dir)).length();
        assert!(
            (miss - (isle.radius as f64 + 40.0)).abs() < 1.0,
            "passes {miss} m from the centre"
        );
    }
}
