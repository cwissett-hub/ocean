//! `--bench N`: render N frames uncapped after a warm-up and report GPU and CPU frame times.

use crate::gpu::timing::{FrameLog, rolling_p95};

pub const WARMUP_FRAMES: u32 = 120;

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
}
