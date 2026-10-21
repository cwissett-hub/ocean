//! GPU pass timing from timestamps written between passes, read back asynchronously
//! (a few frames late) so measuring never stalls the frame.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// Frames of history behind every statistic.
pub const HISTORY: usize = 240;
const MAX_MARKS: u32 = 32;
const SLOTS: usize = 3;

/// 95th percentile (nearest rank) of a set of samples; 0 when empty.
pub fn rolling_p95(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let rank = ((0.95 * s.len() as f32).ceil() as usize).clamp(1, s.len());
    s[rank - 1]
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PassStats {
    pub avg_ms: f32,
    pub p95_ms: f32,
    pub max_ms: f32,
}

#[derive(Debug, Clone, Default)]
pub struct PassHistory {
    samples: VecDeque<f32>,
}

impl PassHistory {
    pub fn push(&mut self, ms: f32) {
        if self.samples.len() == HISTORY {
            self.samples.pop_front();
        }
        self.samples.push_back(ms);
    }

    pub fn stats(&self) -> PassStats {
        if self.samples.is_empty() {
            return PassStats::default();
        }
        let v: Vec<f32> = self.samples.iter().copied().collect();
        PassStats {
            avg_ms: v.iter().sum::<f32>() / v.len() as f32,
            p95_ms: rolling_p95(&v),
            max_ms: v.iter().copied().fold(0.0, f32::max),
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

const IDLE: u8 = 0;
const MAPPING: u8 = 1;
const MAPPED: u8 = 2;

struct Slot {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    labels: Vec<&'static str>,
    state: Arc<AtomicU8>,
}

pub struct GpuTimer {
    enabled: bool,
    period_ns: f32,
    slots: Vec<Slot>,
    /// Slot being recorded this frame, if one was free.
    active: Option<usize>,
    next: usize,
    passes: Vec<(&'static str, PassHistory)>,
    total: PassHistory,
    /// Every frame since logging started (benchmarks): totals and per-pass samples.
    log: Option<FrameLog>,
}

#[derive(Debug, Clone, Default)]
pub struct FrameLog {
    pub totals: Vec<f32>,
    pub passes: Vec<(&'static str, Vec<f32>)>,
}

impl GpuTimer {
    /// Timing is silently off when the device lacks encoder timestamps.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuTimer {
        let enabled = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let bytes = (MAX_MARKS * 8) as u64;
        let slots = if enabled {
            (0..SLOTS)
                .map(|_| Slot {
                    queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("gpu timer"),
                        ty: wgpu::QueryType::Timestamp,
                        count: MAX_MARKS,
                    }),
                    resolve: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("gpu timer resolve"),
                        size: bytes,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                    readback: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("gpu timer readback"),
                        size: bytes,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    labels: Vec::new(),
                    state: Arc::new(AtomicU8::new(IDLE)),
                })
                .collect()
        } else {
            Vec::new()
        };
        GpuTimer {
            enabled,
            period_ns: queue.get_timestamp_period(),
            slots,
            active: None,
            next: 0,
            passes: Vec::new(),
            total: PassHistory::default(),
            log: None,
        }
    }

    /// Starts (or restarts) keeping every sample, not just the rolling window.
    pub fn start_log(&mut self) {
        self.log = Some(FrameLog::default());
    }

    pub fn log(&self) -> Option<&FrameLog> {
        self.log.as_ref()
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Starts timing a frame (call before the first pass).
    pub fn start(&mut self, enc: &mut wgpu::CommandEncoder) {
        self.active = None;
        if !self.enabled || self.slots[self.next].state.load(Ordering::Acquire) != IDLE {
            return;
        }
        let i = self.next;
        self.next = (self.next + 1) % SLOTS;
        self.slots[i].labels.clear();
        self.slots[i].labels.push("start");
        enc.write_timestamp(&self.slots[i].queries, 0);
        self.active = Some(i);
    }

    /// Ends the interval since the previous mark; it is reported as `label`.
    pub fn mark(&mut self, enc: &mut wgpu::CommandEncoder, label: &'static str) {
        let Some(i) = self.active else { return };
        let slot = &mut self.slots[i];
        let n = slot.labels.len() as u32;
        if n < MAX_MARKS {
            enc.write_timestamp(&slot.queries, n);
            slot.labels.push(label);
        }
    }

    /// Resolves this frame's timestamps (call after the last mark, before submit).
    pub fn finish(&mut self, enc: &mut wgpu::CommandEncoder) {
        let Some(i) = self.active else { return };
        let slot = &self.slots[i];
        let n = slot.labels.len() as u32;
        enc.resolve_query_set(&slot.queries, 0..n, &slot.resolve, 0);
        enc.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, (n * 8) as u64);
    }

    /// Starts the readback of this frame's timestamps (call right after submit).
    pub fn after_submit(&mut self) {
        let Some(i) = self.active.take() else { return };
        let slot = &self.slots[i];
        slot.state.store(MAPPING, Ordering::Release);
        let state = slot.state.clone();
        slot.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            state.store(if r.is_ok() { MAPPED } else { IDLE }, Ordering::Release);
        });
    }

    /// Collects any finished readbacks (non-blocking).
    pub fn poll(&mut self, device: &wgpu::Device) {
        if !self.enabled {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        for i in 0..self.slots.len() {
            if self.slots[i].state.load(Ordering::Acquire) != MAPPED {
                continue;
            }
            let ticks: Vec<u64> = {
                let slot = &self.slots[i];
                let n = slot.labels.len();
                let data = slot.readback.slice(..).get_mapped_range().expect("mapped timer range");
                bytemuck::cast_slice::<u8, u64>(&data)[..n].to_vec()
            };
            self.slots[i].readback.unmap();
            self.slots[i].state.store(IDLE, Ordering::Release);
            let labels = self.slots[i].labels.clone();
            let to_ms = |a: u64, b: u64| b.saturating_sub(a) as f32 * self.period_ns * 1e-6;
            for k in 1..ticks.len() {
                let ms = to_ms(ticks[k - 1], ticks[k]);
                if let Some(log) = &mut self.log {
                    match log.passes.iter_mut().find(|(l, _)| *l == labels[k]) {
                        Some((_, v)) => v.push(ms),
                        None => log.passes.push((labels[k], vec![ms])),
                    }
                }
                match self.passes.iter_mut().find(|(l, _)| *l == labels[k]) {
                    Some((_, h)) => h.push(ms),
                    None => {
                        let mut h = PassHistory::default();
                        h.push(ms);
                        self.passes.push((labels[k], h));
                    }
                }
            }
            if ticks.len() > 1 {
                let ms = to_ms(ticks[0], ticks[ticks.len() - 1]);
                self.total.push(ms);
                if let Some(log) = &mut self.log {
                    log.totals.push(ms);
                }
            }
        }
    }

    pub fn stats(&self) -> Vec<(&'static str, PassStats)> {
        self.passes.iter().map(|(l, h)| (*l, h.stats())).collect()
    }

    pub fn total(&self) -> PassStats {
        self.total.stats()
    }

    /// Frames of GPU timing collected so far (up to HISTORY).
    pub fn samples(&self) -> usize {
        self.total.len()
    }

    /// Forgets everything measured so far (e.g. after warm-up).
    pub fn reset(&mut self) {
        self.passes.clear();
        self.total = PassHistory::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p95_of_one_to_hundred() {
        let s: Vec<f32> = (1..=100).map(|v| v as f32).collect();
        assert!((rolling_p95(&s) - 95.0).abs() <= 0.5, "{}", rolling_p95(&s));
    }

    #[test]
    fn p95_of_empty_is_zero() {
        assert_eq!(rolling_p95(&[]), 0.0);
    }

    #[test]
    fn p95_ignores_order() {
        let mut s: Vec<f32> = (1..=100).map(|v| v as f32).collect();
        s.reverse();
        assert!((rolling_p95(&s) - 95.0).abs() <= 0.5);
    }

    #[test]
    fn stats_track_a_rolling_window() {
        let mut st = PassHistory::default();
        for _ in 0..1000 {
            st.push(10.0);
        }
        for _ in 0..HISTORY {
            st.push(2.0);
        }
        let s = st.stats();
        assert!((s.avg_ms - 2.0).abs() < 1e-5 && (s.p95_ms - 2.0).abs() < 1e-5 && (s.max_ms - 2.0).abs() < 1e-5);
    }
}
