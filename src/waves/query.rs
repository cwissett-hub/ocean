//! Water height, normal and velocity at arbitrary world points, for anything that floats.
//! The GPU evaluates the same textures the renderer draws (`wave_query.wgsl` mirrors
//! `invert_horizontal`); results come back a few frames later without stalling.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use glam::{Vec2, Vec3};

use crate::gpu::adapter::GpuContext;
use crate::gpu::shaders::ShaderLibrary;
use crate::waves::gpu_ocean::GpuOcean;
use crate::waves::spectrum::CASCADE_SIZES_M;

/// Cascades the query sums (the sub-metre one only makes ripples).
pub const QUERY_CASCADES: usize = 3;
pub const MAX_QUERY_POINTS: usize = 64;
const SLOTS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterSample {
    pub height: f32,
    pub normal: Vec3,
    /// Velocity of the surface at that point (from successive samples), m/s.
    pub velocity: Vec3,
}

pub const INVERT_ITERATIONS: usize = 4;

/// Choppy waves move each surface point sideways, so the water at world point `p` came from
/// some other undisplaced point x with x + D(x).xz = p. Fixed-point iteration finds it.
pub fn invert_horizontal(p: Vec2, disp_at: impl Fn(Vec2) -> Vec3) -> Vec2 {
    let mut x = p;
    for _ in 0..INVERT_ITERATIONS {
        let d = disp_at(x);
        x = p - Vec2::new(d.x, d.z);
    }
    x
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QueryParams {
    count: u32,
    _pad: [u32; 3],
    sizes: [f32; 4],
}

const IDLE: u8 = 0;
const MAPPING: u8 = 1;
const MAPPED: u8 = 2;

struct Slot {
    staging: wgpu::Buffer,
    count: usize,
    state: Arc<AtomicU8>,
}

/// Asynchronous GPU water queries: `request` points, `record` into the frame, call
/// `after_submit`, and `poll` later frames for (height, normal) per point.
pub struct WaveQuery {
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    points: wgpu::Buffer,
    results: wgpu::Buffer,
    params: wgpu::Buffer,
    slots: Vec<Slot>,
    requested: Vec<Vec2>,
    recorded: Option<usize>,
    next: usize,
}

const RESULT_BYTES: u64 = (MAX_QUERY_POINTS * 16) as u64;

impl WaveQuery {
    pub fn new(ctx: &GpuContext, lib: &ShaderLibrary, ocean: &GpuOcean) -> WaveQuery {
        let d = &ctx.device;
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wave_query"),
            source: wgpu::ShaderSource::Wgsl(lib.source("wave_query").expect("shader source").into()),
        });
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wave_query"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label, size, usage| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        let points = buffer("query points", (MAX_QUERY_POINTS * 8) as u64, U::STORAGE | U::COPY_DST);
        let results = buffer("query results", RESULT_BYTES, U::STORAGE | U::COPY_SRC);
        let params = buffer(
            "query params",
            std::mem::size_of::<QueryParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("query repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        use wgpu::BindingResource as B;
        let bind_group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wave_query"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: B::TextureView(&ocean.displacement_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: B::TextureView(&ocean.derivatives_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: B::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: results.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let slots = (0..SLOTS)
            .map(|_| Slot {
                staging: buffer("query staging", RESULT_BYTES, U::MAP_READ | U::COPY_DST),
                count: 0,
                state: Arc::new(AtomicU8::new(IDLE)),
            })
            .collect();
        WaveQuery {
            queue: ctx.queue.clone(),
            pipeline,
            bind_group,
            points,
            results,
            params,
            slots,
            requested: Vec::new(),
            recorded: None,
            next: 0,
        }
    }

    /// Points (world xz) to evaluate in the next `record`; at most MAX_QUERY_POINTS are used.
    pub fn request(&mut self, points: &[Vec2]) {
        self.requested = points.iter().take(MAX_QUERY_POINTS).copied().collect();
    }

    /// Records the query after this frame's wave update. Skipped when every readback slot is
    /// still busy (results then simply arrive a frame later).
    pub fn record(&mut self, enc: &mut wgpu::CommandEncoder) {
        self.recorded = None;
        let i = self.next;
        if self.requested.is_empty() || self.slots[i].state.load(Ordering::Acquire) != IDLE {
            return;
        }
        let n = self.requested.len();
        let flat: Vec<[f32; 2]> = self.requested.iter().map(|p| p.to_array()).collect();
        self.queue.write_buffer(&self.points, 0, bytemuck::cast_slice(&flat));
        let p = QueryParams {
            count: n as u32,
            _pad: [0; 3],
            sizes: CASCADE_SIZES_M,
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wave query"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.results, 0, &self.slots[i].staging, 0, (n * 16) as u64);
        self.slots[i].count = n;
        self.recorded = Some(i);
        self.next = (i + 1) % SLOTS;
    }

    pub fn after_submit(&mut self) {
        let Some(i) = self.recorded.take() else { return };
        let slot = &self.slots[i];
        slot.state.store(MAPPING, Ordering::Release);
        let state = slot.state.clone();
        slot.staging
            .slice(..(slot.count * 16) as u64)
            .map_async(wgpu::MapMode::Read, move |r| {
                state.store(if r.is_ok() { MAPPED } else { IDLE }, Ordering::Release);
            });
    }

    /// The newest finished results, if any arrived since the last call (non-blocking).
    pub fn poll(&mut self, device: &wgpu::Device) -> Option<Vec<(f32, Vec3)>> {
        let _ = device.poll(wgpu::PollType::Poll);
        let mut newest = None;
        // Oldest first, so the last one read is the newest.
        for k in 0..SLOTS {
            let i = (self.next + k) % SLOTS;
            let slot = &self.slots[i];
            if slot.state.load(Ordering::Acquire) != MAPPED {
                continue;
            }
            let range = ..(slot.count * 16) as u64;
            let data: Vec<[f32; 4]> =
                bytemuck::cast_slice(&slot.staging.slice(range).get_mapped_range().expect("mapped query")).to_vec();
            slot.staging.unmap();
            slot.state.store(IDLE, Ordering::Release);
            newest = Some(data.iter().map(|r| (r[0], Vec3::new(r[1], r[2], r[3]))).collect());
        }
        newest
    }

    /// Blocking query of the ocean's current textures. Tests only.
    pub fn read_blocking(&mut self, ctx: &GpuContext, points: &[Vec2]) -> Vec<(f32, Vec3)> {
        self.request(points);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        self.record(&mut enc);
        ctx.queue.submit([enc.finish()]);
        self.after_submit();
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        self.poll(&ctx.device).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};

    /// A single Gerstner wave: point p0 on the flat sea moves to p0 + D(p0).
    fn gerstner(amplitude: f32, wavelength: f32, choppiness: f32, dir: Vec2) -> impl Fn(Vec2) -> Vec3 {
        let k = std::f32::consts::TAU / wavelength;
        move |p: Vec2| {
            let phase = k * dir.dot(p);
            let h = dir * (-choppiness * amplitude * phase.sin());
            Vec3::new(h.x, amplitude * phase.cos(), h.y)
        }
    }

    #[test]
    fn inverts_a_choppy_gerstner_wave() {
        let d = gerstner(1.0, 20.0, 0.8, Vec2::new(0.6, 0.8));
        for i in 0..50 {
            let src = Vec2::new(i as f32 * 1.37 - 30.0, i as f32 * 0.71 + 5.0);
            let dd = d(src);
            let displaced = src + Vec2::new(dd.x, dd.z);
            let found = invert_horizontal(displaced, &d);
            assert!((found - src).length() < 0.01, "src {src} found {found}");
        }
    }

    #[test]
    fn no_displacement_is_identity() {
        let p = Vec2::new(3.0, -4.0);
        assert_eq!(invert_horizontal(p, |_| Vec3::ZERO), p);
    }
}
