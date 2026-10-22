//! GPU particles: spray thrown off breaking crests, sea mist drifting near them, and spray
//! landing back as foam. Anything else that splashes (a boat hull, its wake) pushes
//! `SpawnRequest`s.

use crate::waves::gpu_ocean::DEPOSIT_RES;

pub const MAX_PARTICLES: u32 = 65_536;
/// Spray starts once the wind is strong enough to tear crests.
const SPRAY_WIND_MIN: f32 = 8.0;
const SPRAY_RATE: f32 = 0.6;
const MIST_RATE: f32 = 0.004;

pub const KIND_SPRAY: u32 = 0;
pub const KIND_MIST: u32 = 1;
/// Never drawn: deposits foam where it is on its first update, then dies (wakes).
pub const KIND_FOAM_MARK: u32 = 2;

/// A burst of `count` particles at a point (world), all starting with `velocity`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpawnRequest {
    pub position: [f32; 3],
    pub kind: u32,
    pub velocity: [f32; 3],
    pub count: u32,
}

/// Spray particles per second per m² of breaking crest.
pub fn spray_rate_per_m2(wind: f32, amount: f32) -> f32 {
    let x = (wind - SPRAY_WIND_MIN).max(0.0);
    amount.max(0.0) * SPRAY_RATE * x * x / (1.0 + 0.05 * x * x)
}

/// Mist puffs per second per m² of compressed crest: faint at a gentle breeze.
pub fn mist_rate_per_m2(wind: f32, amount: f32) -> f32 {
    amount.max(0.0) * MIST_RATE * (wind / 5.0).powi(2)
}

/// Keeps requests in order, truncating counts so the total fits `budget`; drops empty ones.
pub fn pack_requests(reqs: &[SpawnRequest], budget: u32) -> Vec<SpawnRequest> {
    let mut left = budget;
    let mut out = Vec::new();
    for r in reqs {
        if left == 0 {
            break;
        }
        let n = r.count.min(left);
        if n > 0 {
            out.push(SpawnRequest { count: n, ..*r });
            left -= n;
        }
    }
    out
}

const PARTICLE_BYTES: u64 = 48;
const MAX_REQUESTS: usize = 256;
/// Particles CPU requests may add per frame.
pub const REQUEST_BUDGET: u32 = 4096;
const EMIT_CANDIDATES: u32 = 16_384;
/// Side of the square ahead of the camera where crests are checked for spray, m.
const EMIT_AREA_M: f32 = 192.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    dt: f32,
    spray_rate: f32,
    mist_rate: f32,
    threshold: f32,
    wind: [f32; 2],
    frame_index: u32,
    request_count: u32,
    area_origin: [f32; 2],
    area_size: f32,
    candidates: u32,
    shift: [f32; 2],
    _pad: [f32; 2],
}

/// What the particles need from the rest of the frame.
pub struct ParticleInputs {
    pub dt: f32,
    pub wind_speed: f32,
    pub wind_dir_deg: f32,
    pub amount: f32,
    pub foam_threshold: f32,
    pub cam_pos: glam::Vec3,
    pub cam_forward: glam::Vec3,
}

pub struct ParticleSystem {
    queue: wgpu::Queue,
    params: wgpu::Buffer,
    requests_buf: wgpu::Buffer,
    pending: Vec<SpawnRequest>,
    /// Origin rebase to apply to every live particle in the next `record`.
    pending_shift: glam::Vec2,
    sim_group: wgpu::BindGroup,
    clear: wgpu::ComputePipeline,
    emit: wgpu::ComputePipeline,
    spawn: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    draw_layout: wgpu::BindGroupLayout,
    draw_group: wgpu::BindGroup,
    draw: wgpu::RenderPipeline,
    particles: wgpu::Buffer,
    frame: u32,
}

fn layout_entry(binding: u32, visibility: wgpu::ShaderStages, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    }
}

fn storage(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

const UNIFORM: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Uniform,
    has_dynamic_offset: false,
    min_binding_size: None,
};

impl ParticleSystem {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d: &wgpu::Device,
        queue: &wgpu::Queue,
        lib: &crate::gpu::shaders::ShaderLibrary,
        frame: &wgpu::Buffer,
        ocean: &crate::waves::gpu_ocean::GpuOcean,
        sky: &crate::render::sky::SkyLuts,
        g: &crate::render::gbuffer::GBuffer,
        seabed: &crate::render::seabed::SeabedMap,
    ) -> ParticleSystem {
        let buffer = |label, size, usage| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        let particles = buffer("particles", MAX_PARTICLES as u64 * PARTICLE_BYTES, U::STORAGE);
        let head = buffer("particle ring head", 16, U::STORAGE);
        let params = buffer(
            "particle params",
            std::mem::size_of::<SimParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        let requests_buf = buffer("spawn requests", (MAX_REQUESTS * 32) as u64, U::STORAGE | U::COPY_DST);
        let cs = wgpu::ShaderStages::COMPUTE;
        let tex_array = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        };
        let sim_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle sim"),
            entries: &[
                layout_entry(0, cs, UNIFORM),
                layout_entry(1, cs, tex_array),
                layout_entry(2, cs, tex_array),
                layout_entry(3, cs, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                layout_entry(4, cs, storage(false)),
                layout_entry(5, cs, storage(false)),
                layout_entry(6, cs, storage(false)),
                layout_entry(7, cs, UNIFORM),
                layout_entry(8, cs, storage(true)),
                layout_entry(
                    9,
                    cs,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
            ],
        });
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("particle waves"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        use wgpu::BindingResource as B;
        let sim_group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle sim"),
            layout: &sim_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: B::TextureView(&ocean.displacement_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: B::TextureView(&ocean.derivatives_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: B::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: head.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: ocean.deposits().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: requests_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: B::TextureView(seabed.view()),
                },
            ],
        });
        let sim_pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle sim"),
            bind_group_layouts: &[Some(&sim_layout)],
            immediate_size: 0,
        });
        let sim_module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particles_sim"),
            source: wgpu::ShaderSource::Wgsl(lib.source("particles_sim").expect("shader source").into()),
        });
        let compute = |entry: &str| {
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&sim_pl),
                module: &sim_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (clear, emit, spawn, update) = (
            compute("clear_deposits"),
            compute("emit"),
            compute("spawn_requests"),
            compute("update"),
        );

        let fs = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let draw_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle draw"),
            entries: &[
                layout_entry(0, fs, UNIFORM),
                layout_entry(1, fs, storage(true)),
                layout_entry(
                    2,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
            ],
        });
        let draw_group = Self::draw_group(d, &draw_layout, frame, &particles, g);
        let draw_pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle draw"),
            bind_group_layouts: &[Some(&draw_layout), Some(sky.layout())],
            immediate_size: 0,
        });
        let draw_module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particles_draw"),
            source: wgpu::ShaderSource::Wgsl(lib.source("particles_draw").expect("shader source").into()),
        });
        let premultiplied = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };
        let draw = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle draw"),
            layout: Some(&draw_pl),
            vertex: wgpu::VertexState {
                module: &draw_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &draw_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::render::gbuffer::HDR_FORMAT,
                    blend: Some(premultiplied),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        ParticleSystem {
            queue: queue.clone(),
            params,
            requests_buf,
            pending: Vec::new(),
            pending_shift: glam::Vec2::ZERO,
            sim_group,
            clear,
            emit,
            spawn,
            update,
            draw_layout,
            draw_group,
            draw,
            particles,
            frame: 0,
        }
    }

    fn draw_group(
        d: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        frame: &wgpu::Buffer,
        particles: &wgpu::Buffer,
        g: &crate::render::gbuffer::GBuffer,
    ) -> wgpu::BindGroup {
        d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle draw"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&g.depth),
                },
            ],
        })
    }

    /// Call after the G-buffer is recreated.
    pub fn resize(&mut self, d: &wgpu::Device, frame: &wgpu::Buffer, g: &crate::render::gbuffer::GBuffer) {
        self.draw_group = Self::draw_group(d, &self.draw_layout, frame, &self.particles, g);
    }

    /// Follows an origin rebase: every live particle moves by −shift (in the next `record`).
    pub fn shift(&mut self, shift: glam::Vec2) {
        self.pending_shift += shift;
    }

    /// Queues a burst for the next frame (boat hull, wake, ...).
    pub fn push(&mut self, r: SpawnRequest) {
        self.pending.push(r);
    }

    /// Simulation: clear last frame's deposits (the foam pass has read them), emit, spawn
    /// requests, integrate. Record after the ocean update.
    pub fn record(&mut self, enc: &mut wgpu::CommandEncoder, inp: &ParticleInputs) {
        let reqs = pack_requests(&std::mem::take(&mut self.pending), REQUEST_BUDGET);
        let reqs = &reqs[..reqs.len().min(MAX_REQUESTS)];
        if !reqs.is_empty() {
            self.queue
                .write_buffer(&self.requests_buf, 0, bytemuck::cast_slice(reqs));
        }
        let fwd = glam::Vec2::new(inp.cam_forward.x, inp.cam_forward.z).normalize_or(glam::Vec2::Y);
        let centre = glam::Vec2::new(inp.cam_pos.x, inp.cam_pos.z) + fwd * (EMIT_AREA_M * 0.5);
        let a = inp.wind_dir_deg.to_radians();
        let p = SimParams {
            dt: inp.dt.clamp(0.0, 0.1),
            spray_rate: spray_rate_per_m2(inp.wind_speed, inp.amount),
            mist_rate: mist_rate_per_m2(inp.wind_speed, inp.amount),
            threshold: inp.foam_threshold,
            wind: [a.sin() * inp.wind_speed, a.cos() * inp.wind_speed],
            frame_index: self.frame,
            request_count: reqs.len() as u32,
            area_origin: (centre - glam::Vec2::splat(EMIT_AREA_M * 0.5)).to_array(),
            area_size: EMIT_AREA_M,
            candidates: EMIT_CANDIDATES,
            shift: std::mem::take(&mut self.pending_shift).to_array(),
            _pad: [0.0; 2],
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particles"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.sim_group, &[]);
        pass.set_pipeline(&self.clear);
        pass.dispatch_workgroups(DEPOSIT_RES * DEPOSIT_RES / 64, 1, 1);
        pass.set_pipeline(&self.emit);
        pass.dispatch_workgroups(EMIT_CANDIDATES.div_ceil(64), 1, 1);
        if !reqs.is_empty() {
            pass.set_pipeline(&self.spawn);
            pass.dispatch_workgroups((reqs.len() as u32).div_ceil(64), 1, 1);
        }
        pass.set_pipeline(&self.update);
        pass.dispatch_workgroups(MAX_PARTICLES / 64, 1, 1);
        self.frame = self.frame.wrapping_add(1);
    }

    /// Draws into the HDR target (after shading).
    pub fn draw(&self, enc: &mut wgpu::CommandEncoder, hdr: &wgpu::TextureView, sky: &crate::render::sky::SkyLuts) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.draw);
        pass.set_bind_group(0, &self.draw_group, &[]);
        pass.set_bind_group(1, sky.bind_group(), &[]);
        pass.draw(0..6, 0..MAX_PARTICLES);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(count: u32) -> SpawnRequest {
        SpawnRequest {
            position: [0.0; 3],
            kind: 0,
            velocity: [0.0; 3],
            count,
        }
    }

    #[test]
    fn no_spray_in_a_gentle_breeze_or_when_turned_off() {
        assert_eq!(spray_rate_per_m2(5.0, 1.0), 0.0);
        assert_eq!(spray_rate_per_m2(15.0, 0.0), 0.0);
        assert!(spray_rate_per_m2(15.0, 1.0) > 0.0);
    }

    #[test]
    fn spray_grows_with_wind_and_stays_finite() {
        let mut prev = 0.0;
        for i in 0..=100 {
            let w = 0.5 + i as f32 * 0.245;
            let r = spray_rate_per_m2(w, 1.0);
            assert!(r.is_finite() && r >= prev, "wind {w}: {r} < {prev}");
            prev = r;
        }
        assert!(spray_rate_per_m2(25.0, 2.0).is_finite());
    }

    #[test]
    fn mist_is_faint_but_present_at_defaults() {
        let m = mist_rate_per_m2(5.0, 1.0);
        assert!(m > 0.0 && m < spray_rate_per_m2(15.0, 1.0));
        assert_eq!(mist_rate_per_m2(5.0, 0.0), 0.0);
    }

    #[test]
    fn packing_never_exceeds_the_budget() {
        let r = pack_requests(&[req(10), req(0), req(30), req(5)], 25);
        assert_eq!(r.iter().map(|q| q.count).sum::<u32>(), 25);
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].count, r[1].count), (10, 15));
    }

    #[test]
    fn packing_preserves_order_and_drops_empty() {
        let r = pack_requests(&[req(0), req(3), req(0), req(4)], 100);
        assert_eq!(r.iter().map(|q| q.count).collect::<Vec<_>>(), vec![3, 4]);
    }

    #[test]
    fn zero_budget_packs_nothing() {
        assert!(pack_requests(&[req(5)], 0).is_empty());
    }
}
