//! Owns every pass and records one frame in order:
//! wave sim → sky LUTs (if the sun moved) → G-buffer → shade (HDR) → bloom + tonemap.

use glam::Mat4;

use crate::gpu::adapter::GpuContext;
use crate::gpu::extent::render_extent;
use crate::gpu::timing::{GpuTimer, PassStats};
use crate::params::OceanParams;
use crate::render::gbuffer::{DEPTH_FORMAT, GBuffer, MATERIAL_FORMAT, MOTION_FORMAT, NORMAL_FORMAT};
use crate::render::ocean_mesh::{BASE_CELL_M, GRID, LEVELS, OceanMesh};
use crate::render::post::{Post, exposure_ev, exposure_multiplier};
use crate::render::reflect::ReflectionPass;
use crate::gpu::shaders::ShaderLibrary;
use crate::render::shade::ShadePass;
use crate::render::sky::{SkyLuts, sun_direction};
use crate::scene::camera::{CameraRig, FixedCamera};
use crate::waves::foam::foam_params;
use crate::waves::gpu_ocean::GpuOcean;
use crate::waves::spectrum::{
    CASCADE_SIZES_M, REPEAT_PERIOD_S, SpectrumSet, build_spectrum, cascade_band, slope_variance,
    spectrum_significant_height, unresolved_slope_variance,
};

const SEED: u64 = 42;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sun_dir: [f32; 4],
    sizes: [f32; 4],
    lambda_min: [f32; 4],
    grid: [f32; 4],
    slope_var: [f32; 4],
    misc: [f32; 4],
    resolution: [f32; 4],
}

/// The parameters that change the wave spectrum (rebuilding it costs a few milliseconds).
#[derive(PartialEq, Clone, Copy)]
struct SpectrumKey {
    wind_speed: f32,
    wind_dir_deg: f32,
    fetch_km: f32,
    swell_amount: f32,
}

impl SpectrumKey {
    fn of(p: &OceanParams) -> Self {
        SpectrumKey { wind_speed: p.wind_speed, wind_dir_deg: p.wind_dir_deg, fetch_km: p.fetch_km, swell_amount: p.swell_amount }
    }
}

struct SpectrumStats {
    slope_var: [f32; 4],
    unresolved: f32,
    hs: f32,
}

impl SpectrumStats {
    fn of(s: &SpectrumSet, p: &OceanParams) -> Self {
        SpectrumStats {
            slope_var: slope_variance(s),
            unresolved: unresolved_slope_variance(s, p.wind_speed),
            hs: spectrum_significant_height(s),
        }
    }
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ocean: GpuOcean,
    sky: SkyLuts,
    mesh: OceanMesh,
    uniforms: wgpu::Buffer,
    /// One per foam ping-pong side (see GpuOcean::foam_index).
    ocean_bind_groups: [wgpu::BindGroup; 2],
    gbuffer_pipeline: wgpu::RenderPipeline,
    gbuffer: GBuffer,
    reflections: ReflectionPass,
    shade: ShadePass,
    post: Post,
    surface_extent: (u32, u32),
    camera: FixedCamera,
    prev_view_proj: Option<Mat4>,
    spectrum_key: SpectrumKey,
    stats: SpectrumStats,
    timer: GpuTimer,
}

fn internal_extent(surface: (u32, u32), scale: f32) -> (u32, u32) {
    render_extent(surface.0, surface.1, scale).unwrap_or((1, 1))
}

impl Renderer {
    pub fn new(
        ctx: &GpuContext,
        lib: &ShaderLibrary,
        format: wgpu::TextureFormat,
        surface_extent: (u32, u32),
        params: &OceanParams,
    ) -> Renderer {
        let d = &ctx.device;
        let spectrum = build_spectrum(params, SEED);
        let ocean = GpuOcean::new(ctx, lib, &spectrum);
        let sky = SkyLuts::new(ctx, lib);
        let mesh = OceanMesh::new(d);
        let uniforms = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ocean repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let vis = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ocean gbuffer"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vis,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex(1),
                tex(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: vis,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        use wgpu::BindingResource as B;
        let ocean_bind_groups = std::array::from_fn(|i| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ocean gbuffer"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: B::TextureView(&ocean.displacement_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: B::TextureView(&ocean.derivatives_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: B::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 4, resource: B::TextureView(&ocean.foam_views()[i]) },
                ],
            })
        });
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ocean gbuffer"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ocean_gbuffer"),
            source: wgpu::ShaderSource::Wgsl(
                lib.source("ocean_gbuffer").expect("shader source").into(),
            ),
        });
        let gbuffer_pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ocean gbuffer"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_ocean"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_ocean"),
                compilation_options: Default::default(),
                targets: &[Some(NORMAL_FORMAT.into()), Some(MATERIAL_FORMAT.into()), Some(MOTION_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let gbuffer = GBuffer::new(d, internal_extent(surface_extent, params.render_scale));
        let reflections = ReflectionPass::new(d, lib, &ctx.queue, &uniforms, &gbuffer);
        let shade = ShadePass::new(d, lib, &uniforms, &sky, &gbuffer, reflections.views());
        let post = Post::new(d, lib, format, &gbuffer);
        Renderer {
            device: d.clone(),
            queue: ctx.queue.clone(),
            ocean,
            sky,
            mesh,
            uniforms,
            ocean_bind_groups,
            gbuffer_pipeline,
            gbuffer,
            reflections,
            shade,
            post,
            surface_extent,
            camera: FixedCamera { yaw_deg: params.sun_azimuth_deg, ..Default::default() },
            prev_view_proj: None,
            spectrum_key: SpectrumKey::of(params),
            stats: SpectrumStats::of(&spectrum, params),
            timer: GpuTimer::new(d, &ctx.queue),
        }
    }

    pub fn resize(&mut self, surface_extent: (u32, u32)) {
        self.surface_extent = surface_extent;
    }

    fn ensure_targets(&mut self, scale: f32) {
        let want = internal_extent(self.surface_extent, scale);
        if want != self.gbuffer.extent {
            self.gbuffer = GBuffer::new(&self.device, want);
            self.reflections.resize(&self.device, &self.uniforms, &self.gbuffer);
            self.shade.rebind(&self.device, &self.uniforms, &self.gbuffer, self.reflections.views());
            self.post.resize(&self.device, &self.gbuffer);
            self.prev_view_proj = None;
        }
    }

    fn ensure_spectrum(&mut self, params: &OceanParams) {
        let key = SpectrumKey::of(params);
        if key != self.spectrum_key {
            let s = build_spectrum(params, SEED);
            self.ocean.set_spectrum(&self.queue, &s);
            self.stats = SpectrumStats::of(&s, params);
            self.spectrum_key = key;
        }
    }

    /// GPU time per pass and per frame over the last few seconds.
    pub fn gpu_stats(&self) -> (Vec<(&'static str, PassStats)>, PassStats) {
        (self.timer.stats(), self.timer.total())
    }

    pub fn gpu_timer(&mut self) -> &mut GpuTimer {
        &mut self.timer
    }

    /// Records and submits one frame into `view` (the surface texture). `overlay` records any
    /// passes drawn on top (the UI) into the same submission.
    pub fn render(
        &mut self,
        view: &wgpu::TextureView,
        params: &OceanParams,
        t: f32,
        dt: f32,
        overlay: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView),
    ) {
        self.timer.poll(&self.device);
        let params = params.clamped();
        self.ensure_targets(params.render_scale);
        self.ensure_spectrum(&params);
        self.camera.yaw_deg = params.sun_azimuth_deg;
        let cam = self.camera.pose();
        let (w, h) = self.gbuffer.extent;
        let vp = cam.view_proj(w as f32 / h as f32);
        let prev = self.prev_view_proj.unwrap_or(vp);
        self.prev_view_proj = Some(vp);
        let sun = sun_direction(params.sun_elevation_deg, params.sun_azimuth_deg);
        let exposure = exposure_multiplier(exposure_ev(params.sun_elevation_deg));
        let u = FrameUniforms {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            prev_view_proj: prev.to_cols_array_2d(),
            cam_pos: [cam.position.x, cam.position.y, cam.position.z, t],
            sun_dir: [sun.x, sun.y, sun.z, 0.0],
            sizes: CASCADE_SIZES_M,
            lambda_min: std::array::from_fn(|c| std::f32::consts::TAU / cascade_band(c).1),
            grid: [BASE_CELL_M, LEVELS as f32, GRID as f32, 0.0],
            slope_var: self.stats.slope_var,
            misc: [self.stats.unresolved, self.stats.hs, exposure, params.wind_dir_deg.to_radians()],
            resolution: [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32],
        };
        self.queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        let mut enc = self.device.create_command_encoder(&Default::default());
        self.timer.start(&mut enc);
        self.ocean.update(&mut enc, t % REPEAT_PERIOD_S, params.choppiness, foam_params(&params), dt);
        self.timer.mark(&mut enc, "ocean sim");
        self.sky.update(&mut enc, sun);
        self.timer.mark(&mut enc, "sky");
        {
            let g = &self.gbuffer;
            let color = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })
            };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gbuffer"),
                color_attachments: &[color(&g.normal), color(&g.material), color(&g.motion)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &g.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.gbuffer_pipeline);
            pass.set_bind_group(0, &self.ocean_bind_groups[self.ocean.foam_index()], &[]);
            pass.set_index_buffer(self.mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.mesh.index_count, 0, 0..LEVELS);
        }
        self.timer.mark(&mut enc, "gbuffer");
        self.reflections.record(&mut enc, params.reflection_mode);
        self.timer.mark(&mut enc, "reflections");
        self.shade.record(&mut enc, &self.gbuffer, &self.sky, self.reflections.index());
        self.gbuffer.record_history(&mut enc);
        self.timer.mark(&mut enc, "shade");
        self.post.record(&mut enc, &self.queue, view, exposure);
        self.timer.mark(&mut enc, "post");
        overlay(&mut enc, view);
        self.timer.mark(&mut enc, "ui");
        self.timer.finish(&mut enc);
        self.queue.submit([enc.finish()]);
        self.timer.after_submit();
    }
}
