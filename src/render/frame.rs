//! Owns every pass and records one frame in order.

use glam::{Mat4, Vec3};

use crate::gpu::adapter::GpuContext;
use crate::params::OceanParams;
use crate::render::ocean_mesh::{BASE_CELL_M, GRID, LEVELS, OceanMesh};
use crate::scene::camera::{CameraRig, FixedCamera};
use crate::waves::gpu_ocean::GpuOcean;
use crate::waves::spectrum::{CASCADE_SIZES_M, REPEAT_PERIOD_S, build_spectrum, cascade_band};

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SEED: u64 = 42;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sun_dir: [f32; 4],
    sizes: [f32; 4],
    lambda_min: [f32; 4],
    grid: [f32; 4],
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ocean: GpuOcean,
    mesh: OceanMesh,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    sky_pipeline: wgpu::RenderPipeline,
    ocean_pipeline: wgpu::RenderPipeline,
    depth: wgpu::TextureView,
    extent: (u32, u32),
    camera: FixedCamera,
}

fn depth_view(device: &wgpu::Device, (w, h): (u32, u32)) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// Temporary until the sky module owns it (Task 4).
fn sun_dir(p: &OceanParams) -> Vec3 {
    let (el, az) = (p.sun_elevation_deg.to_radians(), p.sun_azimuth_deg.to_radians());
    Vec3::new(az.sin() * el.cos(), el.sin(), az.cos() * el.cos())
}

impl Renderer {
    pub fn new(ctx: &GpuContext, format: wgpu::TextureFormat, extent: (u32, u32), params: &OceanParams) -> Renderer {
        let d = &ctx.device;
        let ocean = GpuOcean::new(ctx, &build_spectrum(params, SEED));
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
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
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
            label: Some("ocean frame"),
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
            ],
        });
        let bind_group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ocean frame"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ocean.displacement_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&ocean.derivatives_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ocean frame"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ocean_gbuffer"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ocean_gbuffer.wgsl").into()),
        });
        let pipeline = |label, vs, fs, depth: Option<wgpu::DepthStencilState>| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: depth,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let depth_state = |write: bool, compare| wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(write),
            depth_compare: Some(compare),
            stencil: Default::default(),
            bias: Default::default(),
        };
        let sky_pipeline = pipeline("sky", "vs_sky", "fs_sky", Some(depth_state(false, wgpu::CompareFunction::Always)));
        let ocean_pipeline = pipeline("ocean", "vs_ocean", "fs_ocean", Some(depth_state(true, wgpu::CompareFunction::Greater)));
        Renderer {
            device: d.clone(),
            queue: ctx.queue.clone(),
            ocean,
            mesh,
            uniforms,
            bind_group,
            sky_pipeline,
            ocean_pipeline,
            depth: depth_view(d, extent),
            extent,
            camera: FixedCamera { yaw_deg: params.sun_azimuth_deg, ..Default::default() },
        }
    }

    pub fn resize(&mut self, extent: (u32, u32)) {
        if extent != self.extent {
            self.extent = extent;
            self.depth = depth_view(&self.device, extent);
        }
    }

    /// Records and submits one frame into `view` (same size as the last `resize`).
    pub fn render(&mut self, view: &wgpu::TextureView, params: &OceanParams, t: f32, _dt: f32) {
        self.camera.yaw_deg = params.sun_azimuth_deg;
        let cam = self.camera.pose();
        let aspect = self.extent.0 as f32 / self.extent.1 as f32;
        let vp: Mat4 = cam.view_proj(aspect);
        let s = sun_dir(params);
        let lambda_min = std::array::from_fn(|c| std::f32::consts::TAU / cascade_band(c).1);
        let u = FrameUniforms {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            cam_pos: [cam.position.x, cam.position.y, cam.position.z, t],
            sun_dir: [s.x, s.y, s.z, 0.0],
            sizes: CASCADE_SIZES_M,
            lambda_min,
            grid: [BASE_CELL_M, LEVELS as f32, GRID as f32, 0.0],
        };
        self.queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        let mut enc = self.device.create_command_encoder(&Default::default());
        self.ocean.update(&mut enc, t % REPEAT_PERIOD_S, params.choppiness);
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ocean forward"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.sky_pipeline);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.ocean_pipeline);
            pass.set_index_buffer(self.mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.mesh.index_count, 0, 0..LEVELS);
        }
        self.queue.submit([enc.finish()]);
    }
}
