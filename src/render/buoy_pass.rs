//! Draws the buoy into the G-buffer and into the opaque pre-pass used by water refraction.

use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::gpu::shaders::ShaderLibrary;
use crate::render::gbuffer::{DEPTH_FORMAT, GBuffer, MATERIAL_FORMAT, MOTION_FORMAT, NORMAL_FORMAT};
use crate::scene::buoy::BuoyMesh;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelUniforms {
    transform: [[f32; 4]; 4],
    prev_transform: [[f32; 4]; 4],
}

pub struct BuoyPass {
    queue: wgpu::Queue,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    model: wgpu::Buffer,
    model_group: wgpu::BindGroup,
    gbuffer_pipeline: wgpu::RenderPipeline,
    albedo_pipeline: wgpu::RenderPipeline,
    prev: Option<Mat4>,
}

impl BuoyPass {
    /// `frame_layout` is the G-buffer pass's group-0 layout (frame uniforms first).
    pub fn new(
        d: &wgpu::Device,
        queue: &wgpu::Queue,
        lib: &ShaderLibrary,
        frame_layout: &wgpu::BindGroupLayout,
        mesh: &BuoyMesh,
    ) -> BuoyPass {
        let vertices = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buoy vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
        });
        let indices = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buoy indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
        });
        let model = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buoy model"),
            size: std::mem::size_of::<ModelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let model_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("buoy model"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let model_group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("buoy model"),
            layout: &model_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: model.as_entire_binding(),
            }],
        });
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("buoy"),
            bind_group_layouts: &[Some(frame_layout), Some(&model_layout)],
            immediate_size: 0,
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("buoy_gbuffer"),
            source: wgpu::ShaderSource::Wgsl(lib.source("buoy_gbuffer").expect("shader source").into()),
        });
        let attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        let layout = Some(wgpu::VertexBufferLayout {
            array_stride: 24,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &attrs,
        });
        let pipeline = |label, fs, targets: &[Option<wgpu::ColorTargetState>]| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_buoy"),
                    compilation_options: Default::default(),
                    buffers: std::slice::from_ref(&layout),
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
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
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let gbuffer_pipeline = pipeline(
            "buoy gbuffer",
            "fs_gbuffer",
            &[
                Some(NORMAL_FORMAT.into()),
                Some(MATERIAL_FORMAT.into()),
                Some(MOTION_FORMAT.into()),
            ],
        );
        let albedo_pipeline = pipeline("buoy albedo", "fs_albedo", &[Some(MATERIAL_FORMAT.into())]);
        BuoyPass {
            queue: queue.clone(),
            vertices,
            indices,
            index_count: mesh.indices.len() as u32,
            model,
            model_group,
            gbuffer_pipeline,
            albedo_pipeline,
            prev: None,
        }
    }

    pub fn set_transform(&mut self, t: Mat4) {
        let prev = self.prev.unwrap_or(t);
        self.prev = Some(t);
        let u = ModelUniforms {
            transform: t.to_cols_array_2d(),
            prev_transform: prev.to_cols_array_2d(),
        };
        self.queue.write_buffer(&self.model, 0, bytemuck::bytes_of(&u));
    }

    /// The buoy's own pass into the opaque depth/albedo targets (before the G-buffer pass).
    pub fn record_opaque(&self, enc: &mut wgpu::CommandEncoder, g: &GBuffer, frame_group: &wgpu::BindGroup) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("buoy opaque"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &g.opaque_albedo,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &g.opaque_depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        self.draw(&mut pass, &self.albedo_pipeline, frame_group);
    }

    /// Draws into an already-open G-buffer pass.
    pub fn draw_gbuffer(&self, pass: &mut wgpu::RenderPass<'_>, frame_group: &wgpu::BindGroup) {
        self.draw(pass, &self.gbuffer_pipeline, frame_group);
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>, pipeline: &wgpu::RenderPipeline, frame_group: &wgpu::BindGroup) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, frame_group, &[]);
        pass.set_bind_group(1, &self.model_group, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    /// Buffers for the ray-tracing BLAS (Task 9).
    pub fn geometry(&self) -> (&wgpu::Buffer, &wgpu::Buffer, u32) {
        (&self.vertices, &self.indices, self.index_count)
    }
}
