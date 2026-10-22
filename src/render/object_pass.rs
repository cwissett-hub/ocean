//! Draws any number of mesh instances (buoy, raft, islands, palms) into the G-buffer and into
//! the opaque pre-pass that water refraction reads. Each instance has a stable id so its
//! previous transform (for motion vectors) survives reordering.

use std::collections::HashMap;

use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::gpu::shaders::ShaderLibrary;
use crate::render::gbuffer::{DEPTH_FORMAT, GBuffer, MATERIAL_FORMAT, MOTION_FORMAT, NORMAL_FORMAT};
use crate::scene::mesh::MeshData;

pub const MAX_INSTANCES: usize = 512;
/// Uniform slots are 256-byte aligned for dynamic offsets.
const SLOT: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u32);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelUniforms {
    transform: [[f32; 4]; 4],
    prev_transform: [[f32; 4]; 4],
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    vertex_count: u32,
}

pub struct ObjectPass {
    queue: wgpu::Queue,
    meshes: Vec<Option<GpuMesh>>,
    models: wgpu::Buffer,
    model_group: wgpu::BindGroup,
    gbuffer_pipeline: wgpu::RenderPipeline,
    albedo_pipeline: wgpu::RenderPipeline,
    /// This frame's draws: (mesh, uniform slot).
    draws: Vec<(MeshHandle, u32)>,
    prev: HashMap<u64, Mat4>,
}

impl ObjectPass {
    /// `frame_layout` is the G-buffer pass's group-0 layout (frame uniforms first).
    pub fn new(
        d: &wgpu::Device,
        queue: &wgpu::Queue,
        lib: &ShaderLibrary,
        frame_layout: &wgpu::BindGroupLayout,
    ) -> ObjectPass {
        let models = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object models"),
            size: SLOT * MAX_INSTANCES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let model_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object model"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ModelUniforms>() as u64),
                },
                count: None,
            }],
        });
        let model_group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object model"),
            layout: &model_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &models,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<ModelUniforms>() as u64),
                }),
            }],
        });
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("objects"),
            bind_group_layouts: &[Some(frame_layout), Some(&model_layout)],
            immediate_size: 0,
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("object_gbuffer"),
            source: wgpu::ShaderSource::Wgsl(lib.source("object_gbuffer").expect("shader source").into()),
        });
        let attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32, 3 => Float32];
        let layout = Some(wgpu::VertexBufferLayout {
            array_stride: 32,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &attrs,
        });
        let pipeline = |label, fs, targets: &[Option<wgpu::ColorTargetState>]| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_object"),
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
            "objects gbuffer",
            "fs_gbuffer",
            &[
                Some(NORMAL_FORMAT.into()),
                Some(MATERIAL_FORMAT.into()),
                Some(MOTION_FORMAT.into()),
            ],
        );
        let albedo_pipeline = pipeline("objects albedo", "fs_albedo", &[Some(MATERIAL_FORMAT.into())]);
        ObjectPass {
            queue: queue.clone(),
            meshes: Vec::new(),
            models,
            model_group,
            gbuffer_pipeline,
            albedo_pipeline,
            draws: Vec::new(),
            prev: HashMap::new(),
        }
    }

    pub fn add_mesh(&mut self, d: &wgpu::Device, mesh: &MeshData) -> MeshHandle {
        let usage = wgpu::BufferUsages::BLAS_INPUT;
        let vertices = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("object vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX | usage,
        });
        let indices = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("object indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX | usage,
        });
        let gpu = GpuMesh {
            vertices,
            indices,
            index_count: mesh.indices.len() as u32,
            vertex_count: mesh.vertices.len() as u32,
        };
        if let Some(free) = self.meshes.iter().position(Option::is_none) {
            self.meshes[free] = Some(gpu);
            MeshHandle(free as u32)
        } else {
            self.meshes.push(Some(gpu));
            MeshHandle(self.meshes.len() as u32 - 1)
        }
    }

    /// Releases a mesh (an island that streamed out); its handle may be reused.
    pub fn remove_mesh(&mut self, h: MeshHandle) {
        if let Some(slot) = self.meshes.get_mut(h.0 as usize) {
            *slot = None;
        }
    }

    /// This frame's instances: (stable id, mesh, model transform). At most MAX_INSTANCES.
    pub fn set_instances(&mut self, items: &[(u64, MeshHandle, Mat4)]) {
        self.draws.clear();
        let mut next_prev = HashMap::with_capacity(items.len());
        for (slot, (id, mesh, t)) in items.iter().take(MAX_INSTANCES).enumerate() {
            if self.meshes.get(mesh.0 as usize).is_none_or(Option::is_none) {
                continue;
            }
            let prev = self.prev.get(id).copied().unwrap_or(*t);
            let u = ModelUniforms {
                transform: t.to_cols_array_2d(),
                prev_transform: prev.to_cols_array_2d(),
            };
            self.queue
                .write_buffer(&self.models, slot as u64 * SLOT, bytemuck::bytes_of(&u));
            self.draws.push((*mesh, slot as u32));
            next_prev.insert(*id, *t);
        }
        self.prev = next_prev;
    }

    /// Shifts every remembered previous transform (world origin rebase).
    pub fn rebase(&mut self, shift: glam::Vec2) {
        let m = Mat4::from_translation(glam::Vec3::new(-shift.x, 0.0, -shift.y));
        for t in self.prev.values_mut() {
            *t = m * *t;
        }
    }

    /// Objects only, into the opaque depth/albedo targets (before the G-buffer pass).
    pub fn record_opaque(&self, enc: &mut wgpu::CommandEncoder, g: &GBuffer, frame_group: &wgpu::BindGroup) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("objects opaque"),
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
        for (mesh, slot) in &self.draws {
            let Some(Some(m)) = self.meshes.get(mesh.0 as usize) else {
                continue;
            };
            pass.set_bind_group(1, &self.model_group, &[*slot * SLOT as u32]);
            pass.set_vertex_buffer(0, m.vertices.slice(..));
            pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..1);
        }
    }

    /// Buffers for a BLAS: (vertices, indices, index count, vertex count). Vertex stride 32.
    pub fn mesh_geometry(&self, h: MeshHandle) -> Option<(&wgpu::Buffer, &wgpu::Buffer, u32, u32)> {
        self.meshes
            .get(h.0 as usize)
            .and_then(|m| m.as_ref())
            .map(|m| (&m.vertices, &m.indices, m.index_count, m.vertex_count))
    }
}
