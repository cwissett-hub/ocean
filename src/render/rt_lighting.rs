//! Ray-traced sun shadows and ambient occlusion from the objects (`rt_lighting.wgsl`): one
//! jittered ray each per pixel into the scene TLAS, resolved to soft shadows by TAA. Without
//! ray query the shade pass gets `white`, i.e. no shadows.

use crate::gpu::shaders::ShaderLibrary;
use crate::render::gbuffer::GBuffer;
use crate::render::rt_scene::RtScene;

pub const LIGHTING_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A 1×1 "fully lit" mask for devices without ray tracing.
pub fn white(d: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let t = d.create_texture(&wgpu::TextureDescriptor {
        label: Some("no shadows"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: LIGHTING_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        t.as_image_copy(),
        &[255, 255, 255, 255],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    t.create_view(&Default::default())
}

pub struct RtLighting {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    group: wgpu::BindGroup,
    view: wgpu::TextureView,
    extent: (u32, u32),
}

fn target(d: &wgpu::Device, extent: (u32, u32)) -> wgpu::TextureView {
    d.create_texture(&wgpu::TextureDescriptor {
        label: Some("rt lighting"),
        size: wgpu::Extent3d {
            width: extent.0,
            height: extent.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: LIGHTING_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
    .create_view(&Default::default())
}

impl RtLighting {
    pub fn new(
        d: &wgpu::Device,
        lib: &ShaderLibrary,
        frame: &wgpu::Buffer,
        g: &GBuffer,
        scene: &RtScene,
    ) -> RtLighting {
        let cs = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cs,
            ty,
            count: None,
        };
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt lighting"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(
                    1,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    2,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    3,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: LIGHTING_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                ),
                entry(4, wgpu::BindingType::AccelerationStructure { vertex_return: false }),
            ],
        });
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rt lighting"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rt_lighting"),
            source: wgpu::ShaderSource::Wgsl(lib.source("rt_lighting").expect("shader source").into()),
        });
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rt lighting"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let view = target(d, g.extent);
        let group = Self::group(d, &layout, frame, g, &view, scene);
        RtLighting {
            pipeline,
            layout,
            group,
            view,
            extent: g.extent,
        }
    }

    fn group(
        d: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        frame: &wgpu::Buffer,
        g: &GBuffer,
        view: &wgpu::TextureView,
        scene: &RtScene,
    ) -> wgpu::BindGroup {
        use wgpu::BindingResource as B;
        d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt lighting"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: B::TextureView(&g.depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: B::TextureView(&g.normal),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: B::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scene.tlas().as_binding(),
                },
            ],
        })
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Call after the G-buffer is recreated.
    pub fn resize(&mut self, d: &wgpu::Device, frame: &wgpu::Buffer, g: &GBuffer, scene: &RtScene) {
        self.view = target(d, g.extent);
        self.extent = g.extent;
        self.group = Self::group(d, &self.layout, frame, g, &self.view, scene);
    }

    /// After the G-buffer and the TLAS build.
    pub fn record(&self, enc: &mut wgpu::CommandEncoder) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rt lighting"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.extent.0.div_ceil(8), self.extent.1.div_ceil(8), 1);
    }
}
