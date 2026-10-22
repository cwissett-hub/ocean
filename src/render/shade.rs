//! Full-screen composite of sky and water into the HDR target.

use crate::gpu::shaders::ShaderLibrary;
use crate::render::gbuffer::{GBuffer, HDR_FORMAT};
use crate::render::sky::SkyLuts;

pub struct ShadePass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    seabed: wgpu::TextureView,
    /// One per reflection history side (see ReflectionPass::index).
    bind_groups: [wgpu::BindGroup; 2],
}

fn tex_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl ShadePass {
    pub fn new(
        d: &wgpu::Device,
        lib: &ShaderLibrary,
        frame: &wgpu::Buffer,
        sky: &SkyLuts,
        g: &GBuffer,
        refl: &[wgpu::TextureView; 2],
        seabed: &wgpu::TextureView,
    ) -> ShadePass {
        let float = wgpu::TextureSampleType::Float { filterable: false };
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shade"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(1, wgpu::TextureSampleType::Depth),
                tex_entry(2, float),
                tex_entry(3, float),
                tex_entry(4, wgpu::TextureSampleType::Float { filterable: true }),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex_entry(6, wgpu::TextureSampleType::Depth),
                tex_entry(7, float),
                tex_entry(8, float),
            ],
        });
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shade"),
            bind_group_layouts: &[Some(&layout), Some(sky.layout())],
            immediate_size: 0,
        });
        let src = lib.source("shade").expect("shader source");
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shade"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shade"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_full"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_shade"),
                compilation_options: Default::default(),
                targets: &[Some(HDR_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shade linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_groups = Self::bind(d, &layout, frame, g, refl, &sampler, seabed);
        ShadePass {
            pipeline,
            layout,
            sampler,
            seabed: seabed.clone(),
            bind_groups,
        }
    }

    fn bind(
        d: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        frame: &wgpu::Buffer,
        g: &GBuffer,
        refl: &[wgpu::TextureView; 2],
        sampler: &wgpu::Sampler,
        seabed: &wgpu::TextureView,
    ) -> [wgpu::BindGroup; 2] {
        use wgpu::BindingResource::TextureView as T;
        std::array::from_fn(|i| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shade"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: T(&g.depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: T(&g.normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: T(&g.material),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: T(&refl[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: T(&g.opaque_depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: T(&g.opaque_albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: T(seabed),
                    },
                ],
            })
        })
    }

    /// Call after the G-buffer is recreated.
    pub fn rebind(&mut self, d: &wgpu::Device, frame: &wgpu::Buffer, g: &GBuffer, refl: &[wgpu::TextureView; 2]) {
        self.bind_groups = Self::bind(d, &self.layout, frame, g, refl, &self.sampler, &self.seabed);
    }

    pub fn record(&self, enc: &mut wgpu::CommandEncoder, g: &GBuffer, sky: &SkyLuts, refl_index: usize) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shade"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &g.hdr,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_groups[refl_index], &[]);
        pass.set_bind_group(1, sky.bind_group(), &[]);
        pass.draw(0..3, 0..1);
    }
}
