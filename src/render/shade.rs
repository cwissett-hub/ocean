//! Full-screen composite of sky and water into the HDR target.

use crate::render::gbuffer::{GBuffer, HDR_FORMAT};
use crate::render::sky::{SkyLuts, with_sky};

pub const FRAME_COMMON: &str = include_str!("../../shaders/frame_common.wgsl");

pub struct ShadePass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
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
    pub fn new(d: &wgpu::Device, frame: &wgpu::Buffer, sky: &SkyLuts, g: &GBuffer) -> ShadePass {
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
            ],
        });
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shade"),
            bind_group_layouts: &[Some(&layout), Some(sky.layout())],
            immediate_size: 0,
        });
        let src = format!("{FRAME_COMMON}\n{}", with_sky(include_str!("../../shaders/shade.wgsl")));
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
        let bind_group = Self::bind(d, &layout, frame, g);
        ShadePass { pipeline, layout, bind_group }
    }

    fn bind(d: &wgpu::Device, layout: &wgpu::BindGroupLayout, frame: &wgpu::Buffer, g: &GBuffer) -> wgpu::BindGroup {
        use wgpu::BindingResource::TextureView as T;
        d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shade"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frame.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: T(&g.depth) },
                wgpu::BindGroupEntry { binding: 2, resource: T(&g.normal) },
                wgpu::BindGroupEntry { binding: 3, resource: T(&g.material) },
            ],
        })
    }

    /// Call after the G-buffer is recreated.
    pub fn rebind(&mut self, d: &wgpu::Device, frame: &wgpu::Buffer, g: &GBuffer) {
        self.bind_group = Self::bind(d, &self.layout, frame, g);
    }

    pub fn record(&self, enc: &mut wgpu::CommandEncoder, g: &GBuffer, sky: &SkyLuts) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shade"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &g.hdr,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_bind_group(1, sky.bind_group(), &[]);
        pass.draw(0..3, 0..1);
    }
}
