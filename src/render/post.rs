//! Bloom, exposure and AgX tonemapping from the HDR target to the swapchain.

/// Exposure value for the scene at this sun elevation (lower = brighter image). The sky is
/// computed for unit solar illuminance, so the frame only brightens by about a stop and a half
/// from civil twilight to a high sun; past ~25° the exposure stays put.
pub fn exposure_ev(sun_elevation_deg: f32) -> f32 {
    let t = ((sun_elevation_deg + 6.0) / 30.0).clamp(0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    -3.5 + 1.5 * s
}

pub fn exposure_multiplier(ev: f32) -> f32 {
    (-ev).exp2()
}

use crate::gpu::extent::half_extent;
use crate::render::gbuffer::{GBuffer, HDR_FORMAT};

const BLOOM_LEVELS: u32 = 6;
/// Fraction of the final image taken from the (level-averaged) bloom chain.
const BLOOM_STRENGTH: f32 = 0.04;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostParams {
    exposure: f32,
    bloom_strength: f32,
    _pad: [f32; 2],
}

pub struct Post {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    finish: wgpu::RenderPipeline,
    chain: Chain,
}

/// Bloom mip views and the bind groups that read them; rebuilt when the G-buffer resizes.
struct Chain {
    mips: Vec<wgpu::TextureView>,
    down_groups: Vec<wgpu::BindGroup>,
    up_groups: Vec<wgpu::BindGroup>,
    final_group: wgpu::BindGroup,
}

impl Post {
    pub fn new(
        d: &wgpu::Device,
        lib: &crate::gpu::shaders::ShaderLibrary,
        output: wgpu::TextureFormat,
        g: &GBuffer,
    ) -> Post {
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[
                tex(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"),
            source: wgpu::ShaderSource::Wgsl(lib.source("post").expect("shader source").into()),
        });
        let pipeline = |label, fs, format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
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
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let down = pipeline("bloom down", "fs_down", HDR_FORMAT, None);
        let up = pipeline("bloom up", "fs_up", HDR_FORMAT, Some(additive));
        let finish = pipeline("tonemap", "fs_final", output, None);
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post linear clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post params"),
            size: std::mem::size_of::<PostParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chain = build_chain(d, &layout, &sampler, &params, g);
        Post {
            layout,
            sampler,
            params,
            down,
            up,
            finish,
            chain,
        }
    }

    /// Rebuilds the bloom chain for the G-buffer's current size.
    pub fn resize(&mut self, d: &wgpu::Device, g: &GBuffer) {
        self.chain = build_chain(d, &self.layout, &self.sampler, &self.params, g);
    }

    pub fn record(
        &self,
        enc: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        output: &wgpu::TextureView,
        exposure: f32,
    ) {
        let c = &self.chain;
        let p = PostParams {
            exposure,
            bloom_strength: BLOOM_STRENGTH / c.mips.len() as f32,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let pass = |enc: &mut wgpu::CommandEncoder,
                    target: &wgpu::TextureView,
                    pipe: &wgpu::RenderPipeline,
                    bg: &wgpu::BindGroup,
                    load| {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            rp.set_pipeline(pipe);
            rp.set_bind_group(0, bg, &[]);
            rp.draw(0..3, 0..1);
        };
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        for (m, bg) in c.down_groups.iter().enumerate() {
            pass(enc, &c.mips[m], &self.down, bg, clear);
        }
        for m in (0..c.up_groups.len()).rev() {
            pass(enc, &c.mips[m], &self.up, &c.up_groups[m], wgpu::LoadOp::Load);
        }
        pass(enc, output, &self.finish, &c.final_group, clear);
    }
}

fn build_chain(
    d: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    params: &wgpu::Buffer,
    g: &GBuffer,
) -> Chain {
    let (w, h) = half_extent(g.extent.0, g.extent.1);
    let levels = BLOOM_LEVELS.min(32 - w.min(h).leading_zeros());
    let tex = d.create_texture(&wgpu::TextureDescriptor {
        label: Some("bloom"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let mips: Vec<_> = (0..levels)
        .map(|m| {
            tex.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: m,
                mip_level_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    let group = |src: &wgpu::TextureView, extra: &wgpu::TextureView| {
        d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(extra),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    };
    // Downsample m reads HDR (m = 0) or mip m-1; upsample into mip m reads mip m+1.
    // Binding 2 is unused by those passes; HDR is bound there as it is never a target then.
    let down_groups = (0..mips.len())
        .map(|m| group(if m == 0 { &g.hdr } else { &mips[m - 1] }, &g.hdr))
        .collect();
    let up_groups = (0..mips.len() - 1).map(|m| group(&mips[m + 1], &g.hdr)).collect();
    let final_group = group(&g.hdr, &mips[0]);
    Chain {
        mips,
        down_groups,
        up_groups,
        final_group,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_is_monotonic_in_sun_elevation() {
        let mut prev = exposure_ev(-6.0);
        let mut el = -6.0;
        while el <= 90.0 {
            let ev = exposure_ev(el);
            assert!(ev.is_finite());
            assert!(ev >= prev - 1e-6, "ev dropped at {el}: {prev} -> {ev}");
            prev = ev;
            el += 0.5;
        }
    }

    #[test]
    fn daylight_is_not_underexposed() {
        // Measured: with the sky in units of solar illuminance the frame only brightens about
        // one stop from a 4° sunset to a high sun (mean luma 77 at EV −3.1 vs 78–98 at EV −2
        // for 25–90°). The old curve took 6 stops away and noon rendered black.
        assert!(
            (exposure_ev(4.0) + 3.1).abs() < 0.05,
            "sunset look changed: {}",
            exposure_ev(4.0)
        );
        for el in [25.0, 45.0, 60.0, 90.0] {
            let ev = exposure_ev(el);
            assert!((-2.3..=-1.7).contains(&ev), "EV {ev} at {el}°");
        }
    }

    #[test]
    fn darker_sky_gets_more_exposure() {
        // Lower EV means a larger multiplier; twilight must be brighter-exposed than noon.
        assert!(exposure_ev(-6.0) < exposure_ev(4.0));
        assert!(exposure_ev(4.0) < exposure_ev(60.0));
        assert!((exposure_multiplier(0.0) - 1.0).abs() < 1e-6);
    }
}
