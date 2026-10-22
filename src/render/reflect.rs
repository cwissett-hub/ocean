//! Half-resolution reflections (screen-space now, ray traced in Task 9) with temporal
//! accumulation. Shading reads `ReflectionPass::views()[index()]`.

use crate::gpu::extent::half_extent;
use crate::gpu::shaders::ShaderLibrary;
use crate::params::ReflectionMode;
use crate::render::gbuffer::GBuffer;
use crate::render::rt_scene::RtScene;
use crate::render::sky::SkyLuts;

/// History weight for temporal accumulation: none without valid history, 0.9 for still
/// pixels, falling linearly to 0.5 at 4 px of motion. `reflect_resolve.wgsl` mirrors this.
pub fn temporal_blend(history_valid: bool, motion_px: f32) -> f32 {
    if !history_valid || !motion_px.is_finite() {
        return 0.0;
    }
    0.5 + 0.4 * (1.0 - (motion_px / 4.0).clamp(0.0, 1.0))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReflectParams {
    jitter: [u32; 2],
    history_valid: u32,
    patch_grid: u32,
}

/// 2x2 sample pattern cycled per frame, so accumulation covers every full-res pixel.
const JITTER: [[u32; 2]; 4] = [[0, 0], [1, 1], [1, 0], [0, 1]];

pub struct ReflectionPass {
    queue: wgpu::Queue,
    params: wgpu::Buffer,
    ssr: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    raw: wgpu::TextureView,
    history: [wgpu::TextureView; 2],
    ssr_group: wgpu::BindGroup,
    /// resolve_groups[i] reads history[i] and writes history[1 - i].
    resolve_groups: [wgpu::BindGroup; 2],
    current: usize,
    frame: u32,
    history_valid: bool,
    half: (u32, u32),
    rt: Option<RtPath>,
    patch_grid: u32,
}

fn storage_target(d: &wgpu::Device, label: &str, (w, h): (u32, u32)) -> wgpu::TextureView {
    d.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
    .create_view(&Default::default())
}

fn pipeline(d: &wgpu::Device, label: &str, src: &str) -> wgpu::ComputePipeline {
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

impl ReflectionPass {
    pub fn new(
        d: &wgpu::Device,
        lib: &ShaderLibrary,
        queue: &wgpu::Queue,
        frame: &wgpu::Buffer,
        g: &GBuffer,
    ) -> ReflectionPass {
        let ssr = pipeline(d, "reflect ssr", &lib.source("reflect_ssr").expect("shader source"));
        let resolve = pipeline(
            d,
            "reflect resolve",
            &lib.source("reflect_resolve").expect("shader source"),
        );
        let params = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reflect params"),
            size: std::mem::size_of::<ReflectParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (raw, history, ssr_group, resolve_groups, half) = Self::targets(d, &ssr, &resolve, &params, frame, g);
        ReflectionPass {
            queue: queue.clone(),
            params,
            ssr,
            resolve,
            raw,
            history,
            ssr_group,
            resolve_groups,
            current: 0,
            frame: 0,
            history_valid: false,
            half,
            rt: None,
            patch_grid: 0,
        }
    }

    #[allow(clippy::type_complexity)]
    fn targets(
        d: &wgpu::Device,
        ssr: &wgpu::ComputePipeline,
        resolve: &wgpu::ComputePipeline,
        params: &wgpu::Buffer,
        frame: &wgpu::Buffer,
        g: &GBuffer,
    ) -> (
        wgpu::TextureView,
        [wgpu::TextureView; 2],
        wgpu::BindGroup,
        [wgpu::BindGroup; 2],
        (u32, u32),
    ) {
        let half = half_extent(g.extent.0, g.extent.1);
        let raw = storage_target(d, "reflection raw", half);
        let history = [
            storage_target(d, "reflection a", half),
            storage_target(d, "reflection b", half),
        ];
        let lin = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("reflect linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        use wgpu::BindingResource as B;
        let group = |p: &wgpu::ComputePipeline, entries: Vec<(u32, B<'_>)>| {
            let e: Vec<_> = entries
                .into_iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource })
                .collect();
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reflect"),
                layout: &p.get_bind_group_layout(0),
                entries: &e,
            })
        };
        let ssr_group = group(
            ssr,
            vec![
                (0, frame.as_entire_binding()),
                (1, B::TextureView(&g.depth)),
                (2, B::TextureView(&g.normal)),
                (3, B::TextureView(&g.prev_hdr)),
                (4, B::Sampler(&lin)),
                (5, B::TextureView(&raw)),
                (6, params.as_entire_binding()),
            ],
        );
        let resolve_groups = std::array::from_fn(|i| {
            group(
                resolve,
                vec![
                    (0, frame.as_entire_binding()),
                    (1, B::TextureView(&raw)),
                    (2, B::TextureView(&history[i])),
                    (3, B::TextureView(&g.motion)),
                    (4, B::TextureView(&history[1 - i])),
                    (5, params.as_entire_binding()),
                ],
            )
        });
        (raw, history, ssr_group, resolve_groups, half)
    }

    /// Call after the G-buffer is recreated; history restarts.
    pub fn resize(&mut self, d: &wgpu::Device, frame: &wgpu::Buffer, g: &GBuffer, scene: Option<&RtScene>) {
        (self.raw, self.history, self.ssr_group, self.resolve_groups, self.half) =
            Self::targets(d, &self.ssr, &self.resolve, &self.params, frame, g);
        if let (Some(rt), Some(scene)) = (&mut self.rt, scene) {
            rt.group = rt_group(d, &rt.layout, frame, g, &self.raw, &self.params, scene);
        }
        self.history_valid = false;
        self.current = 0;
    }

    /// Adds the hardware ray-traced path (only on devices with ray query).
    pub fn enable_rt(
        &mut self,
        d: &wgpu::Device,
        lib: &ShaderLibrary,
        sky: &SkyLuts,
        frame: &wgpu::Buffer,
        g: &GBuffer,
        scene: &RtScene,
    ) {
        let cs = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cs,
            ty,
            count: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let tex = |sample_type| wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reflect rt"),
            entries: &[
                entry(0, uniform),
                entry(1, tex(wgpu::TextureSampleType::Depth)),
                entry(2, tex(wgpu::TextureSampleType::Float { filterable: false })),
                entry(
                    3,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                ),
                entry(4, uniform),
                entry(5, wgpu::BindingType::AccelerationStructure { vertex_return: false }),
                entry(
                    6,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(
                    7,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                ),
                entry(8, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(9, tex(wgpu::TextureSampleType::Float { filterable: false })),
            ],
        });
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reflect rt"),
            bind_group_layouts: &[Some(&layout), Some(sky.layout())],
            immediate_size: 0,
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reflect_rt"),
            source: wgpu::ShaderSource::Wgsl(lib.source("reflect_rt").expect("shader source").into()),
        });
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reflect rt"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = rt_group(d, &layout, frame, g, &self.raw, &self.params, scene);
        self.patch_grid = scene.config().grid;
        self.rt = Some(RtPath {
            pipeline,
            layout,
            group,
        });
    }

    pub fn rt_enabled(&self) -> bool {
        self.rt.is_some()
    }

    /// Records this frame's reflections: ray traced when asked and available, else
    /// screen-space; then the temporal resolve.
    pub fn record(&mut self, enc: &mut wgpu::CommandEncoder, mode: ReflectionMode, sky: &SkyLuts) {
        let p = ReflectParams {
            jitter: JITTER[(self.frame % 4) as usize],
            history_valid: self.history_valid as u32,
            patch_grid: self.patch_grid,
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let groups = (self.half.0.div_ceil(8), self.half.1.div_ceil(8));
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reflections"),
            timestamp_writes: None,
        });
        match (&self.rt, mode) {
            (Some(rt), ReflectionMode::Rt) => {
                pass.set_pipeline(&rt.pipeline);
                pass.set_bind_group(0, &rt.group, &[]);
                pass.set_bind_group(1, sky.bind_group(), &[]);
            }
            _ => {
                pass.set_pipeline(&self.ssr);
                pass.set_bind_group(0, &self.ssr_group, &[]);
            }
        }
        pass.dispatch_workgroups(groups.0, groups.1, 1);
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, &self.resolve_groups[self.current], &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
        self.current = 1 - self.current;
        self.frame = self.frame.wrapping_add(1);
        self.history_valid = true;
    }

    pub fn views(&self) -> &[wgpu::TextureView; 2] {
        &self.history
    }

    /// Index into `views()` of the latest resolved reflections.
    pub fn index(&self) -> usize {
        self.current
    }
}

struct RtPath {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    group: wgpu::BindGroup,
}

fn rt_group(
    d: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    frame: &wgpu::Buffer,
    g: &GBuffer,
    raw: &wgpu::TextureView,
    params: &wgpu::Buffer,
    scene: &RtScene,
) -> wgpu::BindGroup {
    use wgpu::BindingResource as B;
    d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reflect rt"),
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
                resource: B::TextureView(raw),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: scene.tlas().as_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: scene.patch_vertices().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: B::TextureView(scene.derivatives()),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: B::Sampler(scene.wave_sampler()),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: B::TextureView(scene.seabed()),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_history_is_ignored() {
        assert_eq!(temporal_blend(false, 0.0), 0.0);
        assert_eq!(temporal_blend(false, 10.0), 0.0);
    }

    #[test]
    fn still_pixels_keep_most_history() {
        assert!((temporal_blend(true, 0.0) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn fast_motion_keeps_half() {
        assert!((temporal_blend(true, 4.0) - 0.5).abs() < 1e-6);
        assert!((temporal_blend(true, 40.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn blend_falls_monotonically_with_motion() {
        let mut prev = temporal_blend(true, 0.0);
        for i in 1..100 {
            let b = temporal_blend(true, i as f32 * 0.1);
            assert!(b <= prev + 1e-6 && (0.5..=0.9).contains(&b));
            prev = b;
        }
    }

    #[test]
    fn non_finite_motion_drops_history() {
        assert_eq!(temporal_blend(true, f32::NAN), 0.0);
    }
}
