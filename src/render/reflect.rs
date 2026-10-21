//! Half-resolution reflections (screen-space now, ray traced in Task 9) with temporal
//! accumulation. Shading reads `ReflectionPass::views()[index()]`.

use crate::gpu::extent::half_extent;
use crate::params::ReflectionMode;
use crate::render::gbuffer::GBuffer;
use crate::render::shade::FRAME_COMMON;

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
    _pad: u32,
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
}

fn storage_target(d: &wgpu::Device, label: &str, (w, h): (u32, u32)) -> wgpu::TextureView {
    d.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
    .create_view(&Default::default())
}

fn pipeline(d: &wgpu::Device, label: &str, body: &str) -> wgpu::ComputePipeline {
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(format!("{FRAME_COMMON}\n{body}").into()),
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
    pub fn new(d: &wgpu::Device, queue: &wgpu::Queue, frame: &wgpu::Buffer, g: &GBuffer) -> ReflectionPass {
        let ssr = pipeline(d, "reflect ssr", include_str!("../../shaders/reflect_ssr.wgsl"));
        let resolve = pipeline(d, "reflect resolve", include_str!("../../shaders/reflect_resolve.wgsl"));
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
    ) -> (wgpu::TextureView, [wgpu::TextureView; 2], wgpu::BindGroup, [wgpu::BindGroup; 2], (u32, u32)) {
        let half = half_extent(g.extent.0, g.extent.1);
        let raw = storage_target(d, "reflection raw", half);
        let history = [storage_target(d, "reflection a", half), storage_target(d, "reflection b", half)];
        let lin = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("reflect linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        use wgpu::BindingResource as B;
        let group = |p: &wgpu::ComputePipeline, entries: Vec<(u32, B<'_>)>| {
            let e: Vec<_> = entries.into_iter().map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource }).collect();
            d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("reflect"), layout: &p.get_bind_group_layout(0), entries: &e })
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
    pub fn resize(&mut self, d: &wgpu::Device, frame: &wgpu::Buffer, g: &GBuffer) {
        (self.raw, self.history, self.ssr_group, self.resolve_groups, self.half) =
            Self::targets(d, &self.ssr, &self.resolve, &self.params, frame, g);
        self.history_valid = false;
        self.current = 0;
    }

    /// Records this frame's reflections. `mode` is honoured from Task 9; until then ray-traced
    /// mode renders screen-space reflections too.
    pub fn record(&mut self, enc: &mut wgpu::CommandEncoder, _mode: ReflectionMode) {
        let p = ReflectParams {
            jitter: JITTER[(self.frame % 4) as usize],
            history_valid: self.history_valid as u32,
            _pad: 0,
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let groups = (self.half.0.div_ceil(8), self.half.1.div_ceil(8));
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("reflections"), timestamp_writes: None });
        pass.set_pipeline(&self.ssr);
        pass.set_bind_group(0, &self.ssr_group, &[]);
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
