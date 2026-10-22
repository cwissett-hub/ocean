//! Temporal anti-aliasing: the projection is nudged by a sub-pixel Halton offset each frame, and
//! `taa.wgsl` blends each new frame into a reprojected, neighbourhood-clamped history, so edges,
//! crests and glints resolve over several frames instead of stair-stepping and popping.

use glam::{Mat4, Vec2};

/// Jitter sequence length (Halton 2, 3).
pub const JITTER_SAMPLES: u32 = 8;

fn halton(mut i: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

/// Sub-pixel offset for `frame`, in pixels, each component in (−0.5, 0.5).
pub fn jitter(frame: u32) -> Vec2 {
    let i = frame % JITTER_SAMPLES + 1;
    Vec2::new(halton(i, 2), halton(i, 3)) - 0.5
}

/// `proj` shifted so the image moves by `jitter_px` pixels (x right, y down) on a target of
/// `extent` pixels.
pub fn jittered(proj: Mat4, jitter_px: Vec2, extent: (u32, u32)) -> Mat4 {
    // A clip-space translation scaled by w is a constant NDC shift; NDC y grows up, pixels down.
    let ndc = Vec2::new(
        2.0 * jitter_px.x / extent.0 as f32,
        -2.0 * jitter_px.y / extent.1 as f32,
    );
    Mat4::from_translation(glam::Vec3::new(ndc.x, ndc.y, 0.0)) * proj
}

/// How much of the reprojected history to keep (mirrors `history_weight` in taa.wgsl): none
/// when it is invalid or the pixel came from off screen, less when the pixel moved fast.
pub fn history_weight(valid: bool, motion_px: f32, on_screen: bool) -> f32 {
    if !valid || !on_screen || !motion_px.is_finite() {
        return 0.0;
    }
    0.9 - 0.35 * (motion_px / 16.0).clamp(0.0, 1.0)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaParams {
    history_valid: u32,
    _pad: [u32; 3],
}

/// The resolve pass: two history targets ping-pong; the resolved frame is copied back into the
/// G-buffer's HDR target so everything after (history for SSR, bloom, tonemap) sees it.
pub struct TaaPass {
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    history: [wgpu::Texture; 2],
    groups: [wgpu::BindGroup; 2],
    current: usize,
    valid: bool,
}

fn history_texture(d: &wgpu::Device, extent: (u32, u32)) -> wgpu::Texture {
    d.create_texture(&wgpu::TextureDescriptor {
        label: Some("taa history"),
        size: wgpu::Extent3d {
            width: extent.0,
            height: extent.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::render::gbuffer::HDR_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

impl TaaPass {
    pub fn new(
        d: &wgpu::Device,
        lib: &crate::gpu::shaders::ShaderLibrary,
        frame: &wgpu::Buffer,
        g: &crate::render::gbuffer::GBuffer,
    ) -> TaaPass {
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa"),
            source: wgpu::ShaderSource::Wgsl(lib.source("taa").expect("shader source").into()),
        });
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("taa"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa params"),
            size: std::mem::size_of::<TaaParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("taa history"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let history = [history_texture(d, g.extent), history_texture(d, g.extent)];
        let groups = Self::groups(d, &pipeline, frame, g, &history, &params, &sampler);
        TaaPass {
            pipeline,
            params,
            sampler,
            history,
            groups,
            current: 0,
            valid: false,
        }
    }

    fn groups(
        d: &wgpu::Device,
        pipeline: &wgpu::ComputePipeline,
        frame: &wgpu::Buffer,
        g: &crate::render::gbuffer::GBuffer,
        history: &[wgpu::Texture; 2],
        params: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
    ) -> [wgpu::BindGroup; 2] {
        use wgpu::BindingResource as B;
        let views: Vec<wgpu::TextureView> = history.iter().map(|t| t.create_view(&Default::default())).collect();
        std::array::from_fn(|i| {
            let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("taa"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    entry(0, frame.as_entire_binding()),
                    entry(1, B::TextureView(&g.hdr)),
                    entry(2, B::TextureView(&views[1 - i])),
                    entry(3, B::TextureView(&g.motion)),
                    entry(4, B::TextureView(&g.depth)),
                    entry(5, B::TextureView(&views[i])),
                    entry(6, B::Sampler(sampler)),
                    entry(7, params.as_entire_binding()),
                ],
            })
        })
    }

    /// Call after the G-buffer is recreated.
    pub fn resize(&mut self, d: &wgpu::Device, frame: &wgpu::Buffer, g: &crate::render::gbuffer::GBuffer) {
        self.history = [history_texture(d, g.extent), history_texture(d, g.extent)];
        self.groups = Self::groups(d, &self.pipeline, frame, g, &self.history, &self.params, &self.sampler);
        self.valid = false;
    }

    /// The next frame starts a fresh history (camera cut, origin rebase, TAA switched on).
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Resolves this frame's HDR into the history and copies the result back into `g.hdr`.
    pub fn record(&mut self, enc: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, g: &crate::render::gbuffer::GBuffer) {
        let i = self.current;
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&TaaParams {
                history_valid: self.valid as u32,
                _pad: [0; 3],
            }),
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("taa"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.groups[i], &[]);
            pass.dispatch_workgroups(g.extent.0.div_ceil(8), g.extent.1.div_ceil(8), 1);
        }
        enc.copy_texture_to_texture(
            self.history[i].as_image_copy(),
            g.hdr_tex.as_image_copy(),
            wgpu::Extent3d {
                width: g.extent.0,
                height: g.extent.1,
                depth_or_array_layers: 1,
            },
        );
        self.current = 1 - i;
        self.valid = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn jitter_is_subpixel_centred_and_distinct() {
        let s: Vec<Vec2> = (0..JITTER_SAMPLES).map(jitter).collect();
        for (i, a) in s.iter().enumerate() {
            assert!(a.abs().max_element() < 0.5, "{a}");
            for b in &s[i + 1..] {
                assert!((*a - *b).length() > 0.05, "repeats {a}");
            }
        }
        let mean = s.iter().sum::<Vec2>() / s.len() as f32;
        assert!(mean.length() < 0.1, "biased {mean}");
        assert_eq!(jitter(3), jitter(3 + JITTER_SAMPLES), "repeats every {JITTER_SAMPLES}");
    }

    #[test]
    fn jittered_projection_moves_the_image_by_the_jitter() {
        let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 16.0 / 9.0, 0.1);
        let extent = (1600u32, 900u32);
        let to_px = |m: Mat4, p: Vec3| {
            let c = m.project_point3(p);
            Vec2::new((c.x * 0.5 + 0.5) * extent.0 as f32, (0.5 - c.y * 0.5) * extent.1 as f32)
        };
        let j = Vec2::new(0.3, -0.2);
        for p in [Vec3::new(0.0, 0.0, -5.0), Vec3::new(3.0, -1.0, -40.0)] {
            let d = to_px(jittered(proj, j, extent), p) - to_px(proj, p);
            assert!((d - j).length() < 1e-3, "moved {d}, wanted {j}");
        }
    }

    #[test]
    fn history_is_dropped_when_it_cannot_be_trusted() {
        assert_eq!(history_weight(false, 0.0, true), 0.0);
        assert_eq!(history_weight(true, 0.0, false), 0.0, "came from off screen");
        assert_eq!(history_weight(true, f32::NAN, true), 0.0);
        let still = history_weight(true, 0.0, true);
        assert!((0.85..0.95).contains(&still), "still {still}");
        assert!(history_weight(true, 20.0, true) < still, "fast motion keeps less");
        assert!(history_weight(true, 20.0, true) >= 0.5);
    }
}
