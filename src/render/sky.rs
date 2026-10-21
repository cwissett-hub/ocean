//! Physically based sky (Hillaire 2020): transmittance, multiple-scattering and sky-view LUTs
//! computed on the GPU whenever the sun moves. Every pass that needs sky light binds
//! `SkyLuts::bind_group`.

use glam::Vec3;

/// Unit vector toward the sun. y up; azimuth 0 = +z, 90 = +x (clockwise from above), matching
/// the camera's yaw.
pub fn sun_direction(elevation_deg: f32, azimuth_deg: f32) -> Vec3 {
    let (el, az) = (elevation_deg.to_radians(), azimuth_deg.to_radians());
    Vec3::new(az.sin() * el.cos(), el.sin(), az.cos() * el.cos())
}

pub const SKY_COMMON: &str = include_str!("../../shaders/sky_common.wgsl");
pub const SKY_SAMPLE: &str = include_str!("../../shaders/sky_sample.wgsl");

/// Source for a shader that samples the sky: shared constants + group-1 bindings + body.
pub fn with_sky(body: &str) -> String {
    format!("{SKY_COMMON}\n{SKY_SAMPLE}\n{body}")
}

const TRANSMITTANCE: (u32, u32) = (256, 64);
const MULTISCATTER: (u32, u32) = (32, 32);
const SKY_VIEW: (u32, u32) = (192, 108);

pub struct SkyLuts {
    queue: wgpu::Queue,
    params: wgpu::Buffer,
    sky_view_pass: (wgpu::ComputePipeline, wgpu::BindGroup),
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    last_sun: Option<Vec3>,
}

fn lut(d: &wgpu::Device, label: &str, (w, h): (u32, u32)) -> (wgpu::Texture, wgpu::TextureView) {
    let t = d.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let v = t.create_view(&Default::default());
    (t, v)
}

fn groups((w, h): (u32, u32)) -> (u32, u32) {
    (w.div_ceil(8), h.div_ceil(8))
}

impl SkyLuts {
    pub fn new(ctx: &crate::gpu::adapter::GpuContext) -> SkyLuts {
        let d = &ctx.device;
        let (_t, tv) = lut(d, "sky transmittance", TRANSMITTANCE);
        let (_m, mv) = lut(d, "sky multiscatter", MULTISCATTER);
        let (_s, sv) = lut(d, "sky view", SKY_VIEW);
        let clamp = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let wrap = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky wrap u"),
            address_mode_u: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky_luts"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{SKY_COMMON}\n{}", include_str!("../../shaders/sky_luts.wgsl")).into(),
            ),
        });
        use wgpu::BindingResource as B;
        let pass = |entry: &str, bindings: Vec<(u32, B<'_>)>| {
            let p = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            let entries: Vec<_> =
                bindings.into_iter().map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource }).collect();
            let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &p.get_bind_group_layout(0),
                entries: &entries,
            });
            (p, bg)
        };
        let trans_pass = pass("transmittance", vec![(4, B::TextureView(&tv))]);
        let ms_pass = pass(
            "multiscatter_lut",
            vec![(1, B::TextureView(&tv)), (3, B::Sampler(&clamp)), (4, B::TextureView(&mv))],
        );
        let sky_view_pass = pass(
            "sky_view",
            vec![
                (0, params.as_entire_binding()),
                (1, B::TextureView(&tv)),
                (2, B::TextureView(&mv)),
                (3, B::Sampler(&clamp)),
                (4, B::TextureView(&sv)),
            ],
        );
        // Transmittance and multiple scattering don't depend on the sun: compute them once.
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            for ((p, bg), size) in [(&trans_pass, TRANSMITTANCE), (&ms_pass, MULTISCATTER)] {
                cp.set_pipeline(p);
                cp.set_bind_group(0, bg, &[]);
                let (x, y) = groups(size);
                cp.dispatch_workgroups(x, y, 1);
            }
        }
        ctx.queue.submit([enc.finish()]);

        let frag_tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let samp = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky sample"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                frag_tex(1),
                frag_tex(2),
                samp(3),
                samp(4),
            ],
        });
        let bind_group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky sample"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: B::TextureView(&tv) },
                wgpu::BindGroupEntry { binding: 2, resource: B::TextureView(&sv) },
                wgpu::BindGroupEntry { binding: 3, resource: B::Sampler(&clamp) },
                wgpu::BindGroupEntry { binding: 4, resource: B::Sampler(&wrap) },
            ],
        });
        SkyLuts { queue: ctx.queue.clone(), params, sky_view_pass, layout, bind_group, last_sun: None }
    }

    /// Recomputes the sky-view LUT if the sun moved since the last call.
    pub fn update(&mut self, enc: &mut wgpu::CommandEncoder, sun_dir: Vec3) {
        if self.last_sun.is_some_and(|s| s.abs_diff_eq(sun_dir, 1e-6)) {
            return;
        }
        self.last_sun = Some(sun_dir);
        self.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&[sun_dir.x, sun_dir.y, sun_dir.z, 0.0]));
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&self.sky_view_pass.0);
        cp.set_bind_group(0, &self.sky_view_pass.1, &[]);
        let (x, y) = groups(SKY_VIEW);
        cp.dispatch_workgroups(x, y, 1);
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_on_horizon_at_azimuth_zero_points_along_z() {
        let s = sun_direction(0.0, 0.0);
        assert!((s - Vec3::Z).length() < 1e-6, "{s}");
    }

    #[test]
    fn sun_at_zenith_points_up() {
        for az in [0.0, 123.0, 300.0] {
            assert!((sun_direction(90.0, az).y - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn sun_below_horizon_points_down() {
        assert!(sun_direction(-6.0, 0.0).y < 0.0);
    }

    #[test]
    fn sun_azimuth_matches_camera_yaw_convention() {
        let s = sun_direction(0.0, 90.0);
        assert!((s - Vec3::X).length() < 1e-6, "{s}");
    }
}
