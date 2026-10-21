//! The per-frame wave simulation on the GPU: evolve h0 → inverse FFT → unpack into the
//! displacement and derivative textures that every later pass samples.

use crate::gpu::adapter::GpuContext;
use crate::gpu::shaders::ShaderLibrary;
use crate::waves::foam::FoamParams;
use crate::waves::spectrum::{CASCADE_SIZES_M, N, SpectrumSet};

pub const CASCADES: u32 = 4;
const WG: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    time: f32,
    choppiness: f32,
    dt: f32,
    threshold: f32,
    inject: f32,
    decay_per_s: f32,
    _pad: [f32; 2],
    sizes: [f32; 4],
}

/// Foam texture: cascade 0's period at 0.5 m per texel.
pub const FOAM_RES: u32 = 2048;
/// Longest step foam integrates in one frame; a hitch must not flood the sea with foam.
const MAX_FOAM_DT: f32 = 0.1;

pub struct GpuOcean {
    queue: wgpu::Queue,
    h0: wgpu::Texture,
    omega: wgpu::Texture,
    params: wgpu::Buffer,
    /// (Dx, height, Dz, ∂Dx/∂z) per cascade layer, Rgba16Float, repeat-sampled.
    pub displacement: wgpu::Texture,
    /// (∂h/∂x, ∂h/∂z, ∂Dx/∂x, ∂Dz/∂z) per cascade layer.
    pub derivatives: wgpu::Texture,
    pub displacement_view: wgpu::TextureView,
    pub derivatives_view: wgpu::TextureView,
    update: (wgpu::ComputePipeline, wgpu::BindGroup),
    fft_h: (wgpu::ComputePipeline, wgpu::BindGroup),
    fft_v: (wgpu::ComputePipeline, wgpu::BindGroup),
    assemble: (wgpu::ComputePipeline, wgpu::BindGroup),
    /// Spray deposits (512² u32, 1/256 units per 2 m cell) written by particles, read by foam.
    deposits: wgpu::Buffer,
    /// Foam ping-pong: pass i reads foam_views[i] and writes foam_views[1 - i].
    foam_passes: [(wgpu::ComputePipeline, wgpu::BindGroup); 2],
    foam_views: [wgpu::TextureView; 2],
    /// Index of the foam view holding the latest result.
    foam_current: usize,
}

fn array_texture(d: &wgpu::Device, label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
    d.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: N as u32, height: N as u32, depth_or_array_layers: CASCADES },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn array_view(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

fn compute(
    d: &wgpu::Device,
    label: &str,
    src: &str,
    constants: &[(&str, f64)],
    resources: &[wgpu::BindingResource<'_>],
) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
        cache: None,
    });
    let entries: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r.clone() })
        .collect();
    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    (pipeline, bg)
}

impl GpuOcean {
    pub fn new(ctx: &GpuContext, lib: &ShaderLibrary, spectrum: &SpectrumSet) -> GpuOcean {
        let src = |n: &str| lib.source(n).expect("shader source");
        let d = &ctx.device;
        use wgpu::TextureFormat as F;
        use wgpu::TextureUsages as U;
        let h0 = array_texture(d, "h0", F::Rgba32Float, U::TEXTURE_BINDING | U::COPY_DST);
        let omega = array_texture(d, "omega", F::R32Float, U::TEXTURE_BINDING | U::COPY_DST);
        let work = U::TEXTURE_BINDING | U::STORAGE_BINDING;
        let ping: Vec<_> = ["fft a0", "fft a1", "fft b0", "fft b1"]
            .iter()
            .map(|l| array_texture(d, l, F::Rgba32Float, work))
            .collect();
        let out = work | U::COPY_SRC;
        let displacement = array_texture(d, "displacement", F::Rgba16Float, out);
        let derivatives = array_texture(d, "derivatives", F::Rgba16Float, out);
        let params = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ocean params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let v: Vec<_> = ping.iter().map(array_view).collect();
        let (h0v, omv) = (array_view(&h0), array_view(&omega));
        let (dispv, derv) = (array_view(&displacement), array_view(&derivatives));
        use wgpu::BindingResource::TextureView as T;
        let update = compute(
            d,
            "spectrum_update",
            &src("spectrum_update"),
            &[],
            &[T(&h0v), T(&omv), T(&v[0]), T(&v[1]), params.as_entire_binding()],
        );
        let fft_src = &src("fft");
        let fft_h = compute(d, "fft rows", fft_src, &[("HORIZONTAL", 1.0)], &[T(&v[0]), T(&v[1]), T(&v[2]), T(&v[3])]);
        let fft_v = compute(d, "fft cols", fft_src, &[("HORIZONTAL", 0.0)], &[T(&v[2]), T(&v[3]), T(&v[0]), T(&v[1])]);
        let assemble = compute(
            d,
            "assemble",
            &src("assemble"),
            &[],
            &[T(&v[0]), T(&v[1]), T(&dispv), T(&derv)],
        );
        let foam_tex = |label| {
            d.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: FOAM_RES, height: FOAM_RES, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: F::Rgba16Float,
                usage: U::TEXTURE_BINDING | U::STORAGE_BINDING | U::COPY_SRC,
                view_formats: &[],
            })
            .create_view(&Default::default())
        };
        let foam_views = [foam_tex("foam a"), foam_tex("foam b")];
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ocean repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let deposits = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spray deposits"),
            size: 512 * 512 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let foam_src = &src("foam");
        let foam_pass = |i: usize| {
            compute(
                d,
                "foam",
                foam_src,
                &[],
                &[
                    T(&dispv),
                    T(&derv),
                    wgpu::BindingResource::Sampler(&sampler),
                    T(&foam_views[i]),
                    T(&foam_views[1 - i]),
                    params.as_entire_binding(),
                    deposits.as_entire_binding(),
                ],
            )
        };
        let foam_passes = [foam_pass(0), foam_pass(1)];
        let ocean = GpuOcean {
            deposits,
            foam_passes,
            foam_views,
            foam_current: 0,
            queue: ctx.queue.clone(),
            h0,
            omega,
            params,
            displacement,
            derivatives,
            displacement_view: dispv,
            derivatives_view: derv,
            update,
            fft_h,
            fft_v,
            assemble,
        };
        ocean.set_spectrum(&ctx.queue, spectrum);
        ocean
    }

    /// Uploads new initial amplitudes (after a wind / fetch / swell change).
    pub fn set_spectrum(&self, queue: &wgpu::Queue, spectrum: &SpectrumSet) {
        for c in 0..CASCADES as usize {
            let origin = wgpu::Origin3d { x: 0, y: 0, z: c as u32 };
            let size = wgpu::Extent3d { width: N as u32, height: N as u32, depth_or_array_layers: 1 };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &self.h0, mip_level: 0, origin, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&spectrum.h0[c]),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(N as u32 * 16), rows_per_image: None },
                size,
            );
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &self.omega, mip_level: 0, origin, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&spectrum.omega[c]),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(N as u32 * 4), rows_per_image: None },
                size,
            );
        }
    }

    /// Records the simulation for time `t` (already reduced modulo the repeat period).
    pub fn update(&mut self, enc: &mut wgpu::CommandEncoder, t: f32, choppiness: f32, foam: FoamParams, dt: f32) {
        // Queue writes land before the next submit, so one update per submit.
        let p = Params {
            time: t,
            choppiness,
            dt: dt.clamp(0.0, MAX_FOAM_DT),
            threshold: foam.threshold,
            inject: foam.inject,
            decay_per_s: foam.decay_per_s,
            _pad: [0.0; 2],
            sizes: CASCADE_SIZES_M,
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let n = N as u32;
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ocean sim"), timestamp_writes: None });
        for ((pipeline, bg), groups) in [
            (&self.update, (n / WG, n / WG, CASCADES)),
            (&self.fft_h, (1, n, CASCADES)),
            (&self.fft_v, (1, n, CASCADES)),
            (&self.assemble, (n / WG, n / WG, CASCADES)),
        ] {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        let (pipeline, bg) = &self.foam_passes[self.foam_current];
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(FOAM_RES / WG, FOAM_RES / WG, 1);
        self.foam_current = 1 - self.foam_current;
    }

    /// Both foam textures; `foam_index` says which holds the latest result.
    pub fn foam_views(&self) -> &[wgpu::TextureView; 2] {
        &self.foam_views
    }

    pub fn foam_index(&self) -> usize {
        self.foam_current
    }

    pub fn deposits(&self) -> &wgpu::Buffer {
        &self.deposits
    }

    /// Blocking readback of the latest foam amounts (FOAM_RES², row-major). Tests only.
    pub fn read_foam(&self, ctx: &GpuContext) -> Vec<f32> {
        read_rgba16(ctx, self.foam_views[self.foam_current].texture(), 0, FOAM_RES).into_iter().map(|t| t[0]).collect()
    }

    /// Blocking readback of one cascade's displacement (Dx, height, Dz, ∂Dx/∂z). Tests only.
    pub fn read_displacement(&self, ctx: &GpuContext, cascade: u32) -> Vec<[f32; 4]> {
        read_rgba16(ctx, &self.displacement, cascade, N as u32)
    }

    /// Blocking readback of one cascade's derivatives (∂h/∂x, ∂h/∂z, ∂Dx/∂x, ∂Dz/∂z). Tests only.
    pub fn read_derivatives(&self, ctx: &GpuContext, cascade: u32) -> Vec<[f32; 4]> {
        read_rgba16(ctx, &self.derivatives, cascade, N as u32)
    }
}

fn read_rgba16(ctx: &GpuContext, tex: &wgpu::Texture, layer: u32, size: u32) -> Vec<[f32; 4]> {
    let row = size * 8;
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * size) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
        },
        wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map readback"));
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = buf.slice(..).get_mapped_range().expect("mapped range");
    let halves: &[u16] = bytemuck::cast_slice(&data);
    halves
        .chunks_exact(4)
        .map(|c| [f16_to_f32(c[0]), f16_to_f32(c[1]), f16_to_f32(c[2]), f16_to_f32(c[3])])
        .collect()
}

/// IEEE 754 binary16 → f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: renormalise.
            let shift = man.leading_zeros() - 21;
            sign | ((127 - 15 - shift + 1) << 23) | (((man << shift) & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn f16_decodes_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8); // smallest subnormal
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
    }
}
