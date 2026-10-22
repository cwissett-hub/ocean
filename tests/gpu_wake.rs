//! The GPU wake step must match the CPU reference (render::wake::iwave_step). Headless.

use glam::Vec2;
use ocean::gpu::adapter::GpuContext;
use ocean::gpu::shaders::ShaderLibrary;
use ocean::render::wake::{WAKE_CELL_M, WAKE_RES, WakeSim, hull_pressure, iwave_kernel, iwave_step, window_origin};

fn read_heights(ctx: &GpuContext, sim: &WakeSim) -> Vec<f32> {
    let d = &ctx.device;
    let row = WAKE_RES as u32 * 8;
    let buf = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (row * WAKE_RES as u32) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = d.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        sim.texture().as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: WAKE_RES as u32,
            height: WAKE_RES as u32,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    d.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let rg: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&buf.slice(..).get_mapped_range().expect("mapped")[..]).to_vec();
    rg.chunks(2).map(|c| c[0]).collect()
}

#[test]
fn gpu_wake_matches_cpu() {
    let ctx = GpuContext::new_headless().expect("gpu");
    let lib = ShaderLibrary::embedded();
    let mut sim = WakeSim::new(&ctx.device, &ctx.queue, &lib);
    let (raft, heading) = (Vec2::new(3.1, -7.4), 0.5);
    let steps = 10;
    for _ in 0..steps {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        sim.record(&mut enc, &ctx.queue, raft, heading, 1.0 / 60.0);
        ctx.queue.submit([enc.finish()]);
    }
    let gpu = read_heights(&ctx, &sim);

    let n = WAKE_RES as usize;
    let origin = window_origin(raft);
    let mut src = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            let cell = origin + glam::IVec2::new(x as i32, y as i32);
            src[y * n + x] = hull_pressure((cell.as_vec2() + 0.5) * WAKE_CELL_M, raft, heading);
        }
    }
    let (mut h, mut prev) = (vec![0.0; n * n], vec![0.0; n * n]);
    let k = iwave_kernel();
    for _ in 0..steps {
        iwave_step(&mut h, &mut prev, &src, &k, n, 1.0 / 60.0);
    }
    let mut worst = 0.0f32;
    let mut deepest = 0.0f32;
    for y in 0..n {
        for x in 0..n {
            let cell = origin + glam::IVec2::new(x as i32, y as i32);
            let t = cell.rem_euclid(glam::IVec2::splat(WAKE_RES));
            let g = gpu[t.y as usize * n + t.x as usize];
            worst = worst.max((g - h[y * n + x]).abs());
            deepest = deepest.min(g);
        }
    }
    assert!(deepest < -0.01, "the hull pushed nothing down ({deepest})");
    assert!(worst < 1e-4, "gpu vs cpu worst {worst}");
}
