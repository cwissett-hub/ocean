//! The shaders' seabed lookup must agree with the CPU height functions the raft physics uses.
//! Headless, no window.

use glam::{DVec2, IVec2, Vec2};
use ocean::gpu::adapter::GpuContext;
use ocean::gpu::shaders::ShaderLibrary;
use ocean::render::seabed::{SEABED_RES, SEABED_TEXEL_M, SeabedMap, map_corner};
use ocean::world::islands::{IslandDesc, WORLD_SEED, islands_in_cell, seabed_height};
use ocean::world::origin::WorldOrigin;

const KERNEL: &str = r#"
@group(0) @binding(0) var seabed: texture_2d<f32>;
@group(0) @binding(1) var<uniform> sp: vec4<f32>;
@group(0) @binding(2) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> heights: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&points)) {
        return;
    }
    heights[id.x] = seabed_height(seabed, sp, points[id.x]);
}
"#;

/// Samples the map at local points on the GPU.
fn gpu_heights(ctx: &GpuContext, map: &SeabedMap, sp: [f32; 4], points: &[Vec2]) -> Vec<f32> {
    let d = &ctx.device;
    let lib = ShaderLibrary::embedded();
    let src = format!("{}\n{KERNEL}", lib.source("seabed_common").expect("seabed_common"));
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("seabed probe"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("seabed probe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    use wgpu::BufferUsages as U;
    use wgpu::util::DeviceExt;
    let uniform = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&sp),
        usage: U::UNIFORM,
    });
    let pts = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&points.iter().map(|p| p.to_array()).collect::<Vec<_>>()),
        usage: U::STORAGE,
    });
    let size = (points.len() * 4) as u64;
    let out = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: U::STORAGE | U::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: U::MAP_READ | U::COPY_DST,
        mapped_at_creation: false,
    });
    let group = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(map.view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(points.len().div_ceil(64) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &staging, 0, size);
    ctx.queue.submit([enc.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    d.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let v: Vec<f32> =
        bytemuck::cast_slice::<u8, f32>(&staging.slice(..).get_mapped_range().expect("mapped")[..]).to_vec();
    v
}

/// The first archipelago in the world, plus the start island.
fn island_set() -> Vec<IslandDesc> {
    for r in 1..20 {
        for y in -r..=r {
            for x in -r..=r {
                let v = islands_in_cell(WORLD_SEED, IVec2::new(x, y));
                if v.len() > 1 {
                    return v;
                }
            }
        }
    }
    panic!("no archipelago near the start");
}

#[test]
fn gpu_seabed_matches_cpu() {
    let ctx = GpuContext::new_headless().expect("gpu");
    let islands = island_set();
    let focus = islands[0].center;
    // A rebased origin, so the local/world split is exercised too.
    let origin = WorldOrigin {
        offset: (focus / 1024.0).round() * 1024.0,
    };
    let mut map = SeabedMap::new(&ctx.device);
    map.update(&ctx.queue, WORLD_SEED, focus, &islands);
    let corner = map_corner(focus);

    // Texel centres around every island (beach, shallows, dunes) and across the open floor.
    let mut world: Vec<DVec2> = Vec::new();
    for (k, d) in islands.iter().enumerate() {
        for s in 0..30 {
            let a = s as f64 * 2.4 + k as f64;
            let r = d.radius as f64 * (0.1 + 0.1 * s as f64);
            world.push(d.center + DVec2::new(a.cos(), a.sin()) * r);
        }
    }
    while world.len() < 200 {
        let s = world.len() as f64;
        world.push(focus + DVec2::new((s * 37.0) % 1800.0 - 900.0, (s * 53.0) % 1800.0 - 900.0));
    }
    let snap = |p: DVec2| corner + (((p - corner) / SEABED_TEXEL_M).floor() + 0.5) * SEABED_TEXEL_M;
    let world: Vec<DVec2> = world.into_iter().map(snap).collect();
    let local: Vec<Vec2> = world.iter().map(|w| origin.to_local(*w)).collect();
    let got = gpu_heights(&ctx, &map, map.params(&origin), &local);
    let mut worst = 0.0f32;
    let mut land = 0;
    for (w, g) in world.iter().zip(&got) {
        let want = seabed_height(WORLD_SEED, &islands, *w);
        land += (want > 0.0) as usize;
        worst = worst.max((g - want).abs());
    }
    assert!(land >= 20, "only {land} probes on dry land");
    assert!(worst < 0.05, "worst seabed error {worst} m");

    // Off the map, and before any map exists, the shaders see the open-sea depth.
    let far = origin.to_local(corner + DVec2::splat(-500.0));
    assert_eq!(gpu_heights(&ctx, &map, map.params(&origin), &[far]), vec![-25.0]);
    assert_eq!(gpu_heights(&ctx, &map, [0.0; 4], &[local[0]]), vec![-25.0]);
    assert_eq!(SEABED_RES, 1024);
}
