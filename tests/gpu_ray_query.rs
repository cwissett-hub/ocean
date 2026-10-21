//! Headless proof that hardware ray queries work through wgpu on this machine. No window.

use ocean::gpu::adapter::GpuContext;
use wgpu::util::DeviceExt;

fn read_back(ctx: &GpuContext, src: &wgpu::Buffer, size: u64) -> Vec<f32> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    ctx.queue.submit([enc.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec()
}

#[test]
fn ray_hits_single_triangle() {
    let ctx = GpuContext::new_headless().expect("gpu");
    if !ctx.rt_available {
        eprintln!("skipping: no ray query on {}", ctx.adapter_name);
        return;
    }
    let d = &ctx.device;
    let verts: [f32; 9] = [-1.0, -1.0, 5.0, 1.0, -1.0, 5.0, 0.0, 1.0, 5.0];
    let vbuf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tri"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = d.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("tri blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
    );
    let mut tlas = d.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("tlas"),
        max_instances: 1,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    tlas[0] = Some(wgpu::TlasInstance::new(&blas, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0, 0xFF));

    let out = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rq_smoke"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rq_smoke.wgsl").into()),
    });
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rq_smoke"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tlas.as_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
        ],
    });
    let mut enc = d.create_command_encoder(&Default::default());
    enc.build_acceleration_structures(
        std::iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &vbuf,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        std::iter::once(&tlas),
    );
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    ctx.queue.submit([enc.finish()]);

    let r = read_back(&ctx, &out, 32);
    assert_eq!(r[0], 1.0, "forward ray should hit a triangle, got kind {}", r[0]);
    assert!((r[1] - 5.0).abs() < 1e-4, "t = {}", r[1]);
    assert_eq!(r[4], 0.0, "backward ray should miss, got kind {}", r[4]);
}
