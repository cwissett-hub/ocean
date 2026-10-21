//! The ray-tracing scene: a near-field patch of the displaced waves (rebuilt or refitted every
//! frame) and the buoy, in one TLAS.

use glam::{Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::gpu::adapter::GpuContext;
use crate::gpu::shaders::ShaderLibrary;
use crate::waves::gpu_ocean::GpuOcean;
use crate::waves::spectrum::{CASCADE_SIZES_M, cascade_band};

/// TLAS instance masks / custom data.
pub const PATCH_ID: u32 = 0;
pub const BUOY_ID: u32 = 1;
/// Full BLAS rebuild cadence (refits in between slowly degrade the BVH).
const REBUILD_EVERY: u32 = 60;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtPatchConfig {
    pub size_m: f32,
    pub grid: u32,
}

impl Default for RtPatchConfig {
    fn default() -> Self {
        RtPatchConfig {
            size_m: 512.0,
            grid: 512,
        }
    }
}

impl RtPatchConfig {
    pub fn cell(&self) -> f32 {
        self.size_m / self.grid as f32
    }
}

/// Min corner of the patch: centred ahead of the camera along its horizontal forward, leaving
/// 16 m behind the eye, snapped to whole cells so the patch never swims.
pub fn patch_origin(cam_pos: Vec3, cam_fwd: Vec3, cfg: &RtPatchConfig) -> Vec2 {
    let f = Vec2::new(cam_fwd.x, cam_fwd.z).normalize_or(Vec2::Y);
    let centre = Vec2::new(cam_pos.x, cam_pos.z) + f * (cfg.size_m / 2.0 - 16.0);
    let corner = centre - Vec2::splat(cfg.size_m / 2.0);
    (corner / cfg.cell()).round() * cfg.cell()
}

/// Triangle list for the (grid+1)² patch vertices; triangle 2q and 2q+1 make quad q.
pub fn patch_indices(grid: u32) -> Vec<u32> {
    let v = grid + 1;
    let mut out = Vec::with_capacity((grid * grid * 6) as usize);
    for j in 0..grid {
        for i in 0..grid {
            let a = j * v + i;
            out.extend_from_slice(&[a, a + 1, a + v + 1, a, a + v + 1, a + v]);
        }
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PatchParams {
    origin: [f32; 2],
    cell: f32,
    grid: u32,
    sizes: [f32; 4],
    lambda_min: [f32; 4],
}

pub struct RtScene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cfg: RtPatchConfig,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    params: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    patch_size: wgpu::BlasTriangleGeometrySizeDescriptor,
    patch_blas: wgpu::Blas,
    buoy_size: wgpu::BlasTriangleGeometrySizeDescriptor,
    buoy_blas: wgpu::Blas,
    buoy_built: bool,
    tlas: wgpu::Tlas,
    since_rebuild: u32,
    last_origin: Option<Vec2>,
    derivatives: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

fn patch_blas(d: &wgpu::Device, size: &wgpu::BlasTriangleGeometrySizeDescriptor) -> wgpu::Blas {
    d.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("wave patch blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD | wgpu::AccelerationStructureFlags::ALLOW_UPDATE,
            update_mode: wgpu::AccelerationStructureUpdateMode::PreferUpdate,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size.clone()],
        },
    )
}

/// glam's column-major 4x4 → the TLAS's row-major 3x4.
fn instance_transform(m: Mat4) -> [f32; 12] {
    let t = m.transpose().to_cols_array();
    std::array::from_fn(|i| t[i])
}

impl RtScene {
    pub fn new(
        ctx: &GpuContext,
        lib: &ShaderLibrary,
        ocean: &GpuOcean,
        buoy: (&wgpu::Buffer, &wgpu::Buffer, u32),
        buoy_vertices: u32,
        cfg: RtPatchConfig,
    ) -> RtScene {
        let d = &ctx.device;
        let n_verts = (cfg.grid + 1) * (cfg.grid + 1);
        let vertices = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave patch vertices"),
            size: (n_verts * 12) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        });
        let idx = patch_indices(cfg.grid);
        let indices = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wave patch indices"),
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let params = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave patch params"),
            size: std::mem::size_of::<PatchParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rt_patch"),
            source: wgpu::ShaderSource::Wgsl(lib.source("rt_patch").expect("shader source").into()),
        });
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rt_patch"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rt patch repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_patch"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ocean.displacement_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let patch_size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: n_verts,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(idx.len() as u32),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let buoy_size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: buoy_vertices,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(buoy.2),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let buoy_blas = d.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("buoy blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![buoy_size.clone()],
            },
        );
        let tlas = d.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("scene tlas"),
            max_instances: 2,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        RtScene {
            device: d.clone(),
            queue: ctx.queue.clone(),
            patch_blas: patch_blas(d, &patch_size),
            cfg,
            vertices,
            indices,
            params,
            pipeline,
            bind_group,
            patch_size,
            buoy_size,
            buoy_blas,
            buoy_built: false,
            tlas,
            since_rebuild: 0,
            last_origin: None,
            derivatives: ocean.derivatives_view.clone(),
            sampler,
        }
    }

    pub fn config(&self) -> RtPatchConfig {
        self.cfg
    }

    /// World positions of the patch vertices (3 floats each), for hit normals.
    pub fn patch_vertices(&self) -> &wgpu::Buffer {
        &self.vertices
    }

    pub fn tlas(&self) -> &wgpu::Tlas {
        &self.tlas
    }

    /// The wave derivative textures (for smooth normals at ray hits) and their sampler.
    pub fn derivatives(&self) -> &wgpu::TextureView {
        &self.derivatives
    }

    pub fn wave_sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Writes this frame's patch at `origin`, refits/rebuilds its BLAS and rebuilds the TLAS.
    pub fn record(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        origin: Vec2,
        buoy: Mat4,
        buoy_geometry: (&wgpu::Buffer, &wgpu::Buffer),
    ) {
        let p = PatchParams {
            origin: origin.to_array(),
            cell: self.cfg.cell(),
            grid: self.cfg.grid,
            sizes: CASCADE_SIZES_M,
            lambda_min: std::array::from_fn(|c| std::f32::consts::TAU / cascade_band(c).1),
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rt patch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let g = (self.cfg.grid + 1).div_ceil(16);
            pass.dispatch_workgroups(g, g, 1);
        }
        // A moved patch or a long run of refits gets a fresh BLAS (a full build).
        self.since_rebuild += 1;
        if self.last_origin != Some(origin) || self.since_rebuild >= REBUILD_EVERY {
            self.patch_blas = patch_blas(&self.device, &self.patch_size);
            self.since_rebuild = 0;
            self.last_origin = Some(origin);
        }
        let patch_geo = wgpu::BlasTriangleGeometry {
            size: &self.patch_size,
            vertex_buffer: &self.vertices,
            first_vertex: 0,
            vertex_stride: 12,
            index_buffer: Some(&self.indices),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        };
        let buoy_geo = wgpu::BlasTriangleGeometry {
            size: &self.buoy_size,
            vertex_buffer: buoy_geometry.0,
            first_vertex: 0,
            vertex_stride: 24,
            index_buffer: Some(buoy_geometry.1),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        };
        let identity = instance_transform(Mat4::IDENTITY);
        self.tlas[0] = Some(wgpu::TlasInstance::new(&self.patch_blas, identity, PATCH_ID, 0x01));
        self.tlas[1] = Some(wgpu::TlasInstance::new(
            &self.buoy_blas,
            instance_transform(buoy),
            BUOY_ID,
            0x02,
        ));
        let mut entries = vec![wgpu::BlasBuildEntry {
            blas: &self.patch_blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![patch_geo]),
        }];
        if !self.buoy_built {
            entries.push(wgpu::BlasBuildEntry {
                blas: &self.buoy_blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![buoy_geo]),
            });
            self.buoy_built = true;
        }
        enc.build_acceleration_structures(entries.iter(), std::iter::once(&self.tlas));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};

    fn fwd(yaw_deg: f32) -> Vec3 {
        let y = yaw_deg.to_radians();
        Vec3::new(y.sin(), -0.4, y.cos()).normalize()
    }

    #[test]
    fn origin_snaps_to_cells() {
        let cfg = RtPatchConfig::default();
        for (cam, yaw) in [
            (Vec3::new(0.0, 18.0, 0.0), 30.0),
            (Vec3::new(123.4, 18.0, -56.7), 200.0),
        ] {
            let o = patch_origin(cam, fwd(yaw), &cfg) / cfg.cell();
            assert!((o - o.round()).abs().max_element() < 1e-3, "{o}");
        }
    }

    #[test]
    fn patch_lies_ahead_and_covers_the_near_water() {
        let cfg = RtPatchConfig::default();
        for yaw in [0.0, 30.0, 135.0, 290.0] {
            let cam = Vec3::new(5.0, 18.0, -3.0);
            let f = fwd(yaw);
            let o = patch_origin(cam, f, &cfg);
            let centre = o + Vec2::splat(cfg.size_m / 2.0);
            let f2 = Vec2::new(f.x, f.z).normalize();
            assert!((centre - Vec2::new(cam.x, cam.z)).dot(f2) > 0.0, "yaw {yaw}");
            // The nearest visible water (~13 m ahead) and the buoy (35 m) are inside.
            for d in [13.0, 35.0, 100.0] {
                let p = Vec2::new(cam.x, cam.z) + f2 * d;
                assert!(
                    p.x > o.x && p.y > o.y && p.x < o.x + cfg.size_m && p.y < o.y + cfg.size_m,
                    "yaw {yaw} d {d}"
                );
            }
        }
    }

    #[test]
    fn sub_cell_camera_motion_moves_the_patch_by_at_most_one_cell() {
        let cfg = RtPatchConfig::default();
        let base = patch_origin(Vec3::new(10.0, 18.0, 10.0), fwd(30.0), &cfg);
        let moved = patch_origin(Vec3::new(10.0 + 0.3 * cfg.cell(), 18.0, 10.0), fwd(30.0), &cfg);
        let d = ((moved - base) / cfg.cell()).abs();
        assert!(d.max_element() <= 1.0 + 1e-3, "{d}");
    }

    #[test]
    fn patch_indices_cover_the_grid() {
        let cfg = RtPatchConfig::default();
        let idx = patch_indices(cfg.grid);
        assert_eq!(idx.len() as u32, cfg.grid * cfg.grid * 6);
        assert!(idx.iter().all(|&i| i < (cfg.grid + 1) * (cfg.grid + 1)));
    }
}
