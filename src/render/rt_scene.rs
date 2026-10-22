//! The ray-tracing scene: a near-field patch of the displaced waves (rebuilt or refitted every
//! frame) and every object mesh (buoy, raft, palms, islands), in one TLAS.

use std::collections::HashMap;

use glam::{Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::gpu::adapter::GpuContext;
use crate::gpu::shaders::ShaderLibrary;
use crate::render::object_pass::MeshHandle;
use crate::waves::gpu_ocean::GpuOcean;
use crate::waves::spectrum::{CASCADE_SIZES_M, cascade_band};

/// TLAS custom data is the material id (scene::mesh::material); the wave patch is water.
pub const PATCH_ID: u32 = crate::scene::mesh::material::WATER;
/// Instance masks: the patch is traced one-sided, objects two-sided in their own query.
const PATCH_MASK: u8 = 0x01;
const OBJECT_MASK: u8 = 0x02;
pub const MAX_INSTANCES: usize = 1024;

/// The `cap` instances nearest `focus`, nearest first (the TLAS has room for MAX_INSTANCES).
pub fn instance_list(instances: &[(MeshHandle, Mat4, u32)], focus: Vec3, cap: usize) -> Vec<(MeshHandle, Mat4, u32)> {
    let mut v = instances.to_vec();
    let dist = |m: &Mat4| m.w_axis.truncate().distance_squared(focus);
    v.sort_by(|a, b| dist(&a.1).total_cmp(&dist(&b.1)));
    v.truncate(cap);
    v
}

/// Which meshes have a BLAS, keyed by mesh handle and remembering the geometry (`K`) each was
/// built from, since a handle freed by an island that streamed out can be reused by another.
pub struct BlasRegistry<K, B> {
    entries: HashMap<MeshHandle, (K, B)>,
}

impl<K, B> Default for BlasRegistry<K, B> {
    fn default() -> Self {
        BlasRegistry {
            entries: HashMap::new(),
        }
    }
}

impl<K: PartialEq + Clone, B> BlasRegistry<K, B> {
    /// Makes the live set exactly `meshes`: drops the rest, builds (via `build`) the new or
    /// changed ones and returns their handles.
    pub fn sync(&mut self, meshes: &[(MeshHandle, K)], mut build: impl FnMut(MeshHandle, &K) -> B) -> Vec<MeshHandle> {
        self.entries
            .retain(|h, (k, _)| meshes.iter().any(|(mh, mk)| mh == h && mk == k));
        let mut new = Vec::new();
        for (h, k) in meshes {
            if !self.entries.contains_key(h) {
                self.entries.insert(*h, (k.clone(), build(*h, k)));
                new.push(*h);
            }
        }
        new
    }

    pub fn get(&self, h: MeshHandle) -> Option<&B> {
        self.entries.get(&h).map(|(_, b)| b)
    }

    pub fn live(&self) -> Vec<MeshHandle> {
        let mut v: Vec<MeshHandle> = self.entries.keys().copied().collect();
        v.sort_by_key(|h| h.0);
        v
    }
}

/// A mesh's (vertices, indices, index count, vertex count), as ObjectPass::mesh_geometry.
pub type MeshGeometry<'a> = (&'a wgpu::Buffer, &'a wgpu::Buffer, u32, u32);

/// An object mesh's BLAS and the geometry it is built from.
struct ObjectBlas {
    blas: wgpu::Blas,
    size: wgpu::BlasTriangleGeometrySizeDescriptor,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
}
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
    seabed: [f32; 4],
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
    /// A BLAS per object mesh (raft, buoy, palm, each streamed-in island).
    objects: BlasRegistry<wgpu::Buffer, ObjectBlas>,
    /// Object BLASes created but not yet built.
    unbuilt: Vec<MeshHandle>,
    /// TLAS slots filled last frame.
    instances_used: usize,
    tlas: wgpu::Tlas,
    since_rebuild: u32,
    last_origin: Option<Vec2>,
    derivatives: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// The seabed map (terrain normals for island hits).
    seabed: wgpu::TextureView,
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
        seabed: &crate::render::seabed::SeabedMap,
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(seabed.view()),
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
        let tlas = d.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("scene tlas"),
            max_instances: MAX_INSTANCES as u32,
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
            objects: BlasRegistry::default(),
            unbuilt: Vec::new(),
            instances_used: 0,
            tlas,
            since_rebuild: 0,
            last_origin: None,
            derivatives: ocean.derivatives_view.clone(),
            seabed: seabed.view().clone(),
            sampler,
        }
    }

    /// The object meshes that can appear in `record`'s instances: (handle, (vertices, indices,
    /// index count, vertex count)), vertex stride 32. BLASes are created for new meshes (built
    /// in the next `record`) and freed for meshes no longer listed.
    pub fn set_static_meshes(&mut self, meshes: &[(MeshHandle, MeshGeometry)]) {
        let keys: Vec<(MeshHandle, wgpu::Buffer)> = meshes.iter().map(|(h, g)| (*h, g.0.clone())).collect();
        let d = &self.device;
        let new = self.objects.sync(&keys, |h, _| {
            let (vb, ib, index_count, vertex_count) = meshes.iter().find(|(mh, _)| *mh == h).expect("listed").1;
            let size = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count,
                index_format: Some(wgpu::IndexFormat::Uint32),
                index_count: Some(index_count),
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };
            let blas = d.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some("object blas"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![size.clone()],
                },
            );
            ObjectBlas {
                blas,
                size,
                vertices: vb.clone(),
                indices: ib.clone(),
            }
        });
        self.unbuilt.retain(|h| keys.iter().any(|(k, _)| k == h));
        self.unbuilt.extend(new);
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

    pub fn seabed(&self) -> &wgpu::TextureView {
        &self.seabed
    }

    /// Writes this frame's patch at `origin`, refits/rebuilds its BLAS, builds any new object
    /// BLASes and rebuilds the TLAS from `instances` (mesh, transform, material id; the caller
    /// orders them nearest first, see `instance_list`).
    pub fn record(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        origin: Vec2,
        instances: &[(MeshHandle, Mat4, u32)],
        seabed: [f32; 4],
    ) {
        let p = PatchParams {
            origin: origin.to_array(),
            cell: self.cfg.cell(),
            grid: self.cfg.grid,
            sizes: CASCADE_SIZES_M,
            lambda_min: std::array::from_fn(|c| std::f32::consts::TAU / cascade_band(c).1),
            seabed,
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
        let identity = instance_transform(Mat4::IDENTITY);
        self.tlas[0] = Some(wgpu::TlasInstance::new(
            &self.patch_blas,
            identity,
            PATCH_ID,
            PATCH_MASK,
        ));
        let mut used = 1;
        for (h, transform, material) in instances.iter().take(MAX_INSTANCES - 1) {
            if let Some(o) = self.objects.get(*h) {
                self.tlas[used] = Some(wgpu::TlasInstance::new(
                    &o.blas,
                    instance_transform(*transform),
                    *material,
                    OBJECT_MASK,
                ));
                used += 1;
            }
        }
        for i in used..self.instances_used {
            self.tlas[i] = None;
        }
        self.instances_used = used;
        let mut entries = vec![wgpu::BlasBuildEntry {
            blas: &self.patch_blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![patch_geo]),
        }];
        for h in self.unbuilt.drain(..) {
            if let Some(o) = self.objects.get(h) {
                entries.push(wgpu::BlasBuildEntry {
                    blas: &o.blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                        size: &o.size,
                        vertex_buffer: &o.vertices,
                        first_vertex: 0,
                        vertex_stride: 32,
                        index_buffer: Some(&o.indices),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    }]),
                });
            }
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
    fn instance_list_respects_capacity() {
        use crate::render::object_pass::MeshHandle;
        let at = |x: f32| Mat4::from_translation(Vec3::new(x, 0.0, 0.0));
        let all: Vec<_> = (0..50)
            .map(|i| (MeshHandle(i), at(500.0 - 10.0 * i as f32), 4))
            .collect();
        let kept = instance_list(&all, Vec3::ZERO, 20);
        assert_eq!(kept.len(), 20);
        // The 20 nearest (handles 30..50, at 200..10 m), nearest first.
        assert_eq!(kept[0].0, MeshHandle(49));
        assert!(
            kept.iter().all(|k| k.0.0 >= 30),
            "{:?}",
            kept.iter().map(|k| k.0).collect::<Vec<_>>()
        );
        assert_eq!(instance_list(&all[..5], Vec3::ZERO, 20).len(), 5);
    }

    #[test]
    fn blas_registry_frees_unloaded_islands() {
        use crate::render::object_pass::MeshHandle;
        use std::rc::Rc;
        let alive = Rc::new(());
        let mut reg: BlasRegistry<u64, Rc<()>> = BlasRegistry::default();
        let mut built = Vec::new();
        let mut build = |h: MeshHandle, _: &u64| {
            built.push(h);
            alive.clone()
        };
        reg.sync(
            &[(MeshHandle(0), 10), (MeshHandle(1), 11), (MeshHandle(2), 12)],
            &mut build,
        );
        assert_eq!(reg.live(), vec![MeshHandle(0), MeshHandle(1), MeshHandle(2)]);
        assert_eq!(Rc::strong_count(&alive), 4);
        // Island 1 streams out: its BLAS is dropped, the others are not rebuilt.
        reg.sync(&[(MeshHandle(0), 10), (MeshHandle(2), 12)], &mut build);
        assert_eq!(reg.live(), vec![MeshHandle(0), MeshHandle(2)]);
        assert_eq!(Rc::strong_count(&alive), 3);
        // A new island reuses handle 2 with different geometry: rebuilt.
        reg.sync(&[(MeshHandle(0), 10), (MeshHandle(2), 99)], &mut build);
        assert_eq!(Rc::strong_count(&alive), 3);
        assert!(reg.get(MeshHandle(1)).is_none() && reg.get(MeshHandle(2)).is_some());
        assert_eq!(built, vec![MeshHandle(0), MeshHandle(1), MeshHandle(2), MeshHandle(2)]);
    }

    #[test]
    fn patch_indices_cover_the_grid() {
        let cfg = RtPatchConfig::default();
        let idx = patch_indices(cfg.grid);
        assert_eq!(idx.len() as u32, cfg.grid * cfg.grid * 6);
        assert!(idx.iter().all(|&i| i < (cfg.grid + 1) * (cfg.grid + 1)));
    }
}
