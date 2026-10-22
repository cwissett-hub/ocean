//! Owns every pass and records one frame in order:
//! wave sim → sky LUTs (if the sun moved) → G-buffer → shade (HDR) → bloom + tonemap.

use glam::{Mat4, Vec2, Vec3};

use crate::clock::should_rebuild_spectrum;
use crate::gpu::adapter::GpuContext;
use crate::gpu::extent::render_extent;
use crate::gpu::shaders::ShaderLibrary;
use crate::gpu::timing::{GpuTimer, PassStats};
use crate::params::OceanParams;
use crate::render::gbuffer::{DEPTH_FORMAT, GBuffer, MATERIAL_FORMAT, MOTION_FORMAT, NORMAL_FORMAT};
use crate::render::object_pass::{MeshHandle, ObjectPass};
use crate::render::ocean_mesh::{BASE_CELL_M, GRID, LEVELS, OceanMesh};
use crate::render::particles::{ParticleInputs, ParticleSystem};
use crate::render::post::{Post, exposure_ev, exposure_multiplier};
use crate::render::reflect::ReflectionPass;
use crate::render::rt_scene::{RtPatchConfig, RtScene, patch_origin};
use crate::render::seabed::SeabedMap;
use crate::render::shade::ShadePass;
use crate::render::sky::{SkyLuts, sun_direction};
use crate::scene::buoy::{BuoyBody, build_buoy_mesh};
use crate::scene::camera::Camera;
use crate::scene::cameras::{CameraController, CameraMode, FreeFlyInput};
use crate::scene::raft::{RaftBody, RaftControls};
use crate::scene::raft_mesh::build_raft_mesh;
use crate::waves::foam::foam_params;
use crate::waves::gpu_ocean::GpuOcean;
use crate::waves::query::{WaterSample, WaveQuery};
use crate::world::islands::WORLD_SEED;
use crate::world::origin::WorldOrigin;
use crate::world::streaming::{IslandStreamer, ReadyIsland};
use crate::world::terrain_mesh::palm_mesh;

/// An island with a mesh on the GPU.
struct LiveIsland {
    desc: crate::world::islands::IslandDesc,
    mesh: MeshHandle,
    palms: Vec<Mat4>,
}

/// Island meshes uploaded per frame (each is a few MB at most).
const ISLAND_UPLOADS_PER_FRAME: usize = 2;
use crate::waves::spectrum::{
    CASCADE_SIZES_M, REPEAT_PERIOD_S, SpectrumSet, build_spectrum, cascade_band, slope_variance,
    spectrum_significant_height, unresolved_slope_variance,
};

const SEED: u64 = 42;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sun_dir: [f32; 4],
    sizes: [f32; 4],
    lambda_min: [f32; 4],
    grid: [f32; 4],
    slope_var: [f32; 4],
    misc: [f32; 4],
    resolution: [f32; 4],
    /// SeabedMap::params.
    seabed: [f32; 4],
}

/// The parameters that change the wave spectrum (rebuilding it costs a few milliseconds).
#[derive(PartialEq, Clone, Copy)]
struct SpectrumKey {
    wind_speed: f32,
    wind_dir_deg: f32,
    fetch_km: f32,
    swell_amount: f32,
}

impl SpectrumKey {
    fn of(p: &OceanParams) -> Self {
        SpectrumKey {
            wind_speed: p.wind_speed,
            wind_dir_deg: p.wind_dir_deg,
            fetch_km: p.fetch_km,
            swell_amount: p.swell_amount,
        }
    }
}

struct SpectrumStats {
    slope_var: [f32; 4],
    unresolved: f32,
    hs: f32,
}

impl SpectrumStats {
    fn of(s: &SpectrumSet, p: &OceanParams) -> Self {
        SpectrumStats {
            slope_var: slope_variance(s),
            unresolved: unresolved_slope_variance(s, p.wind_speed),
            hs: spectrum_significant_height(s),
        }
    }
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ocean: GpuOcean,
    sky: SkyLuts,
    mesh: OceanMesh,
    uniforms: wgpu::Buffer,
    /// One per foam ping-pong side (see GpuOcean::foam_index).
    ocean_bind_groups: [wgpu::BindGroup; 2],
    gbuffer_pipeline: wgpu::RenderPipeline,
    gbuffer: GBuffer,
    reflections: ReflectionPass,
    shade: ShadePass,
    post: Post,
    surface_extent: (u32, u32),
    cameras: CameraController,
    prev_view_proj: Option<Mat4>,
    spectrum_key: SpectrumKey,
    stats: SpectrumStats,
    since_spectrum: f32,
    timer: GpuTimer,
    /// Ray-tracing scene; None when the device has no ray query.
    rt: Option<RtScene>,
    particles: ParticleSystem,
    objects: ObjectPass,
    origin: WorldOrigin,
    streamer: IslandStreamer,
    palm_mesh: MeshHandle,
    islands: std::collections::HashMap<u64, LiveIsland>,
    ready: std::collections::VecDeque<ReadyIsland>,
    buoy_mesh: MeshHandle,
    raft_mesh: MeshHandle,
    buoy_body: BuoyBody,
    raft: RaftBody,
    seabed: SeabedMap,
    query: WaveQuery,
    /// Latest water samples (raft 9, buoy 5, camera 1) and the heights/time they came from.
    water: Option<[WaterSample; QUERY_POINTS]>,
    last_heights: Option<([f32; QUERY_POINTS], f32)>,
}

/// Per-frame player input.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameInput {
    pub controls: RaftControls,
    pub free: FreeFlyInput,
    pub cycle_camera: bool,
}

/// Water query layout: raft samples, then buoy samples, then the camera.
const RAFT_Q: usize = 0;
const BUOY_Q: usize = 9;
const CAMERA_Q: usize = 14;
const QUERY_POINTS: usize = 15;
/// The buoy is moored here, near the start (world = local until the first rebase).
const BUOY_HOME: Vec3 = Vec3::new(-25.0, 0.0, 40.0);

/// The raft starts 20 m behind the origin, facing the start island under the sun.
fn raft_start() -> RaftBody {
    let heading = 30f32.to_radians();
    let fwd = Vec3::new(heading.sin(), 0.0, heading.cos());
    RaftBody::at(-fwd * 20.0, heading)
}

fn internal_extent(surface: (u32, u32), scale: f32) -> (u32, u32) {
    render_extent(surface.0, surface.1, scale).unwrap_or((1, 1))
}

impl Renderer {
    pub fn new(
        ctx: &GpuContext,
        lib: &ShaderLibrary,
        format: wgpu::TextureFormat,
        surface_extent: (u32, u32),
        params: &OceanParams,
    ) -> Renderer {
        let d = &ctx.device;
        let spectrum = build_spectrum(params, SEED);
        let ocean = GpuOcean::new(ctx, lib, &spectrum);
        let sky = SkyLuts::new(ctx, lib);
        let mesh = OceanMesh::new(d);
        let seabed = SeabedMap::new(d);
        let uniforms = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ocean repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let vis = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ocean gbuffer"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vis,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex(1),
                tex(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: vis,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: vis,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        use wgpu::BindingResource as B;
        let ocean_bind_groups = std::array::from_fn(|i| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ocean gbuffer"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: B::TextureView(&ocean.displacement_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: B::TextureView(&ocean.derivatives_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: B::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: B::TextureView(&ocean.foam_views()[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: B::TextureView(seabed.view()),
                    },
                ],
            })
        });
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ocean gbuffer"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ocean_gbuffer"),
            source: wgpu::ShaderSource::Wgsl(lib.source("ocean_gbuffer").expect("shader source").into()),
        });
        let gbuffer_pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ocean gbuffer"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_ocean"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_ocean"),
                compilation_options: Default::default(),
                targets: &[
                    Some(NORMAL_FORMAT.into()),
                    Some(MATERIAL_FORMAT.into()),
                    Some(MOTION_FORMAT.into()),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        let gbuffer = GBuffer::new(d, internal_extent(surface_extent, params.render_scale));
        let reflections = ReflectionPass::new(d, lib, &ctx.queue, &uniforms, &gbuffer);
        let particles = ParticleSystem::new(d, &ctx.queue, lib, &uniforms, &ocean, &sky, &gbuffer, &seabed);
        let shade = ShadePass::new(d, lib, &uniforms, &sky, &gbuffer, reflections.views(), seabed.view());
        let post = Post::new(d, lib, format, &gbuffer);
        let mut objects = ObjectPass::new(d, &ctx.queue, lib, &layout);
        let buoy_mesh = objects.add_mesh(d, &build_buoy_mesh());
        let raft_mesh = objects.add_mesh(d, &build_raft_mesh());
        let palm_mesh = objects.add_mesh(d, &palm_mesh());
        let query = WaveQuery::new(ctx, lib, &ocean, &seabed);
        let raft = raft_start();
        let cameras = CameraController::new(&raft);
        let mut reflections = reflections;
        let rt = ctx.rt_available.then(|| {
            let (vb, ib, indices, vertices) = objects.mesh_geometry(buoy_mesh).expect("buoy mesh");
            let scene = RtScene::new(
                ctx,
                lib,
                &ocean,
                (vb, ib, indices),
                vertices,
                &seabed,
                RtPatchConfig::default(),
            );
            reflections.enable_rt(d, lib, &sky, &uniforms, &gbuffer, &scene);
            scene
        });
        Renderer {
            rt,
            particles,
            objects,
            buoy_mesh,
            raft_mesh,
            origin: WorldOrigin::default(),
            streamer: IslandStreamer::new(WORLD_SEED),
            palm_mesh,
            islands: std::collections::HashMap::new(),
            ready: std::collections::VecDeque::new(),
            buoy_body: BuoyBody::at(BUOY_HOME),
            raft,
            seabed,
            query,
            water: None,
            last_heights: None,
            device: d.clone(),
            queue: ctx.queue.clone(),
            ocean,
            sky,
            mesh,
            uniforms,
            ocean_bind_groups,
            gbuffer_pipeline,
            gbuffer,
            reflections,
            shade,
            post,
            surface_extent,
            cameras,
            prev_view_proj: None,
            spectrum_key: SpectrumKey::of(params),
            stats: SpectrumStats::of(&spectrum, params),
            since_spectrum: 0.0,
            timer: GpuTimer::new(d, &ctx.queue),
        }
    }

    pub fn resize(&mut self, surface_extent: (u32, u32)) {
        self.surface_extent = surface_extent;
    }

    /// Moves the raft (world position, heading in degrees) — the `--start-at` dev flag.
    pub fn place_raft(&mut self, world: glam::DVec2, heading_deg: f32) {
        let p = self.origin.to_local(world);
        self.raft = RaftBody::at(Vec3::new(p.x, 0.0, p.y), heading_deg.to_radians());
        self.cameras = CameraController::new(&self.raft);
    }

    pub fn raft(&self) -> &RaftBody {
        &self.raft
    }

    pub fn camera_mode(&self) -> CameraMode {
        self.cameras.mode
    }

    /// Steps the raft and buoy on the newest water samples, moves the camera, and queues the
    /// next water query (raft, buoy, camera points).
    fn update_world(&mut self, params: &OceanParams, input: &FrameInput, t: f32, dt: f32) -> Camera {
        if input.cycle_camera {
            self.cameras.mode = self.cameras.mode.next();
        }
        if let Some(results) = self.query.poll(&self.device).filter(|r| r.len() == QUERY_POINTS) {
            let heights: [f32; QUERY_POINTS] = std::array::from_fn(|i| results[i].0);
            let samples = std::array::from_fn(|i| {
                let vy = match self.last_heights {
                    Some((prev, pt)) if t > pt => (heights[i] - prev[i]) / (t - pt),
                    _ => 0.0,
                };
                WaterSample {
                    height: heights[i],
                    normal: results[i].1,
                    velocity: Vec3::new(0.0, vy, 0.0),
                }
            });
            self.last_heights = Some((heights, t));
            self.water = Some(samples);
        }
        let a = params.wind_dir_deg.to_radians();
        let wind = Vec2::new(a.sin(), a.cos()) * params.wind_speed;
        // Free fly leaves the raft drifting with whatever sail it had.
        let controls = if self.cameras.mode == CameraMode::FreeFly {
            RaftControls {
                steer: 0.0,
                paddle: 0.0,
                sail_up: input.controls.sail_up,
            }
        } else {
            input.controls
        };
        self.stream_islands();
        let (streamer, origin) = (&self.streamer, self.origin);
        let terrain = |p: Vec2| streamer.terrain_height(origin.to_world(p));
        let mut camera_water = 0.0;
        if let Some(s) = &self.water {
            let raft_w: [WaterSample; 9] = std::array::from_fn(|i| s[RAFT_Q + i]);
            let buoy_w: [WaterSample; 5] = std::array::from_fn(|i| s[BUOY_Q + i]);
            self.raft.step(&raft_w, wind, &controls, &terrain, dt.min(0.25));
            self.buoy_body.step(&buoy_w, dt.min(0.25));
            camera_water = s[CAMERA_Q].height;
        }
        // The camera's water sample is from where it was last frame; add a margin for waves.
        let ground = |p: Vec2| (camera_water + 0.3 * self.stats.hs).max(terrain(p));
        let cam = self
            .cameras
            .update(dt, &self.raft, params.sun_azimuth_deg, &input.free, &ground);
        let mut instances = vec![
            (1, self.buoy_mesh, self.buoy_body.transform()),
            (2, self.raft_mesh, self.raft.transform()),
        ];
        for (id, isle) in &self.islands {
            let c = self.origin.to_local(isle.desc.center);
            let t = Mat4::from_translation(Vec3::new(c.x, 0.0, c.y));
            instances.push((*id, isle.mesh, t));
            for (k, p) in isle.palms.iter().enumerate() {
                instances.push((
                    id ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(k as u64 + 1)),
                    self.palm_mesh,
                    t * *p,
                ));
            }
        }
        self.objects.set_instances(&instances);
        let mut points = [Vec2::ZERO; QUERY_POINTS];
        points[RAFT_Q..RAFT_Q + 9].copy_from_slice(&self.raft.sample_points());
        points[BUOY_Q..BUOY_Q + 5].copy_from_slice(&self.buoy_body.sample_points());
        points[CAMERA_Q] = Vec2::new(cam.position.x, cam.position.z);
        self.query.request(&points);
        cam
    }

    /// Plans island streaming around the raft, drops far islands, and uploads a few finished
    /// meshes (more wait for later frames, so an archipelago never causes a hitch).
    fn stream_islands(&mut self) {
        let focus = self
            .origin
            .to_world(Vec2::new(self.raft.position.x, self.raft.position.z));
        self.streamer.update(focus);
        self.seabed
            .update(&self.queue, WORLD_SEED, focus, self.streamer.known());
        self.query.set_seabed(self.seabed.params(&self.origin));
        for id in self.streamer.take_unloads() {
            if let Some(isle) = self.islands.remove(&id) {
                self.objects.remove_mesh(isle.mesh);
            }
            self.ready.retain(|r| r.desc.id != id);
        }
        self.ready.extend(self.streamer.take_ready());
        for _ in 0..ISLAND_UPLOADS_PER_FRAME {
            let Some(r) = self.ready.pop_front() else { break };
            let mesh = self.objects.add_mesh(&self.device, &r.mesh);
            if let Some(old) = self.islands.insert(
                r.desc.id,
                LiveIsland {
                    desc: r.desc,
                    mesh,
                    palms: r.palms,
                },
            ) {
                self.objects.remove_mesh(old.mesh);
            }
        }
    }

    fn ensure_targets(&mut self, scale: f32) {
        let want = internal_extent(self.surface_extent, scale);
        if want != self.gbuffer.extent {
            self.gbuffer = GBuffer::new(&self.device, want);
            self.reflections
                .resize(&self.device, &self.uniforms, &self.gbuffer, self.rt.as_ref());
            self.shade
                .rebind(&self.device, &self.uniforms, &self.gbuffer, self.reflections.views());
            self.post.resize(&self.device, &self.gbuffer);
            self.particles.resize(&self.device, &self.uniforms, &self.gbuffer);
            self.prev_view_proj = None;
        }
    }

    /// Rebuilds the spectrum after a sea-setting change, at most every
    /// SPECTRUM_REBUILD_INTERVAL_S while a slider is being dragged.
    fn ensure_spectrum(&mut self, params: &OceanParams, dt: f32) {
        let key = SpectrumKey::of(params);
        self.since_spectrum += dt.max(0.0);
        if should_rebuild_spectrum(key != self.spectrum_key, self.since_spectrum) {
            self.since_spectrum = 0.0;
            let s = build_spectrum(params, SEED);
            self.ocean.set_spectrum(&self.queue, &s);
            self.stats = SpectrumStats::of(&s, params);
            self.spectrum_key = key;
        }
    }

    /// GPU time per pass and per frame over the last few seconds.
    pub fn gpu_stats(&self) -> (Vec<(&'static str, PassStats)>, PassStats) {
        (self.timer.stats(), self.timer.total())
    }

    pub fn gpu_timer(&mut self) -> &mut GpuTimer {
        &mut self.timer
    }

    /// Records and submits one frame into `view` (the surface texture). `overlay` records any
    /// passes drawn on top (the UI) into the same submission.
    pub fn render(
        &mut self,
        view: &wgpu::TextureView,
        params: &OceanParams,
        t: f32,
        dt: f32,
        input: &FrameInput,
        overlay: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView),
    ) {
        self.timer.poll(&self.device);
        let params = params.clamped();
        self.ensure_targets(params.render_scale);
        self.ensure_spectrum(&params, dt);
        let cam = self.update_world(&params, input, t, dt);
        let (w, h) = self.gbuffer.extent;
        let vp = cam.view_proj(w as f32 / h as f32);
        let prev = self.prev_view_proj.unwrap_or(vp);
        self.prev_view_proj = Some(vp);
        let sun = sun_direction(params.sun_elevation_deg, params.sun_azimuth_deg);
        let exposure = exposure_multiplier(exposure_ev(params.sun_elevation_deg));
        let u = FrameUniforms {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: vp.inverse().to_cols_array_2d(),
            prev_view_proj: prev.to_cols_array_2d(),
            cam_pos: [cam.position.x, cam.position.y, cam.position.z, t],
            sun_dir: [sun.x, sun.y, sun.z, 0.0],
            sizes: CASCADE_SIZES_M,
            lambda_min: std::array::from_fn(|c| std::f32::consts::TAU / cascade_band(c).1),
            grid: [BASE_CELL_M, LEVELS as f32, GRID as f32, 0.0],
            slope_var: self.stats.slope_var,
            misc: [
                self.stats.unresolved,
                self.stats.hs,
                exposure,
                params.wind_dir_deg.to_radians(),
            ],
            resolution: [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32],
            seabed: self.seabed.params(&self.origin),
        };
        self.queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        let mut enc = self.device.create_command_encoder(&Default::default());
        self.timer.start(&mut enc);
        self.ocean.update(
            &mut enc,
            t % REPEAT_PERIOD_S,
            params.choppiness,
            foam_params(&params),
            dt,
        );
        self.query.record(&mut enc);
        self.timer.mark(&mut enc, "ocean sim");
        if let Some(rt) = &mut self.rt {
            let origin = patch_origin(cam.position, cam.forward, &rt.config());
            if let Some((vb, ib, _, _)) = self.objects.mesh_geometry(self.buoy_mesh) {
                rt.record(
                    &mut enc,
                    origin,
                    self.buoy_body.transform(),
                    (vb, ib),
                    self.seabed.params(&self.origin),
                );
            }
        }
        self.timer.mark(&mut enc, "rt build");
        self.particles.record(
            &mut enc,
            &ParticleInputs {
                dt,
                wind_speed: params.wind_speed,
                wind_dir_deg: params.wind_dir_deg,
                amount: params.particle_amount,
                foam_threshold: foam_params(&params).threshold,
                cam_pos: cam.position,
                cam_forward: cam.forward,
            },
        );
        self.timer.mark(&mut enc, "particle sim");
        self.sky.update(&mut enc, sun);
        self.timer.mark(&mut enc, "sky");
        self.objects.record_opaque(
            &mut enc,
            &self.gbuffer,
            &self.ocean_bind_groups[self.ocean.foam_index()],
        );
        {
            let g = &self.gbuffer;
            let color = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gbuffer"),
                color_attachments: &[color(&g.normal), color(&g.material), color(&g.motion)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &g.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.gbuffer_pipeline);
            pass.set_bind_group(0, &self.ocean_bind_groups[self.ocean.foam_index()], &[]);
            pass.set_index_buffer(self.mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.mesh.index_count, 0, 0..LEVELS);
            self.objects
                .draw_gbuffer(&mut pass, &self.ocean_bind_groups[self.ocean.foam_index()]);
        }
        self.timer.mark(&mut enc, "gbuffer");
        self.reflections.record(&mut enc, params.reflection_mode, &self.sky);
        self.timer.mark(&mut enc, "reflections");
        self.shade
            .record(&mut enc, &self.gbuffer, &self.sky, self.reflections.index());
        self.particles.draw(&mut enc, &self.gbuffer.hdr, &self.sky);
        self.gbuffer.record_history(&mut enc);
        self.timer.mark(&mut enc, "shade");
        self.post.record(&mut enc, &self.queue, view, exposure);
        self.timer.mark(&mut enc, "post");
        overlay(&mut enc, view);
        self.timer.mark(&mut enc, "ui");
        self.timer.finish(&mut enc);
        self.queue.submit([enc.finish()]);
        self.timer.after_submit();
        self.query.after_submit();
    }
}
