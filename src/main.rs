#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;
use std::time::Instant;

use ocean::bench;
use ocean::clock::sim_time;
use ocean::gpu::adapter::GpuContext;
use ocean::gpu::extent::render_extent;
use ocean::gpu::shaders::ShaderLibrary;
use ocean::input::KeyState;
use ocean::params::{OceanParams, apply_overrides};
use ocean::render::frame::{FrameInput, Renderer};
use ocean::scene::cameras::CameraMode;
use ocean::ui::{self, OverlayInfo};
use ocean::world::islands::WORLD_SEED;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Laptop hybrid-graphics drivers read these exports and run us on the discrete GPU.
#[unsafe(no_mangle)]
#[used]
pub static NvOptimusEnablement: u32 = 1;
#[unsafe(no_mangle)]
#[used]
pub static AmdPowerXpressRequestHighPerformance: u32 = 1;

struct Bench {
    frames: u32,
    out: std::path::PathBuf,
    camera: CameraMode,
    seen: u32,
    cpu_ms: Vec<f32>,
}

struct Gfx {
    window: Arc<Window>,
    ctx: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    shaders: ShaderLibrary,
    shader_errors: Vec<String>,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

struct App {
    gfx: Option<Gfx>,
    params: OceanParams,
    show_panel: bool,
    bench: Option<Bench>,
    start: Instant,
    last_frame: Instant,
    frames: u32,
    fps_since: Instant,
    fps: f32,
    /// Fixed simulation time (for A/B screenshots).
    freeze_time: Option<f32>,
    keys: KeyState,
    sail_up: bool,
    cycle_camera: bool,
    start_at: Option<(f64, f64, f32)>,
}

impl App {
    fn new(params: OceanParams, bench: Option<Bench>) -> Self {
        let now = Instant::now();
        App {
            gfx: None,
            params,
            show_panel: false,
            bench,
            start: now,
            last_frame: now,
            frames: 0,
            fps_since: now,
            fps: 0.0,
            freeze_time: None,
            keys: KeyState::default(),
            sail_up: true,
            cycle_camera: false,
            start_at: None,
        }
    }
}

fn present_mode(vsync: bool) -> wgpu::PresentMode {
    if vsync {
        wgpu::PresentMode::AutoVsync
    } else {
        wgpu::PresentMode::AutoNoVsync
    }
}

impl Gfx {
    fn new(window: Arc<Window>, params: &OceanParams) -> anyhow::Result<Gfx> {
        let (ctx, surface) = GpuContext::new_for_surface(window.clone())?;
        let size = window.inner_size();
        let caps = surface.get_capabilities(&ctx.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: present_mode(params.vsync),
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&ctx.device, &config);
        let mut shaders = ShaderLibrary::discover();
        let mut shader_errors: Vec<String> = shaders
            .load_errors()
            .iter()
            .map(|(n, e)| format!("{n}.wgsl (using built-in copy): {e}"))
            .collect();
        let extent = (config.width, config.height);
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut renderer = Renderer::new(&ctx, &shaders, format, extent, params);
        if let Some(e) = pollster::block_on(scope.pop()) {
            // Disk shaders passed naga but not pipeline creation: fall back to the built-in set.
            shader_errors.push(format!("pipeline (using built-in shaders): {e}"));
            shaders = ShaderLibrary::embedded();
            renderer = Renderer::new(&ctx, &shaders, format, extent, params);
        }
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(ctx.device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(&ctx.device, format, egui_wgpu::RendererOptions::default());
        Ok(Gfx {
            window,
            ctx,
            surface,
            config,
            renderer,
            shaders,
            shader_errors,
            egui_ctx,
            egui_state,
            egui_renderer,
        })
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.ctx.device, &self.config);
        self.renderer.resize((w, h));
    }

    /// F5: re-read shaders from disk; keep the running renderer if anything fails.
    fn reload_shaders(&mut self, params: &OceanParams) {
        let errors = self.shaders.reload_all();
        if !errors.is_empty() {
            self.shader_errors = errors.into_iter().map(|(n, e)| format!("{n}.wgsl: {e}")).collect();
            return;
        }
        let scope = self.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let fresh = Renderer::new(
            &self.ctx,
            &self.shaders,
            self.config.format,
            (self.config.width, self.config.height),
            params,
        );
        match pollster::block_on(scope.pop()) {
            None => {
                self.renderer = fresh;
                self.shader_errors.clear();
            }
            Some(e) => self.shader_errors = vec![format!("pipeline: {e}")],
        }
    }

    fn render(&mut self, params: &mut OceanParams, show_panel: bool, fps: f32, t: f32, dt: f32, input: &FrameInput) {
        let size = self.window.inner_size();
        let Some(internal) = render_extent(size.width, size.height, params.render_scale) else {
            return;
        };
        let want_mode = present_mode(params.vsync);
        if self.config.present_mode != want_mode {
            self.config.present_mode = want_mode;
            self.surface.configure(&self.ctx.device, &self.config);
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                self.surface.configure(&self.ctx.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());

        let (stats, total) = self.renderer.gpu_stats();
        let info = OverlayInfo {
            adapter: self.ctx.adapter_name.clone(),
            rt_reason: self.ctx.reason.clone(),
            fps,
            internal_extent: internal,
            stats,
            total,
            shader_errors: self.shader_errors.clone(),
        };
        let raw = self.egui_state.take_egui_input(&self.window);
        let rt = self.ctx.rt_available;
        let raft = ui::raft_info(
            self.renderer.raft(),
            input.controls.sail_up,
            self.renderer.camera_mode(),
        );
        let out = self.egui_ctx.run_pass(raw, |ctx| {
            if show_panel {
                ui::settings_panel(ctx, params, rt, Some(&raft));
            }
            if params.show_overlay {
                ui::overlay(ctx, &info, params.show_timings);
            } else if !info.shader_errors.is_empty() {
                ui::shader_errors(ctx, &info.shader_errors);
            }
            if params.show_controls {
                ui::controls_hint(ctx);
            }
        });
        self.egui_state
            .handle_platform_output(&self.window, out.platform_output);
        let prims = self.egui_ctx.tessellate(out.shapes, out.pixels_per_point);
        for (id, deltas) in &out.textures_delta.set {
            for delta in deltas {
                self.egui_renderer
                    .update_texture(&self.ctx.device, &self.ctx.queue, *id, delta);
            }
        }
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: out.pixels_per_point,
        };
        let (device, queue, egui_renderer) = (&self.ctx.device, &self.ctx.queue, &mut self.egui_renderer);
        self.renderer.render(&view, params, t, dt, input, |enc, target| {
            egui_renderer.update_buffers(device, queue, enc, &prims, &screen);
            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            egui_renderer.render(&mut pass.forget_lifetime(), &prims, &screen);
        });
        for id in &out.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        self.ctx.queue.present(frame);
        *params = params.clamped();
    }
}

impl ApplicationHandler for App {
    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.keys.mouse_moved(glam::Vec2::new(delta.0 as f32, delta.1 as f32));
        }
    }

    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Ocean")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(el.create_window(attrs).expect("create window"));
        match Gfx::new(window, &self.params) {
            Ok(mut g) => {
                if let Some((x, z, h)) = self.start_at {
                    g.renderer.place_raft(glam::DVec2::new(x, z), h);
                }
                if let Some(b) = &self.bench {
                    g.renderer.set_camera_mode(b.camera);
                }
                self.gfx = Some(g)
            }
            Err(e) => {
                eprintln!("ocean: {e:#}");
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gfx) = self.gfx.as_mut() else { return };
        let consumed = gfx.egui_state.on_window_event(&gfx.window, &event).consumed;
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => {
                gfx.resize(s.width, s.height);
                // Restored from minimised: restart the frame loop.
                gfx.window.request_redraw();
            }
            WindowEvent::Occluded(false) => gfx.window.request_redraw(),
            WindowEvent::KeyboardInput { event: k, .. } if k.state == ElementState::Released => {
                if let PhysicalKey::Code(code) = k.physical_key {
                    self.keys.release(code);
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                self.keys.right_mouse = state == ElementState::Pressed && !consumed;
            }
            WindowEvent::Focused(false) => self.keys = KeyState::default(),
            WindowEvent::KeyboardInput { event: k, .. } if !consumed && k.state == ElementState::Pressed => {
                if let PhysicalKey::Code(code) = k.physical_key {
                    self.keys.press(code);
                }
                if k.repeat {
                    return;
                }
                match k.physical_key {
                    PhysicalKey::Code(KeyCode::Space) => self.sail_up = !self.sail_up,
                    PhysicalKey::Code(KeyCode::KeyC) => self.cycle_camera = true,
                    PhysicalKey::Code(KeyCode::F1) => self.show_panel = !self.show_panel,
                    PhysicalKey::Code(KeyCode::F2) => self.params.show_overlay = !self.params.show_overlay,
                    PhysicalKey::Code(KeyCode::KeyH) => self.params.show_controls = !self.params.show_controls,
                    PhysicalKey::Code(KeyCode::F5) => gfx.reload_shaders(&self.params),
                    PhysicalKey::Code(KeyCode::Escape) => el.exit(),
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                // Minimised: draw nothing and stop asking for frames until Resized/Occluded wakes us.
                let size = gfx.window.inner_size();
                if gfx.window.is_minimized() == Some(true) || size.width == 0 || size.height == 0 {
                    return;
                }
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;
                // f64 until reduced to one wave period: f32 seconds judder after hours.
                let t = sim_time(self.freeze_time.map_or((now - self.start).as_secs_f64(), f64::from));
                // A bench sails its scripted course; otherwise the player has the helm.
                let controls = if self.bench.is_some() {
                    bench::course_controls((now - self.start).as_secs_f32())
                } else {
                    self.keys.raft_controls(self.sail_up)
                };
                let input = FrameInput {
                    controls,
                    free: self.keys.free_fly(),
                    cycle_camera: std::mem::take(&mut self.cycle_camera),
                };
                gfx.render(&mut self.params, self.show_panel, self.fps, t, dt, &input);
                self.frames += 1;
                let secs = self.fps_since.elapsed().as_secs_f32();
                if secs >= 0.5 {
                    self.fps = self.frames as f32 / secs;
                    gfx.window
                        .set_title(&format!("Ocean — {} — {:.0} fps", gfx.ctx.adapter_name, self.fps));
                    self.frames = 0;
                    self.fps_since = Instant::now();
                }
                if let Some(b) = &mut self.bench {
                    b.seen += 1;
                    if b.seen == bench::WARMUP_FRAMES {
                        gfx.renderer.gpu_timer().start_log();
                    } else if b.seen > bench::WARMUP_FRAMES {
                        b.cpu_ms.push(dt * 1000.0);
                    }
                    let logged = gfx.renderer.gpu_timer().log().map_or(0, |l| l.totals.len());
                    if b.seen >= bench::WARMUP_FRAMES + b.frames && logged + 3 >= b.frames as usize {
                        let log = gfx.renderer.gpu_timer().log().cloned().unwrap_or_default();
                        let size = gfx.window.inner_size();
                        let ext = render_extent(size.width, size.height, self.params.render_scale).unwrap_or((0, 0));
                        let report = bench::report(&gfx.ctx.adapter_name, ext, &log, &b.cpu_ms);
                        println!("{report}");
                        if let Err(e) = std::fs::write(&b.out, format!("{report}\n")) {
                            eprintln!("ocean: writing {}: {e}", b.out.display());
                        }
                        el.exit();
                    }
                }
                gfx.window.request_redraw();
            }
            _ => {}
        }
    }
}

/// `--set key=value` (repeatable) overrides any OceanParams field.
/// `--bench N [--bench-out file] [--bench-camera chase|deck|high]` sails the scripted bench
/// course past the start island, rendering N frames uncapped, and writes a JSON report.
/// `--panel` opens the settings panel at start; `--freeze-time S` pins the waves at time S;
/// `--start-at x,z[,heading]` puts the raft at a world position.
struct CliArgs {
    params: OceanParams,
    bench: Option<Bench>,
    panel: bool,
    freeze: Option<f32>,
    start_at: Option<(f64, f64, f32)>,
}

fn parse_args() -> anyhow::Result<CliArgs> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut overrides = Vec::new();
    let mut bench_frames = None;
    let mut bench_out = std::path::PathBuf::from("bench.json");
    let mut bench_camera = CameraMode::Chase;
    let mut panel = false;
    let mut freeze = None;
    let mut start_at = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = || it.next().cloned().ok_or_else(|| anyhow::anyhow!("{a} needs a value"));
        match a.as_str() {
            "--set" => overrides.push(value()?),
            "--bench" => bench_frames = Some(value()?.parse::<u32>()?),
            "--bench-out" => bench_out = value()?.into(),
            "--bench-camera" => {
                let v = value()?;
                bench_camera = bench::bench_camera(&v)
                    .ok_or_else(|| anyhow::anyhow!("--bench-camera chase|deck|high, not {v:?}"))?;
            }
            "--panel" => panel = true,
            "--freeze-time" => freeze = Some(value()?.parse::<f32>()?),
            "--start-at" => {
                let v = value()?;
                let n: Vec<f64> = v.split(',').map(str::parse).collect::<Result<_, _>>()?;
                anyhow::ensure!(n.len() == 2 || n.len() == 3, "--start-at x,z[,heading]");
                start_at = Some((n[0], n[1], n.get(2).copied().unwrap_or(30.0) as f32));
            }
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    let mut params = apply_overrides(OceanParams::default(), &overrides).map_err(anyhow::Error::msg)?;
    let bench = bench_frames.map(|frames| {
        params.vsync = false;
        Bench {
            frames,
            out: bench_out,
            camera: bench_camera,
            seen: 0,
            cpu_ms: Vec::new(),
        }
    });
    // A bench without --start-at sails the course.
    if bench.is_some() && start_at.is_none() {
        let (p, h) = bench::course_start(WORLD_SEED);
        start_at = Some((p.x, p.y, h));
    }
    Ok(CliArgs {
        params,
        bench,
        panel,
        freeze,
        start_at,
    })
}

fn main() -> anyhow::Result<()> {
    let CliArgs {
        params,
        bench,
        panel,
        freeze,
        start_at,
    } = parse_args()?;
    let el = EventLoop::new()?;
    // Wait, not Poll: request_redraw keeps frames coming, and a minimised window (which
    // renders nothing) then sleeps instead of spinning a core.
    el.set_control_flow(ControlFlow::Wait);
    let mut app = App::new(params, bench);
    app.show_panel = panel;
    app.freeze_time = freeze;
    app.start_at = start_at;
    el.run_app(&mut app)?;
    Ok(())
}
