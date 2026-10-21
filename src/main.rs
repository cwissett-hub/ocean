#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;
use std::time::Instant;

use ocean::gpu::adapter::GpuContext;
use ocean::gpu::extent::render_extent;
use ocean::params::OceanParams;
use ocean::render::frame::Renderer;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Laptop hybrid-graphics drivers read these exports and run us on the discrete GPU.
#[unsafe(no_mangle)]
#[used]
pub static NvOptimusEnablement: u32 = 1;
#[unsafe(no_mangle)]
#[used]
pub static AmdPowerXpressRequestHighPerformance: u32 = 1;

struct Gfx {
    window: Arc<Window>,
    ctx: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
}

struct App {
    gfx: Option<Gfx>,
    params: OceanParams,
    start: Instant,
    last_frame: Instant,
    frames: u32,
    fps_since: Instant,
}

impl Default for App {
    fn default() -> Self {
        let now = Instant::now();
        App { gfx: None, params: OceanParams::default(), start: now, last_frame: now, frames: 0, fps_since: now }
    }
}

impl Gfx {
    fn new(window: Arc<Window>, params: &OceanParams) -> anyhow::Result<Gfx> {
        let (ctx, surface) = GpuContext::new_for_surface(window.clone())?;
        let size = window.inner_size();
        let caps = surface.get_capabilities(&ctx.adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
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
        let renderer = Renderer::new(&ctx, format, (config.width, config.height), params);
        Ok(Gfx { window, ctx, surface, config, renderer })
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

    fn render(&mut self, params: &OceanParams, t: f32, dt: f32) {
        let size = self.window.inner_size();
        if render_extent(size.width, size.height, params.render_scale).is_none() {
            return;
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                self.surface.configure(&self.ctx.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        self.renderer.render(&view, params, t, dt);
        self.ctx.queue.present(frame);
    }
}

fn present_mode(vsync: bool) -> wgpu::PresentMode {
    if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Ocean")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(el.create_window(attrs).expect("create window"));
        match Gfx::new(window, &self.params) {
            Ok(g) => self.gfx = Some(g),
            Err(e) => {
                eprintln!("ocean: {e:#}");
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gfx) = self.gfx.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => gfx.resize(s.width, s.height),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;
                gfx.render(&self.params, (now - self.start).as_secs_f32(), dt);
                self.frames += 1;
                let secs = self.fps_since.elapsed().as_secs_f32();
                if secs >= 0.5 {
                    let fps = self.frames as f32 / secs;
                    gfx.window.set_title(&format!("Ocean — {} — {fps:.0} fps", gfx.ctx.adapter_name));
                    self.frames = 0;
                    self.fps_since = Instant::now();
                }
                gfx.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    let el = EventLoop::new()?;
    el.set_control_flow(ControlFlow::Poll);
    el.run_app(&mut App::default())?;
    Ok(())
}
