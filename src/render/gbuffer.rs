//! Render targets at the internal resolution (window size × render scale).

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct GBuffer {
    pub extent: (u32, u32),
    pub depth: wgpu::TextureView,
    /// World normal (xyz) and total slope variance (w).
    pub normal: wgpu::TextureView,
    /// Foam (r), material id (g; 0 = water).
    pub material: wgpu::TextureView,
    pub motion: wgpu::TextureView,
    /// Linear HDR scene colour before exposure.
    pub hdr: wgpu::TextureView,
    pub hdr_tex: wgpu::Texture,
    /// Last frame's HDR colour: what screen-space reflections see.
    pub prev_hdr: wgpu::TextureView,
    pub prev_hdr_tex: wgpu::Texture,
    /// Opaque objects only (the buoy), for water refraction: depth and albedo.
    pub opaque_depth: wgpu::TextureView,
    pub opaque_albedo: wgpu::TextureView,
}

fn target(d: &wgpu::Device, label: &str, extent: (u32, u32), format: wgpu::TextureFormat) -> wgpu::TextureView {
    texture(d, label, extent, format, wgpu::TextureUsages::empty()).create_view(&Default::default())
}

fn texture(
    d: &wgpu::Device,
    label: &str,
    (w, h): (u32, u32),
    format: wgpu::TextureFormat,
    extra: wgpu::TextureUsages,
) -> wgpu::Texture {
    d.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | extra,
        view_formats: &[],
    })
}

impl GBuffer {
    pub fn new(d: &wgpu::Device, extent: (u32, u32)) -> GBuffer {
        let hdr_tex = texture(d, "hdr", extent, HDR_FORMAT, wgpu::TextureUsages::COPY_SRC);
        let prev_hdr_tex = texture(d, "prev hdr", extent, HDR_FORMAT, wgpu::TextureUsages::COPY_DST);
        GBuffer {
            extent,
            depth: target(d, "gbuffer depth", extent, DEPTH_FORMAT),
            normal: target(d, "gbuffer normal", extent, NORMAL_FORMAT),
            material: target(d, "gbuffer material", extent, MATERIAL_FORMAT),
            motion: target(d, "gbuffer motion", extent, MOTION_FORMAT),
            hdr: hdr_tex.create_view(&Default::default()),
            hdr_tex,
            prev_hdr: prev_hdr_tex.create_view(&Default::default()),
            prev_hdr_tex,
            opaque_depth: target(d, "opaque depth", extent, DEPTH_FORMAT),
            opaque_albedo: target(d, "opaque albedo", extent, MATERIAL_FORMAT),
        }
    }
}

impl GBuffer {
    /// Keeps this frame's HDR for next frame's screen-space reflections.
    pub fn record_history(&self, enc: &mut wgpu::CommandEncoder) {
        enc.copy_texture_to_texture(
            self.hdr_tex.as_image_copy(),
            self.prev_hdr_tex.as_image_copy(),
            wgpu::Extent3d { width: self.extent.0, height: self.extent.1, depth_or_array_layers: 1 },
        );
    }
}
