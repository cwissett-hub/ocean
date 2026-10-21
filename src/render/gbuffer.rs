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
}

fn target(d: &wgpu::Device, label: &str, (w, h): (u32, u32), format: wgpu::TextureFormat) -> wgpu::TextureView {
    d.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
    .create_view(&Default::default())
}

impl GBuffer {
    pub fn new(d: &wgpu::Device, extent: (u32, u32)) -> GBuffer {
        GBuffer {
            extent,
            depth: target(d, "gbuffer depth", extent, DEPTH_FORMAT),
            normal: target(d, "gbuffer normal", extent, NORMAL_FORMAT),
            material: target(d, "gbuffer material", extent, MATERIAL_FORMAT),
            motion: target(d, "gbuffer motion", extent, MOTION_FORMAT),
            hdr: target(d, "hdr", extent, HDR_FORMAT),
        }
    }
}
