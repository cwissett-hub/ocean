use std::sync::Arc;

use anyhow::{Context, anyhow};
use winit::window::Window;

/// What `select_adapter` needs to know about one adapter. Built from `wgpu::Adapter` so the
/// choice itself is a pure function.
#[derive(Debug, Clone)]
pub struct AdapterCandidate {
    pub name: String,
    pub device_type: wgpu::DeviceType,
    pub backend: wgpu::Backend,
    pub ray_query: bool,
}

#[derive(Debug, Clone)]
pub struct AdapterChoice {
    pub index: usize,
    pub rt_available: bool,
    /// Human sentence for the overlay explaining the choice.
    pub reason: String,
}

fn type_rank(t: wgpu::DeviceType) -> u32 {
    match t {
        wgpu::DeviceType::DiscreteGpu => 4,
        wgpu::DeviceType::IntegratedGpu => 3,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 1,
        wgpu::DeviceType::Cpu => 0,
    }
}

fn backend_rank(c: &AdapterCandidate) -> u32 {
    match (c.backend, c.ray_query) {
        (wgpu::Backend::Vulkan, true) => 2,
        (wgpu::Backend::Vulkan, false) => 1,
        _ => 0,
    }
}

/// Prefers the most capable device type first (so a discrete GPU beats an integrated one even
/// when both report ray query), then Vulkan with ray query within that type.
pub fn select_adapter(c: &[AdapterCandidate]) -> Option<AdapterChoice> {
    let (index, best) = c
        .iter()
        .enumerate()
        .max_by_key(|(i, a)| (type_rank(a.device_type), backend_rank(a), std::cmp::Reverse(*i)))?;
    let kind = match best.device_type {
        wgpu::DeviceType::DiscreteGpu => "discrete",
        wgpu::DeviceType::IntegratedGpu => "integrated",
        wgpu::DeviceType::VirtualGpu => "virtual",
        wgpu::DeviceType::Cpu => "software",
        wgpu::DeviceType::Other => "unknown",
    };
    let rt = if best.ray_query {
        "hardware ray query available".to_string()
    } else {
        format!("no ray query on {:?}, using screen-space reflections", best.backend)
    };
    Some(AdapterChoice {
        index,
        rt_available: best.ray_query,
        reason: format!("{} ({kind} GPU, {:?}): {rt}", best.name, best.backend),
    })
}

/// The device everything renders with.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub rt_available: bool,
    pub timestamps: bool,
    pub reason: String,
}

impl GpuContext {
    /// A device with no window, for compute tests.
    pub fn new_headless() -> anyhow::Result<GpuContext> {
        pollster::block_on(Self::create(None)).map(|(ctx, _)| ctx)
    }

    pub fn new_for_surface(window: Arc<Window>) -> anyhow::Result<(GpuContext, wgpu::Surface<'static>)> {
        let (ctx, surface) = pollster::block_on(Self::create(Some(window)))?;
        Ok((ctx, surface.expect("surface requested")))
    }

    async fn create(window: Option<Arc<Window>>) -> anyhow::Result<(GpuContext, Option<wgpu::Surface<'static>>)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = match window {
            Some(w) => Some(instance.create_surface(w).context("create window surface")?),
            None => None,
        };
        let mut adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN).await;
        if adapters.is_empty() {
            adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        }
        if let Some(s) = &surface {
            adapters.retain(|a| a.is_surface_supported(s));
        }
        let candidates: Vec<AdapterCandidate> = adapters
            .iter()
            .map(|a| {
                let info = a.get_info();
                AdapterCandidate {
                    name: info.name,
                    device_type: info.device_type,
                    backend: info.backend,
                    ray_query: a.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
                }
            })
            .collect();
        let choice = select_adapter(&candidates).ok_or_else(|| anyhow!("no GPU adapter found"))?;
        let adapter = adapters.swap_remove(choice.index);

        let wanted = wgpu::Features::EXPERIMENTAL_RAY_QUERY
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
            | wgpu::Features::FLOAT32_FILTERABLE;
        let features = adapter.features() & wanted;
        let al = adapter.limits();
        let limits = wgpu::Limits {
            max_blas_primitive_count: al.max_blas_primitive_count,
            max_blas_geometry_count: al.max_blas_geometry_count,
            max_tlas_instance_count: al.max_tlas_instance_count,
            max_acceleration_structures_per_shader_stage: al.max_acceleration_structures_per_shader_stage,
            max_storage_textures_per_shader_stage: al.max_storage_textures_per_shader_stage.min(16),
            max_storage_buffers_per_shader_stage: al.max_storage_buffers_per_shader_stage.min(16),
            max_compute_workgroup_storage_size: al.max_compute_workgroup_storage_size.min(32768),
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ocean"),
                required_features: features,
                required_limits: limits,
                // SAFETY: we opt in to wgpu's experimental ray-query API knowingly.
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("request device")?;
        let ctx = GpuContext {
            instance,
            adapter_name: adapter.get_info().name,
            rt_available: features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
            timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
            reason: choice.reason,
            adapter,
            device,
            queue,
        };
        Ok((ctx, surface))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::Backend::{Dx12, Vulkan};
    use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu};

    fn cand(name: &str, device_type: wgpu::DeviceType, backend: wgpu::Backend, ray_query: bool) -> AdapterCandidate {
        AdapterCandidate {
            name: name.to_string(),
            device_type,
            backend,
            ray_query,
        }
    }

    const NV: &str = "NVIDIA RTX 2000 Ada Generation Laptop GPU";
    const ARC: &str = "Intel(R) Arc(TM) Pro Graphics";

    #[test]
    fn picks_nvidia_vulkan_over_intel_with_ray_query() {
        let c = vec![
            cand(ARC, IntegratedGpu, Vulkan, true),
            cand(NV, DiscreteGpu, Vulkan, true),
            cand(NV, DiscreteGpu, Dx12, false),
        ];
        let ch = select_adapter(&c).unwrap();
        assert_eq!(ch.index, 1);
        assert!(ch.rt_available);
    }

    #[test]
    fn discrete_without_ray_query_still_beats_integrated() {
        let c = vec![
            cand(ARC, IntegratedGpu, Vulkan, true),
            cand(NV, DiscreteGpu, Vulkan, false),
        ];
        let ch = select_adapter(&c).unwrap();
        assert_eq!(ch.index, 1);
        assert!(!ch.rt_available);
        assert!(ch.reason.contains("ray query"), "{}", ch.reason);
    }

    #[test]
    fn integrated_only_falls_back_with_reason() {
        let c = vec![cand(ARC, IntegratedGpu, Vulkan, true)];
        let ch = select_adapter(&c).unwrap();
        assert_eq!(ch.index, 0);
        assert!(ch.rt_available);
        assert!(ch.reason.contains("integrated"), "{}", ch.reason);
    }

    #[test]
    fn cpu_adapter_never_chosen_when_gpu_exists() {
        let c = vec![
            cand("Microsoft Basic Render Driver", Cpu, Dx12, false),
            cand(ARC, IntegratedGpu, Dx12, false),
        ];
        assert_eq!(select_adapter(&c).unwrap().index, 1);
    }

    #[test]
    fn empty_list_is_none() {
        assert!(select_adapter(&[]).is_none());
    }
}
