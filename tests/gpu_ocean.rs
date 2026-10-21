//! The GPU FFT ocean must reproduce the CPU reference field. Headless, no window.

use ocean::gpu::adapter::GpuContext;
use ocean::gpu::shaders::ShaderLibrary;
use ocean::params::OceanParams;
use ocean::waves::cpu_ocean::CpuOcean;
use ocean::waves::gpu_ocean::GpuOcean;
use ocean::waves::spectrum::build_spectrum;

#[test]
fn gpu_matches_cpu_reference() {
    let ctx = GpuContext::new_headless().expect("gpu");
    let p = OceanParams::default();
    let spectrum = build_spectrum(&p, 42);
    let cpu = CpuOcean::new(&spectrum, p.choppiness);
    let mut gpu = GpuOcean::new(&ctx, &ShaderLibrary::embedded(), &spectrum);
    let t = 3.7;
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    gpu.update(&mut enc, t, p.choppiness, ocean::waves::foam::foam_params(&p), 0.0);
    ctx.queue.submit([enc.finish()]);

    for c in 0..4 {
        let want = cpu.evaluate(c, t);
        let disp = gpu.read_displacement(&ctx, c as u32);
        let der = gpu.read_derivatives(&ctx, c as u32);
        let max_h = want.disp.iter().map(|d| d[1].abs()).fold(0.0, f32::max);
        assert!(max_h > 1e-4, "cascade {c} is empty");
        let mut worst = 0.0f32;
        for i in 0..disp.len() {
            let (g, w) = (disp[i], want.disp[i]);
            worst = worst
                .max((g[0] - w[0]).abs())
                .max((g[1] - w[1]).abs())
                .max((g[2] - w[2]).abs());
            worst = worst.max((g[3] - want.derivs[i][2]).abs());
            let d = der[i];
            let ws = [want.slope[i][0], want.slope[i][1], want.derivs[i][0], want.derivs[i][1]];
            for k in 0..4 {
                worst = worst.max((d[k] - ws[k]).abs());
            }
        }
        assert!(worst < 2e-3, "cascade {c}: max abs diff {worst} (max height {max_h})");
    }
}

#[test]
fn foam_accumulates_in_strong_wind_and_not_in_calm() {
    let ctx = GpuContext::new_headless().expect("gpu");
    let frac = |wind: f32| {
        let p = OceanParams {
            wind_speed: wind,
            ..Default::default()
        };
        let mut gpu = GpuOcean::new(&ctx, &ShaderLibrary::embedded(), &build_spectrum(&p, 42));
        for i in 0..120 {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            gpu.update(
                &mut enc,
                i as f32 / 60.0,
                p.choppiness,
                ocean::waves::foam::foam_params(&p),
                1.0 / 60.0,
            );
            ctx.queue.submit([enc.finish()]);
        }
        let foam = gpu.read_foam(&ctx);
        foam.iter().filter(|&&f| f > 0.1).count() as f32 / foam.len() as f32
    };
    let (calm, rough) = (frac(5.0), frac(15.0));
    eprintln!("foam coverage: calm {calm} rough {rough}");
    assert!(rough > 0.005, "rough sea has too little foam: {rough}");
    assert!(calm < 0.01, "calm sea has too much foam: {calm}");
    assert!(rough > 5.0 * calm);
}
