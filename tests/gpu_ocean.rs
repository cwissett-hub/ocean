//! The GPU FFT ocean must reproduce the CPU reference field. Headless, no window.

use ocean::gpu::adapter::GpuContext;
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
    let mut gpu = GpuOcean::new(&ctx, &spectrum);
    let t = 3.7;
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    gpu.update(&mut enc, t, p.choppiness);
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
            worst = worst.max((g[0] - w[0]).abs()).max((g[1] - w[1]).abs()).max((g[2] - w[2]).abs());
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
