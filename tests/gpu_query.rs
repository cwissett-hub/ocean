//! The GPU wave query must agree with the CPU reference surface. Headless, no window.

use glam::{Vec2, Vec3};
use ocean::gpu::adapter::GpuContext;
use ocean::gpu::shaders::ShaderLibrary;
use ocean::params::OceanParams;
use ocean::waves::cpu_ocean::{CpuOcean, OceanField};
use ocean::waves::foam::foam_params;
use ocean::waves::gpu_ocean::GpuOcean;
use ocean::waves::query::{QUERY_CASCADES, WaveQuery, invert_horizontal};
use ocean::waves::spectrum::{CASCADE_SIZES_M, N, build_spectrum};

/// Bilinear sample of a cascade field at world xz, with the renderer's texel convention
/// (texel (x, z) holds world point (x, z)·L/N) and repeat wrapping.
fn sample(f: &OceanField, c: usize, p: Vec2) -> Vec3 {
    let t = p * (N as f32 / CASCADE_SIZES_M[c]);
    let (x0, z0) = (t.x.floor(), t.y.floor());
    let (fx, fz) = (t.x - x0, t.y - z0);
    let at = |x: f32, z: f32| {
        let (xi, zi) = (
            (x as i64).rem_euclid(N as i64) as usize,
            (z as i64).rem_euclid(N as i64) as usize,
        );
        Vec3::from(f.disp[zi * N + xi])
    };
    let a = at(x0, z0).lerp(at(x0 + 1.0, z0), fx);
    let b = at(x0, z0 + 1.0).lerp(at(x0 + 1.0, z0 + 1.0), fx);
    a.lerp(b, fz)
}

#[test]
fn query_matches_rendered_surface() {
    let ctx = GpuContext::new_headless().expect("gpu");
    let lib = ShaderLibrary::embedded();
    let p = OceanParams {
        wind_speed: 9.0,
        ..Default::default()
    };
    let spectrum = build_spectrum(&p, 42);
    let t = 5.3;
    let mut gpu = GpuOcean::new(&ctx, &lib, &spectrum);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    gpu.update(&mut enc, t, p.choppiness, foam_params(&p), 0.0);
    ctx.queue.submit([enc.finish()]);

    let cpu = CpuOcean::new(&spectrum, p.choppiness);
    let fields: Vec<_> = (0..QUERY_CASCADES).map(|c| cpu.evaluate(c, t)).collect();
    let disp_at = |q: Vec2| {
        (0..QUERY_CASCADES)
            .map(|c| sample(&fields[c], c, q))
            .fold(Vec3::ZERO, |a, b| a + b)
    };

    let points: Vec<Vec2> = (0..32)
        .map(|i| Vec2::new(i as f32 * 13.7 - 200.0, (i * i) as f32 * 0.91 - 150.0))
        .collect();
    let mut query = WaveQuery::new(&ctx, &lib, &gpu);
    let got = query.read_blocking(&ctx, &points);
    assert_eq!(got.len(), points.len());
    let mut worst = 0.0f32;
    for (q, (h, n)) in points.iter().zip(&got) {
        let src = invert_horizontal(*q, disp_at);
        let want = disp_at(src).y;
        worst = worst.max((h - want).abs());
        assert!((n.length() - 1.0).abs() < 1e-3 && n.y > 0.3, "normal {n}");
    }
    assert!(worst < 0.01, "worst height error {worst} m");
}
