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
    let map = ocean::render::seabed::SeabedMap::new(&ctx.device);
    let mut query = WaveQuery::new(&ctx, &lib, &gpu, &map);
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

#[test]
fn query_attenuates_like_the_surface() {
    use glam::DVec2;
    use ocean::render::seabed::{SEABED_RES, SEABED_TEXEL_M, SeabedMap, texel_center};
    use ocean::world::islands::wave_attenuation;
    use ocean::world::origin::WorldOrigin;

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

    // Open sea 25 m deep with a 1 m-deep shoal 400 m square around the origin.
    let corner = DVec2::splat(-(SEABED_RES as f64) * SEABED_TEXEL_M / 2.0);
    let mut heights = vec![-25.0f32; SEABED_RES * SEABED_RES];
    for j in 0..SEABED_RES {
        for i in 0..SEABED_RES {
            if texel_center(corner, i, j).abs().max_element() < 200.0 {
                heights[j * SEABED_RES + i] = -1.0;
            }
        }
    }
    let mut map = SeabedMap::new(&ctx.device);
    map.upload(&ctx.queue, corner, &heights);
    let sp = map.params(&WorldOrigin::default());

    let points: Vec<Vec2> = (0..32)
        .map(|i| Vec2::new(i as f32 * 5.7 - 90.0, (i * i) as f32 * 0.17 - 80.0))
        .collect();
    let mut query = WaveQuery::new(&ctx, &lib, &gpu, &map);
    query.set_seabed(sp);
    let shoal = query.read_blocking(&ctx, &points);
    query.set_seabed([0.0; 4]);
    let open = query.read_blocking(&ctx, &points);

    let cpu = CpuOcean::new(&spectrum, p.choppiness);
    let fields: Vec<_> = (0..QUERY_CASCADES).map(|c| cpu.evaluate(c, t)).collect();
    let a = wave_attenuation(1.0);
    let disp_at = |q: Vec2| {
        let d = (0..QUERY_CASCADES)
            .map(|c| sample(&fields[c], c, q))
            .fold(Vec3::ZERO, |a, b| a + b);
        d * if q.abs().max_element() < 200.0 { a } else { 1.0 }
    };
    let mut compared = 0;
    for (q, ((hs, _), (ho, _))) in points.iter().zip(shoal.iter().zip(&open)) {
        let want = disp_at(invert_horizontal(*q, disp_at)).y;
        assert!((hs - want).abs() < 0.01, "shoal height {hs} vs cpu {want} at {q}");
        if ho.abs() > 0.05 {
            compared += 1;
            assert!(hs.abs() <= 0.15 * ho.abs(), "shoal {hs} vs open {ho} at {q}");
        }
    }
    assert!(compared >= 8, "only {compared} points had waves to compare");
}

#[test]
fn results_carry_the_time_they_sample() {
    // Results can arrive a frame or two late and two at once; anything differencing them over
    // time (water vertical velocity) needs the time the waves were sampled, not arrival time.
    let ctx = GpuContext::new_headless().expect("gpu");
    let lib = ShaderLibrary::embedded();
    let spectrum = build_spectrum(&OceanParams::default(), 42);
    let gpu = GpuOcean::new(&ctx, &lib, &spectrum);
    let map = ocean::render::seabed::SeabedMap::new(&ctx.device);
    let mut query = WaveQuery::new(&ctx, &lib, &gpu, &map);
    for t in [7.25f32, 7.5] {
        query.request(&[Vec2::new(1.0, 2.0)], t);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        query.record(&mut enc);
        ctx.queue.submit([enc.finish()]);
        query.after_submit();
    }
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let r = query.poll(&ctx.device).expect("results");
    assert_eq!(r.time, 7.5, "newest result, stamped with its sample time");
    assert_eq!(r.samples.len(), 1);
}
