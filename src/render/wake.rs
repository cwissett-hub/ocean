//! The raft's own waves: Tessendorf's iWave on a 64 m grid that follows the raft. The hull is a
//! moving pressure patch; the waves it makes propagate dispersively (ω² = g·k, through a
//! convolution kernel approximating √(−∇²)), so a moving raft trails a real Kelvin wake and
//! heaps water at its bow. `wake.wgsl` steps it; the ocean surface adds its height.

use glam::{IVec2, Vec2};

/// Grid cells per side and their size, m (64 m × 64 m around the raft).
pub const WAKE_RES: i32 = 256;
pub const WAKE_CELL_M: f32 = 0.25;
/// Kernel half-width, cells (13 × 13 taps).
pub const KERNEL_RADIUS: i32 = 6;
pub const KERNEL_SIZE: usize = (2 * KERNEL_RADIUS as usize + 1) * (2 * KERNEL_RADIUS as usize + 1);

/// The vertical-derivative kernel G (row-major, centre at KERNEL_RADIUS): convolving a plane
/// wave of wavenumber k (1/cell) with it returns about |k| times the wave.
pub fn iwave_kernel() -> Vec<f32> {
    // Radially symmetric: one unknown per (i ≥ j ≥ 0) offset class, fitted by least squares
    // so the response matches |k| over 2–32-cell waves in several directions, with zero
    // response to a flat surface (heavily weighted).
    let r = KERNEL_RADIUS;
    let classes: Vec<(i32, i32)> = (0..=r).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
    let orbit = |(a, b): (i32, i32)| {
        let mut o = Vec::new();
        for (x, y) in [(a, b), (b, a)] {
            for (sx, sy) in [(1, 1), (-1, 1), (1, -1), (-1, -1)] {
                let p = (sx * x, sy * y);
                if !o.contains(&p) {
                    o.push(p);
                }
            }
        }
        o
    };
    let orbits: Vec<Vec<(i32, i32)>> = classes.iter().map(|&c| orbit(c)).collect();
    let row = |k: f64, a: f64| -> Vec<f64> {
        orbits
            .iter()
            .map(|o| {
                o.iter()
                    .map(|&(x, y)| (k * (a.cos() * x as f64 + a.sin() * y as f64)).cos())
                    .sum()
            })
            .collect()
    };
    let n = classes.len();
    let mut ata = vec![vec![0.0f64; n]; n];
    let mut atb = vec![0.0f64; n];
    let mut add = |row: Vec<f64>, target: f64, weight: f64| {
        for i in 0..n {
            atb[i] += weight * row[i] * target;
            for j in 0..n {
                ata[i][j] += weight * row[i] * row[j];
            }
        }
    };
    for step in 0..60 {
        let lambda = 2.0 * 16f64.powf(step as f64 / 59.0); // 2..32 cells
        let k = std::f64::consts::TAU / lambda;
        for a in 0..6 {
            let angle = a as f64 * std::f64::consts::FRAC_PI_4 / 5.0;
            add(row(k, angle), k, 1.0);
        }
    }
    add(row(0.0, 0.0), 0.0, 1e4);
    for (i, r) in ata.iter_mut().enumerate() {
        r[i] += 1e-9;
    }
    // Gaussian elimination with partial pivoting.
    for c in 0..n {
        let p = (c..n)
            .max_by(|&a, &b| ata[a][c].abs().total_cmp(&ata[b][c].abs()))
            .unwrap();
        ata.swap(c, p);
        atb.swap(c, p);
        for r in c + 1..n {
            let f = ata[r][c] / ata[c][c];
            let pivot = ata[c].clone();
            for (a, p) in ata[r][c..].iter_mut().zip(&pivot[c..]) {
                *a -= f * p;
            }
            atb[r] -= f * atb[c];
        }
    }
    let mut g = vec![0.0f64; n];
    for c in (0..n).rev() {
        let s: f64 = (c + 1..n).map(|k| ata[c][k] * g[k]).sum();
        g[c] = (atb[c] - s) / ata[c][c];
    }
    let side = 2 * r + 1;
    let mut out = vec![0.0f32; KERNEL_SIZE];
    for (o, &w) in orbits.iter().zip(&g) {
        for &(x, y) in o {
            out[((y + r) * side + x + r) as usize] = w as f32;
        }
    }
    out
}

/// Wave damping, 1/s (wakes fade over ~10 s).
pub const WAKE_DAMPING: f32 = 0.25;

/// Pressure head of the hull at a cell centre (mirrors `hull` in wake.wgsl): a soft-edged box
/// the raft's size, centred on `raft` and turned to `heading` (compass rad).
pub fn hull_pressure(p: Vec2, raft: Vec2, heading: f32) -> f32 {
    let d = p - raft;
    let (s, c) = heading.sin_cos();
    let local = Vec2::new(c * d.x - s * d.y, s * d.x + c * d.y);
    let edge = 0.6;
    let soft = |half: f32, x: f32| {
        let t = ((x.abs() - half) / (half - edge - half)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    // Along the hull: a sharp entry at the bow (+z) and a long run-out to the stern, as a real
    // hull makes a strong bow wave and a weak stern wave (a box made two equal V wakes).
    let l = HULL_HALF_SIZE.y;
    let ramp = |e0: f32, e1: f32, x: f32| {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let along = ramp(l, l - edge, local.y) * ramp(-l, 0.3 * l, local.y);
    HULL_PRESSURE_M * soft(HULL_HALF_SIZE.x, local.x) * along
}

/// One iWave step on a periodic `n`×`n` grid (mirrors `wake.wgsl`): `h` becomes the new height
/// and `prev` the old one. `source` is the pressure head (m of water) pushed down this step.
pub fn iwave_step(h: &mut [f32], prev: &mut [f32], source: &[f32], kernel: &[f32], n: usize, dt: f32) {
    let r = KERNEL_RADIUS;
    let side = (2 * r + 1) as usize;
    let a = WAKE_DAMPING * dt;
    let g = crate::waves::spectrum::G;
    let mut next = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            let mut vd = 0.0;
            for j in -r..=r {
                for i in -r..=r {
                    let xx = (x as i32 + i).rem_euclid(n as i32) as usize;
                    let yy = (y as i32 + j).rem_euclid(n as i32) as usize;
                    vd += kernel[(j + r) as usize * side + (i + r) as usize] * (h[yy * n + xx] + source[yy * n + xx]);
                }
            }
            let c = y * n + x;
            // Verlet with damping on the velocity (h − prev), not the height.
            next[c] = h[c] * (2.0 - a) - prev[c] * (1.0 - a) - g * dt * dt * vd / WAKE_CELL_M;
        }
    }
    prev.copy_from_slice(h);
    h.copy_from_slice(&next);
}

/// Lower-left grid cell of the window centred on `focus` (local, m).
pub fn window_origin(focus: Vec2) -> IVec2 {
    (focus / WAKE_CELL_M).floor().as_ivec2() - IVec2::splat(WAKE_RES / 2)
}

/// Pressure head under the hull, m (about the logs' draft) and the hull's half beam / length.
pub const HULL_PRESSURE_M: f32 = 0.35;
pub const HULL_HALF_SIZE: Vec2 = Vec2::new(1.25, 1.55);
const STEP_S: f32 = 1.0 / 60.0;
const MAX_STEPS_PER_FRAME: u32 = 4;
const STATE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WakeParams {
    origin: [i32; 2],
    prev_origin: [i32; 2],
    raft: [f32; 2],
    heading: f32,
    dt: f32,
    pressure: f32,
    damping: f32,
    half_size: [f32; 2],
}

fn state_texture(d: &wgpu::Device, label: &str, usage: wgpu::TextureUsages) -> wgpu::Texture {
    d.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: WAKE_RES as u32,
            height: WAKE_RES as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: STATE_FORMAT,
        usage,
        view_formats: &[],
    })
}

/// The GPU simulation: two state textures ping-pong; the newest is copied into `output` (the
/// texture the surface shaders bind) after each frame's steps.
pub struct WakeSim {
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    state: [wgpu::Texture; 2],
    groups: [wgpu::BindGroup; 2],
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
    current: usize,
    origin: Option<IVec2>,
    accumulated: f32,
}

impl WakeSim {
    pub fn new(d: &wgpu::Device, queue: &wgpu::Queue, lib: &crate::gpu::shaders::ShaderLibrary) -> WakeSim {
        use wgpu::TextureUsages as T;
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wake"),
            source: wgpu::ShaderSource::Wgsl(lib.source("wake").expect("shader source").into()),
        });
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wake"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wake params"),
            size: std::mem::size_of::<WakeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        use wgpu::util::DeviceExt;
        let kernel = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wake kernel"),
            contents: bytemuck::cast_slice(&iwave_kernel()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let usage = T::STORAGE_BINDING | T::TEXTURE_BINDING | T::COPY_SRC;
        let state = [
            state_texture(d, "wake state a", usage),
            state_texture(d, "wake state b", usage),
        ];
        let output = state_texture(d, "wake", T::TEXTURE_BINDING | T::COPY_DST | T::COPY_SRC);
        // Start flat (textures are zeroed, but be explicit for the output).
        let zeros = vec![0u8; (WAKE_RES * WAKE_RES * 8) as usize];
        queue.write_texture(
            output.as_image_copy(),
            &zeros,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WAKE_RES as u32 * 8),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: WAKE_RES as u32,
                height: WAKE_RES as u32,
                depth_or_array_layers: 1,
            },
        );
        let views: Vec<wgpu::TextureView> = state.iter().map(|t| t.create_view(&Default::default())).collect();
        let groups = std::array::from_fn(|i| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wake"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&views[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&views[1 - i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: kernel.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            })
        });
        let output_view = output.create_view(&Default::default());
        WakeSim {
            pipeline,
            params,
            state,
            groups,
            output,
            output_view,
            current: 0,
            origin: None,
            accumulated: 0.0,
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// The newest heights (r) and previous heights (g).
    pub fn texture(&self) -> &wgpu::Texture {
        &self.output
    }

    /// Shader parameters: window origin cell (x, z), cell size, resolution (0 = none yet).
    pub fn params(&self) -> [f32; 4] {
        match self.origin {
            Some(o) => [o.x as f32, o.y as f32, WAKE_CELL_M, WAKE_RES as f32],
            None => [0.0; 4],
        }
    }

    /// Follows an origin rebase: local cells move by a whole number of grid periods, so only
    /// the window's coordinates change.
    pub fn rebase(&mut self, shift: Vec2) {
        let cells = (shift / WAKE_CELL_M).round().as_ivec2();
        if let Some(o) = &mut self.origin {
            *o -= cells;
        }
    }

    /// Steps the simulation `dt` seconds with the hull at `raft` (local m) heading `heading`.
    pub fn record(&mut self, enc: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, raft: Vec2, heading: f32, dt: f32) {
        self.accumulated = (self.accumulated + dt.max(0.0)).min(STEP_S * MAX_STEPS_PER_FRAME as f32);
        let steps = (self.accumulated / STEP_S) as u32;
        if steps == 0 {
            return;
        }
        self.accumulated -= steps as f32 * STEP_S;
        let origin = window_origin(raft);
        let prev = self.origin.unwrap_or(origin);
        let p = WakeParams {
            origin: origin.to_array(),
            prev_origin: prev.to_array(),
            raft: raft.to_array(),
            heading,
            dt: STEP_S,
            pressure: HULL_PRESSURE_M,
            damping: WAKE_DAMPING,
            half_size: HULL_HALF_SIZE.to_array(),
        };
        // One parameter set for this frame's steps: cells entering the window read as flat in
        // each (they are at its edge, far from the hull, so that is what they are anyway).
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        for _ in 0..steps {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wake"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.groups[self.current], &[]);
            let g = (WAKE_RES as u32).div_ceil(8);
            pass.dispatch_workgroups(g, g, 1);
            drop(pass);
            self.current = 1 - self.current;
        }
        enc.copy_texture_to_texture(
            self.state[self.current].as_image_copy(),
            self.output.as_image_copy(),
            wgpu::Extent3d {
                width: WAKE_RES as u32,
                height: WAKE_RES as u32,
                depth_or_array_layers: 1,
            },
        );
        self.origin = Some(origin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Response of the kernel to a plane wave cos(k·x) along `dir`, at the origin.
    fn response(g: &[f32], k: f32, dir: Vec2) -> f32 {
        let n = 2 * KERNEL_RADIUS + 1;
        let mut s = 0.0;
        for j in -KERNEL_RADIUS..=KERNEL_RADIUS {
            for i in -KERNEL_RADIUS..=KERNEL_RADIUS {
                let w = g[((j + KERNEL_RADIUS) * n + i + KERNEL_RADIUS) as usize];
                s += w * (k * dir.dot(Vec2::new(i as f32, j as f32))).cos();
            }
        }
        s
    }

    #[test]
    fn kernel_is_the_vertical_derivative_for_resolved_waves() {
        let g = iwave_kernel();
        assert_eq!(g.len(), KERNEL_SIZE);
        // Waves from 3 to 16 cells long (0.75–4 m): response within 12 % of |k|, any direction.
        for lambda in [3.0f32, 4.0, 6.0, 10.0, 16.0] {
            let k = std::f32::consts::TAU / lambda;
            for a in [0.0f32, 0.4, 0.785] {
                let r = response(&g, k, Vec2::new(a.cos(), a.sin()));
                assert!((r / k - 1.0).abs() < 0.12, "λ {lambda} cells, angle {a}: {r} vs {k}");
            }
        }
        // A flat surface makes no waves.
        assert!(g.iter().sum::<f32>().abs() < 0.02, "sum {}", g.iter().sum::<f32>());
    }

    #[test]
    fn waves_run_at_deep_water_phase_speed() {
        // A 2 m plane wave (8 cells) on a periodic 32-cell strip: after T seconds its crest has
        // moved c·T with c = √(g/k).
        let n = 32;
        let lambda = 8.0 * WAKE_CELL_M;
        let k = std::f32::consts::TAU / lambda;
        let c = (crate::waves::spectrum::G / k).sqrt();
        let w = c * k;
        let dt = 1.0 / 60.0;
        let wave = |t: f32| -> Vec<f32> {
            (0..n * n)
                .map(|i| 0.01 * (k * (i % n) as f32 * WAKE_CELL_M - w * t).cos())
                .collect()
        };
        let (mut prev, mut h) = (wave(-dt), wave(0.0));
        let src = vec![0.0; n * n];
        let g = iwave_kernel();
        let steps = 60;
        for _ in 0..steps {
            iwave_step(&mut h, &mut prev, &src, &g, n, dt);
        }
        // Phase of the result along x, by projection onto cos/sin.
        let (mut cs, mut sn) = (0.0f32, 0.0f32);
        for (x, v) in h.iter().take(n).enumerate() {
            cs += v * (k * x as f32 * WAKE_CELL_M).cos();
            sn += v * (k * x as f32 * WAKE_CELL_M).sin();
        }
        let travelled = sn.atan2(cs) / k; // crest position after T
        let want = (c * steps as f32 * dt).rem_euclid(lambda);
        let err = (travelled.rem_euclid(lambda) - want)
            .abs()
            .min(lambda - (travelled.rem_euclid(lambda) - want).abs());
        assert!(err < 0.1 * lambda, "crest at {travelled}, wanted {want} (λ {lambda})");
        let amp = (cs * cs + sn * sn).sqrt() * 2.0 / n as f32;
        assert!((0.008..0.0105).contains(&amp), "amplitude {amp}");
    }

    #[test]
    fn window_follows_the_focus_in_whole_cells() {
        let o = window_origin(Vec2::new(10.1, -3.3));
        assert_eq!(o, IVec2::new(40 - 128, -14 - 128));
        let c = (o.as_vec2() + WAKE_RES as f32 / 2.0) * WAKE_CELL_M;
        assert!((c - Vec2::new(10.1, -3.3)).abs().max_element() <= WAKE_CELL_M);
    }
}
