//! CPU reference evaluation of the ocean field with rustfft. Used by tests to check the GPU
//! and the wave query; never on the per-frame path.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

use crate::waves::spectrum::{N, SpectrumSet, wavevector};

/// One cascade's field on its N×N grid (row-major, texel (x, z) at world (x, z)·L/N).
pub struct OceanField {
    /// (Dx, height, Dz) in metres.
    pub disp: Vec<[f32; 3]>,
    /// (∂height/∂x, ∂height/∂z).
    pub slope: Vec<[f32; 2]>,
    /// (∂Dx/∂x, ∂Dz/∂z, ∂Dx/∂z): the horizontal displacement gradient. Cascades add these,
    /// then the Jacobian is formed from the sums.
    pub derivs: Vec<[f32; 3]>,
    /// Determinant of this cascade alone; < 1 compresses, < 0 folds.
    pub jacobian: Vec<f32>,
    /// Largest imaginary residue of the height transform (should be ~0).
    pub max_imag: f32,
}

pub struct CpuOcean {
    spectrum: SpectrumSet,
    choppiness: f32,
}

const FIELDS: usize = 8; // height, Dx, Dz, sx, sz, Jxx, Jzz, Jxz

impl CpuOcean {
    pub fn new(spectrum: &SpectrumSet, choppiness: f32) -> Self {
        CpuOcean { spectrum: spectrum.clone(), choppiness }
    }

    pub fn evaluate(&self, cascade: usize, t: f32) -> OceanField {
        let lambda = self.choppiness;
        let h0 = &self.spectrum.h0[cascade];
        let om = &self.spectrum.omega[cascade];
        let mut f = vec![vec![Complex32::new(0.0, 0.0); N * N]; FIELDS];
        let i = Complex32::new(0.0, 1.0);
        for z in 0..N {
            for x in 0..N {
                let idx = z * N + x;
                let (kx, kz) = wavevector(cascade, x, z);
                let k = (kx * kx + kz * kz).sqrt();
                if k == 0.0 {
                    continue;
                }
                let phase = ((om[idx] as f64 * t as f64) % std::f64::consts::TAU) as f32;
                let e = Complex32::from_polar(1.0, phase);
                let [a, b, c, d] = h0[idx];
                let h = Complex32::new(a, b) * e + Complex32::new(c, d) * e.conj();
                f[0][idx] = h;
                f[1][idx] = -i * (kx / k) * lambda * h;
                f[2][idx] = -i * (kz / k) * lambda * h;
                f[3][idx] = i * kx * h;
                f[4][idx] = i * kz * h;
                f[5][idx] = (kx * kx / k) * lambda * h;
                f[6][idx] = (kz * kz / k) * lambda * h;
                f[7][idx] = (kx * kz / k) * lambda * h;
            }
        }
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(N);
        let mut col = vec![Complex32::new(0.0, 0.0); N];
        for field in &mut f {
            for row in field.chunks_exact_mut(N) {
                fft.process(row);
            }
            for x in 0..N {
                for z in 0..N {
                    col[z] = field[z * N + x];
                }
                fft.process(&mut col);
                for z in 0..N {
                    // The centred k grid shifts every output texel by (−1)^(x+z).
                    let sign = if (x + z) % 2 == 0 { 1.0 } else { -1.0 };
                    field[z * N + x] = col[z] * sign;
                }
            }
        }
        let max_imag = f[0].iter().map(|c| c.im.abs()).fold(0.0, f32::max);
        OceanField {
            disp: (0..N * N).map(|j| [f[1][j].re, f[0][j].re, f[2][j].re]).collect(),
            slope: (0..N * N).map(|j| [f[3][j].re, f[4][j].re]).collect(),
            derivs: (0..N * N).map(|j| [f[5][j].re, f[6][j].re, f[7][j].re]).collect(),
            jacobian: (0..N * N)
                .map(|j| (1.0 + f[5][j].re) * (1.0 + f[6][j].re) - f[7][j].re * f[7][j].re)
                .collect(),
            max_imag,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::OceanParams;
    use crate::waves::spectrum::{CASCADE_SIZES_M, REPEAT_PERIOD_S, build_spectrum, jonswap};

    fn max_abs_diff(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
        a.iter()
            .zip(b)
            .flat_map(|(p, q)| (0..3).map(move |i| (p[i] - q[i]).abs()))
            .fold(0.0, f32::max)
    }

    fn ocean(p: &OceanParams) -> CpuOcean {
        CpuOcean::new(&build_spectrum(p, 42), p.choppiness)
    }

    #[test]
    fn heightfield_is_real() {
        let o = ocean(&OceanParams::default());
        let f = o.evaluate(1, 3.7);
        let max_h = f.disp.iter().map(|d| d[1].abs()).fold(0.0, f32::max);
        assert!(max_h > 0.0);
        assert!(f.max_imag < 1e-4 * max_h, "imag {} vs height {}", f.max_imag, max_h);
    }

    #[test]
    fn field_variance_matches_spectrum() {
        let p = OceanParams { swell_amount: 0.0, ..Default::default() };
        let o = ocean(&p);
        let var: f32 = (0..4)
            .map(|c| {
                let f = o.evaluate(c, 0.0);
                f.disp.iter().map(|d| d[1] * d[1]).sum::<f32>() / f.disp.len() as f32
            })
            .sum();
        let hs_field = 4.0 * var.sqrt();
        let m0 = crate::waves::spectrum::tests::integrate(
            |w| jonswap(w, p.wind_speed, p.fetch_km * 1000.0),
            0.02,
            40.0,
        );
        let hs_spec = 4.0 * m0.sqrt();
        assert!((hs_field - hs_spec).abs() / hs_spec < 0.2, "field {hs_field} spectrum {hs_spec}");
    }

    #[test]
    fn swell_adds_energy() {
        let calm = ocean(&OceanParams { swell_amount: 0.0, ..Default::default() }).evaluate(0, 0.0);
        let swell = ocean(&OceanParams { swell_amount: 1.0, ..Default::default() }).evaluate(0, 0.0);
        let e = |f: &OceanField| f.disp.iter().map(|d| d[1] * d[1]).sum::<f32>();
        assert!(e(&swell) > 2.0 * e(&calm));
    }

    #[test]
    fn field_repeats_after_period() {
        let o = ocean(&OceanParams::default());
        let a = o.evaluate(0, 12.3);
        let b = o.evaluate(0, 12.3 + REPEAT_PERIOD_S);
        assert!(max_abs_diff(&a.disp, &b.disp) < 1e-4);
    }

    #[test]
    fn field_changes_over_time() {
        let o = ocean(&OceanParams::default());
        assert!(max_abs_diff(&o.evaluate(0, 0.0).disp, &o.evaluate(0, 1.0).disp) > 1e-2);
    }

    #[test]
    fn jacobian_is_one_without_choppiness() {
        let o = ocean(&OceanParams { choppiness: 0.0, ..Default::default() });
        for c in 0..4 {
            let f = o.evaluate(c, 2.0);
            assert!(f.jacobian.iter().all(|j| (j - 1.0).abs() < 1e-5));
            assert!(f.disp.iter().all(|d| d[0] == 0.0 && d[2] == 0.0));
        }
    }

    #[test]
    fn choppiness_folds_some_crests_in_strong_wind() {
        let o = ocean(&OceanParams { wind_speed: 15.0, choppiness: 1.2, ..Default::default() });
        // Sample every cascade at cascade 0's texel centres (4 m is a whole number of texels in
        // each finer grid) and form the Jacobian from the summed gradient, as the GPU does.
        let fields: Vec<_> = (0..4).map(|c| o.evaluate(c, 0.0)).collect();
        let mut min_j = f32::MAX;
        for z in 0..N {
            for x in 0..N {
                let mut d = [0.0f32; 3];
                for (c, f) in fields.iter().enumerate() {
                    let s = CASCADE_SIZES_M[0] / CASCADE_SIZES_M[c];
                    let (cx, cz) = ((x as f32 * s) as usize % N, (z as f32 * s) as usize % N);
                    let g = f.derivs[cz * N + cx];
                    (0..3).for_each(|i| d[i] += g[i]);
                }
                min_j = min_j.min((1.0 + d[0]) * (1.0 + d[1]) - d[2] * d[2]);
            }
        }
        assert!(min_j < 0.5, "min J {min_j}");
    }

    #[test]
    fn slope_matches_finite_difference_of_height() {
        let o = ocean(&OceanParams { choppiness: 0.0, ..Default::default() });
        let f = o.evaluate(0, 1.0);
        // Central differences attenuate the shortest waves, so compare by correlation and sign
        // rather than value: a wrong axis, sign or scale of k would wreck the correlation.
        let dx = CASCADE_SIZES_M[0] / N as f32;
        let h = |x: usize, z: usize| f.disp[(z % N) * N + (x % N)][1];
        let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
        for z in 0..N {
            for x in 0..N {
                let fd = ((h(x + 1, z) - h(x + N - 1, z)) / (2.0 * dx)) as f64;
                let an = f.slope[z * N + x][0] as f64;
                sab += fd * an;
                saa += fd * fd;
                sbb += an * an;
            }
        }
        let corr = sab / (saa.sqrt() * sbb.sqrt());
        assert!(corr > 0.9, "corr {corr}");
    }
}
