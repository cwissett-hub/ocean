//! Ocean wave spectra (JONSWAP + directional spreading) and the per-cascade initial
//! amplitudes `h0(k)` that both the CPU reference and the GPU evolve in time.

use std::f64::consts::{PI as PI64, TAU as TAU64};

use crate::params::OceanParams;
use crate::waves::rng::Pcg32;

pub const G: f32 = 9.81;
/// Wave frequencies are quantised so the whole field repeats exactly after this many seconds;
/// time is passed to the GPU modulo this, so f32 phase never loses precision.
pub const REPEAT_PERIOD_S: f32 = 200.0;
pub const CASCADE_SIZES_M: [f32; 4] = [1024.0, 256.0, 64.0, 16.0];
pub const N: usize = 256;
/// Cascade i (i >= 1) starts at 2π·BAND_CYCLES / L_i, so the shortest waves of cascade i-1 are
/// L_i / 16 long: 4 texels per wavelength in cascade i-1's grid, and the JONSWAP peak usually
/// lands in the finely sampled cascade 0.
const BAND_CYCLES: f32 = 16.0;

const WIND_SEA_SPREAD: f32 = 6.0;
const SWELL_SPREAD: f32 = 24.0;
const SWELL_PERIOD_S: f32 = 10.0;
const SWELL_DIR_OFFSET_DEG: f32 = 20.0;
/// Significant wave height of the swell at `swell_amount = 1`.
const SWELL_HS_AT_FULL_M: f32 = 1.0;
const GAMMA: f64 = 3.3;

/// Deep-water dispersion ω(k) = √(g·k).
pub fn dispersion(k: f32) -> f32 {
    (G * k).sqrt()
}

/// ω(k) rounded down to a multiple of 2π / REPEAT_PERIOD_S.
pub fn quantized_omega(k: f32) -> f32 {
    let base = TAU64 / REPEAT_PERIOD_S as f64;
    ((dispersion(k) as f64 / base).floor() * base) as f32
}

pub fn jonswap_peak_omega(wind: f32, fetch_m: f32) -> f32 {
    let (u, f, g) = (wind as f64, fetch_m as f64, G as f64);
    (22.0 * (g * g / (u * f)).powf(1.0 / 3.0)) as f32
}

/// Dimensionless fetch gF/U² at which the fetch law reaches the Pierson–Moskowitz fully
/// developed height 0.21·U²/g; longer fetch adds no energy.
const FULLY_DEVELOPED_XI: f64 = (0.21 / 1.6e-3) * (0.21 / 1.6e-3);

/// Fetch, capped where the sea becomes fully developed.
pub fn effective_fetch(wind: f32, fetch_m: f32) -> f32 {
    let u = wind as f64;
    (fetch_m as f64).min(FULLY_DEVELOPED_XI * u * u / G as f64) as f32
}

/// JONSWAP fetch-growth law Hs = 1.6e-3·U·√(F/g), on the effective fetch.
pub fn significant_height(wind: f32, fetch_m: f32) -> f32 {
    let f = effective_fetch(wind, fetch_m) as f64;
    (1.6e-3 * wind as f64 * (f / G as f64).sqrt()) as f32
}

/// m0 of the JONSWAP shape with α = 1.
fn shape_m0(omega_p: f64) -> f64 {
    let (n, a, b) = (4000, 0.02f64.ln(), 40.0f64.ln());
    (0..n)
        .map(|i| {
            let w = (a + (i as f64 + 0.5) * (b - a) / n as f64).exp();
            jonswap_shape(w, omega_p, 1.0) * w * (b - a) / n as f64
        })
        .sum()
}

/// Energy scale α so the spectrum carries exactly Hs²/16 of variance. (The classic
/// α = 0.076·(U²/gF)^0.22 fit overshoots the fully developed limit by ~2x at moderate winds.)
fn alpha_for_hs(hs: f64, omega_p: f64) -> f64 {
    let m0 = shape_m0(omega_p);
    if m0 > 0.0 { (hs / 4.0).powi(2) / m0 } else { 0.0 }
}

fn wind_peak_and_alpha(wind: f32, fetch_m: f32) -> (f64, f64) {
    let wp = jonswap_peak_omega(wind, effective_fetch(wind, fetch_m)) as f64;
    (wp, alpha_for_hs(significant_height(wind, fetch_m) as f64, wp))
}

/// JONSWAP shape with explicit peak and energy scale, in m²·s.
fn jonswap_shape(omega: f64, omega_p: f64, alpha: f64) -> f64 {
    if omega <= 0.0 || omega_p <= 0.0 {
        return 0.0;
    }
    let ratio = omega_p / omega;
    if ratio > 8.0 {
        return 0.0; // exp(-1.25·ratio⁴) underflows long before ω⁻⁵ matters
    }
    let g = G as f64;
    let sigma = if omega <= omega_p { 0.07 } else { 0.09 };
    let r = (-(omega - omega_p).powi(2) / (2.0 * sigma * sigma * omega_p * omega_p)).exp();
    alpha * g * g * omega.powi(-5) * (-1.25 * ratio.powi(4)).exp() * GAMMA.powf(r)
}

/// Fetch-limited JONSWAP frequency spectrum S(ω) in m²·s.
pub fn jonswap(omega: f32, wind: f32, fetch_m: f32) -> f32 {
    let (wp, alpha) = wind_peak_and_alpha(wind, fetch_m);
    jonswap_shape(omega as f64, wp, alpha) as f32
}

/// Wavevector of texel (x, z) in a cascade's centred k grid. Shared by everything that walks
/// the grid so band edges are decided by bit-identical f32 maths.
pub fn wavevector(cascade: usize, x: usize, z: usize) -> (f32, f32) {
    let dk = std::f32::consts::TAU / CASCADE_SIZES_M[cascade];
    ((x as f32 - (N / 2) as f32) * dk, (z as f32 - (N / 2) as f32) * dk)
}

fn spreading_norm(s: f32) -> f64 {
    let n = 4096;
    let sum: f64 = (0..n)
        .map(|i| {
            let th = -PI64 + (i as f64 + 0.5) * TAU64 / n as f64;
            (th / 2.0).cos().abs().powf(2.0 * s as f64)
        })
        .sum();
    1.0 / (sum * TAU64 / n as f64)
}

/// Normalised directional spreading cos²ˢ(θ/2); θ is the angle from the mean wave direction.
pub fn spreading(theta: f32, s: f32) -> f32 {
    (spreading_norm(s) * (theta as f64 / 2.0).cos().abs().powf(2.0 * s as f64)) as f32
}

/// `[k_lo, k_hi)` owned by cascade i, so no wavenumber is counted twice.
pub fn cascade_band(i: usize) -> (f32, f32) {
    let lo = |c: usize| if c == 0 { 0.0 } else { std::f32::consts::TAU * BAND_CYCLES / CASCADE_SIZES_M[c] };
    let hi = if i + 1 < CASCADE_SIZES_M.len() { lo(i + 1) } else { std::f32::consts::PI * N as f32 / CASCADE_SIZES_M[i] };
    (lo(i), hi)
}

/// Initial amplitudes for all cascades. Per texel (row-major N×N, k grid centred so texel n has
/// k = 2π(n − N/2)/L): `[h0(k).re, h0(k).im, conj(h0(−k)).re, conj(h0(−k)).im]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectrumSet {
    pub h0: [Vec<[f32; 4]>; 4],
    /// Quantised ω per texel, computed once here so CPU and GPU agree bit-for-bit.
    pub omega: [Vec<f32>; 4],
}

struct Component {
    dir: f64,
    omega_p: f64,
    alpha: f64,
    norm: f64,
    spread: f32,
}

impl Component {
    /// Directional wavenumber spectrum E(k) = S(ω)·D(θ)·(dω/dk)/k, m⁴.
    fn energy(&self, kx: f64, kz: f64) -> f64 {
        let k = (kx * kx + kz * kz).sqrt();
        if k == 0.0 || self.alpha == 0.0 {
            return 0.0;
        }
        let g = G as f64;
        let omega = (g * k).sqrt();
        let mut theta = kx.atan2(kz) - self.dir;
        theta = (theta + PI64).rem_euclid(TAU64) - PI64;
        let d = self.norm * (theta / 2.0).cos().abs().powf(2.0 * self.spread as f64);
        let domega_dk = g / (2.0 * omega);
        jonswap_shape(omega, self.omega_p, self.alpha) * d * domega_dk / k
    }
}

fn components(p: &OceanParams) -> [Component; 2] {
    let wind_dir = (p.wind_dir_deg as f64).to_radians();
    let (wind_wp, wind_alpha) = wind_peak_and_alpha(p.wind_speed, p.fetch_km * 1000.0);
    let wind = Component {
        dir: wind_dir,
        omega_p: wind_wp,
        alpha: wind_alpha,
        norm: spreading_norm(WIND_SEA_SPREAD),
        spread: WIND_SEA_SPREAD,
    };
    let swell_wp = TAU64 / SWELL_PERIOD_S as f64;
    let swell = Component {
        dir: wind_dir + (SWELL_DIR_OFFSET_DEG as f64).to_radians(),
        omega_p: swell_wp,
        alpha: alpha_for_hs(p.swell_amount as f64 * SWELL_HS_AT_FULL_M as f64, swell_wp),
        norm: spreading_norm(SWELL_SPREAD),
        spread: SWELL_SPREAD,
    };
    [wind, swell]
}

pub fn build_spectrum(p: &OceanParams, seed: u64) -> SpectrumSet {
    let p = p.clamped();
    let comps = components(&p);
    let mut h0s: [Vec<[f32; 4]>; 4] = Default::default();
    let mut omegas: [Vec<f32>; 4] = Default::default();
    for c in 0..4 {
        let dk = (std::f32::consts::TAU / CASCADE_SIZES_M[c]) as f64;
        let (lo, hi) = cascade_band(c);
        let mut rng = Pcg32::new(seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(c as u64 + 1)));
        let mut h0 = vec![[0.0f32; 2]; N * N];
        let mut omega = vec![0.0f32; N * N];
        for z in 0..N {
            for x in 0..N {
                // Draw for every texel, even unused ones, so phases don't reshuffle when the
                // wind changes and the band contents move.
                let (g1, g2) = rng.gaussian_pair();
                let (kx, kz) = wavevector(c, x, z);
                let k = (kx * kx + kz * kz).sqrt();
                omega[z * N + x] = quantized_omega(k);
                if k == 0.0 || k < lo || k >= hi {
                    continue;
                }
                let e: f64 = comps.iter().map(|cp| cp.energy(kx as f64, kz as f64)).sum();
                // E|h0|² must be half the bin variance: h(k) gets energy from h0(k) and h0(−k).
                let amp = (e * dk * dk / 4.0).sqrt();
                h0[z * N + x] = [(g1 as f64 * amp) as f32, (g2 as f64 * amp) as f32];
            }
        }
        h0s[c] = (0..N * N)
            .map(|i| {
                let (x, z) = (i % N, i / N);
                let m = ((N - z) % N) * N + (N - x) % N;
                [h0[i][0], h0[i][1], h0[m][0], -h0[m][1]]
            })
            .collect();
        omegas[c] = omega;
    }
    SpectrumSet { h0: h0s, omega: omegas }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::params::OceanParams;
    use std::f32::consts::{PI, TAU};

    /// Midpoint rule in log-space over [a, b].
    pub(crate) fn integrate(f: impl Fn(f32) -> f32, a: f32, b: f32) -> f32 {
        let n = 20_000;
        let (la, lb) = (a.ln(), b.ln());
        (0..n)
            .map(|i| {
                let l = la + (i as f32 + 0.5) * (lb - la) / n as f32;
                let x = l.exp();
                f(x) * x * (lb - la) / n as f32
            })
            .sum()
    }

    #[test]
    fn dispersion_deep_water() {
        assert!((dispersion(1.0) - 3.1321).abs() < 1e-3);
    }

    #[test]
    fn quantized_omega_is_multiple_of_base_and_close() {
        let base = TAU / REPEAT_PERIOD_S;
        for k in [0.01f32, 0.2, 3.0, 40.0] {
            let w = quantized_omega(k);
            let n = w / base;
            assert!((n - n.round()).abs() < 1e-3, "k={k} n={n}");
            assert!((dispersion(k) - w).abs() <= base + 1e-5);
        }
    }

    #[test]
    fn peak_omega_matches_jonswap() {
        assert!((jonswap_peak_omega(5.0, 100_000.0) - 1.2703).abs() < 1e-3);
    }

    #[test]
    fn spreading_integrates_to_one() {
        for s in [1.0, 8.0, 24.0] {
            let n = 4096;
            let sum: f32 = (0..n)
                .map(|i| {
                    let th = -PI + (i as f32 + 0.5) * TAU / n as f32;
                    spreading(th, s) * TAU / n as f32
                })
                .sum();
            assert!((sum - 1.0).abs() < 1e-3, "s={s} sum={sum}");
        }
    }

    fn spectrum_hs(wind: f32, fetch_m: f32) -> f32 {
        4.0 * integrate(|w| jonswap(w, wind, fetch_m), 0.02, 40.0).sqrt()
    }

    #[test]
    fn significant_height_follows_fetch_law() {
        // Fetch law Hs = 1.6e-3·U·√(F/g): U = 10 m/s, F = 20 km -> 0.7225 m (not yet fully developed).
        let hs = spectrum_hs(10.0, 20_000.0);
        assert!((hs - 0.7225).abs() / 0.7225 < 0.03, "hs={hs}");
    }

    #[test]
    fn long_fetch_caps_at_fully_developed() {
        // Pierson–Moskowitz: Hs = 0.21·U²/g = 0.535 m at 5 m/s; 100 km of fetch is past that.
        let hs = spectrum_hs(5.0, 100_000.0);
        assert!((hs - 0.535).abs() / 0.535 < 0.03, "hs={hs}");
        assert!((spectrum_hs(5.0, 500_000.0) - hs).abs() < 1e-3);
    }

    #[test]
    fn fully_developed_peak_matches_pierson_moskowitz() {
        // PM peak ω_p ≈ 0.877·g/U; the capped JONSWAP peak must land within 5 %.
        let wp = jonswap_peak_omega(5.0, effective_fetch(5.0, 500_000.0));
        let pm = 0.877 * G / 5.0;
        assert!((wp - pm).abs() / pm < 0.05, "wp {wp} pm {pm}");
    }

    #[test]
    fn cascade_bands_partition_k() {
        let top = cascade_band(3).1;
        assert!((top - PI * N as f32 / CASCADE_SIZES_M[3]).abs() < 1e-4);
        let n = 10_000;
        let (l0, l1) = (1e-3f32.ln(), top.ln());
        for i in 0..n {
            let k = (l0 + (i as f32 + 0.5) / n as f32 * (l1 - l0)).exp();
            let owners = (0..4)
                .filter(|&c| {
                    let (lo, hi) = cascade_band(c);
                    k >= lo && k < hi
                })
                .count();
            assert_eq!(owners, 1, "k={k}");
        }
    }

    #[test]
    fn every_band_is_below_its_cascade_nyquist() {
        for (c, l) in CASCADE_SIZES_M.iter().enumerate() {
            assert!(cascade_band(c).1 <= PI * N as f32 / l + 1e-4, "cascade {c}");
        }
    }

    #[test]
    fn spectrum_finite_at_extremes() {
        for (w, f) in [(25.0, 10.0), (0.5, 500.0), (25.0, 500.0), (0.5, 10.0)] {
            for swell in [0.0, 1.0] {
                let p = OceanParams { wind_speed: w, fetch_km: f, swell_amount: swell, ..Default::default() };
                let s = build_spectrum(&p, 7);
                assert!(
                    s.h0.iter().flatten().flatten().all(|v| v.is_finite()),
                    "wind {w} fetch {f} swell {swell}"
                );
            }
        }
    }

    #[test]
    fn same_seed_same_spectrum() {
        let p = OceanParams::default();
        let (a, b) = (build_spectrum(&p, 42), build_spectrum(&p, 42));
        assert_eq!(a.h0, b.h0);
        assert_ne!(a.h0, build_spectrum(&p, 43).h0);
    }

    #[test]
    fn h0_is_zero_outside_band_and_at_k_zero() {
        let s = build_spectrum(&OceanParams::default(), 1);
        for c in 0..4 {
            let (lo, hi) = cascade_band(c);
            for z in 0..N {
                for x in 0..N {
                    let (kx, kz) = wavevector(c, x, z);
                    let k = (kx * kx + kz * kz).sqrt();
                    if k == 0.0 || k < lo || k >= hi {
                        assert_eq!(s.h0[c][z * N + x], [0.0; 4], "cascade {c} k={k}");
                    }
                }
            }
        }
    }
}
