//! Prints the distribution of the combined Jacobian (cascades 0–2) on a 1 m grid.
use ocean::params::OceanParams;
use ocean::waves::cpu_ocean::CpuOcean;
use ocean::waves::spectrum::{CASCADE_SIZES_M, N, build_spectrum};
fn main() {
    for (wind, chop) in [(5.0f32, 0.9f32), (10.0, 0.9), (15.0, 0.9), (20.0, 0.9)] {
        let p = OceanParams { wind_speed: wind, choppiness: chop, ..Default::default() };
        let o = CpuOcean::new(&build_spectrum(&p, 42), p.choppiness);
        let f: Vec<_> = (0..3).map(|c| o.evaluate(c, 0.0)).collect();
        let mut js = Vec::new();
        for z in 0..1024usize {
            for x in 0..1024usize {
                let mut d = [0.0f32; 3];
                for (c, fc) in f.iter().enumerate() {
                    let texel = CASCADE_SIZES_M[c] / N as f32;
                    let (cx, cz) = ((x as f32 / texel) as usize % N, (z as f32 / texel) as usize % N);
                    let g = fc.derivs[cz * N + cx];
                    (0..3).for_each(|i| d[i] += g[i]);
                }
                js.push((1.0 + d[0]) * (1.0 + d[1]) - d[2] * d[2]);
            }
        }
        let n = js.len() as f32;
        let frac = |t: f32| js.iter().filter(|&&j| j < t).count() as f32 / n * 100.0;
        let min = js.iter().cloned().fold(f32::MAX, f32::min);
        println!("wind {wind:>4}: min J {min:.3}  <0.7 {:.3}%  <0.6 {:.3}%  <0.5 {:.3}%  <0.4 {:.3}%  <0.3 {:.3}%", frac(0.7), frac(0.6), frac(0.5), frac(0.4), frac(0.3));
    }
}
