/// PCG32 (O'Neill): small, deterministic, good enough for wave phases. Seeded so the CPU
/// reference and the GPU see the identical ocean.
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64) -> Self {
        let mut r = Pcg32 { state: 0, inc: (seed << 1) | 1 };
        r.next_u32();
        r.state = r.state.wrapping_add(0x853c_49e6_748f_ea9b ^ seed);
        r.next_u32();
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Two independent standard normals (Box–Muller).
    pub fn gaussian_pair(&mut self) -> (f32, f32) {
        let u1 = 1.0 - self.next_f32(); // (0, 1], keeps ln finite
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let a = std::f32::consts::TAU * u2;
        (r * a.cos(), r * a.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let (mut a, mut b) = (Pcg32::new(9), Pcg32::new(9));
        for _ in 0..100 {
            assert_eq!(a.next_f32(), b.next_f32());
        }
    }

    #[test]
    fn uniform_in_unit_interval() {
        let mut r = Pcg32::new(1);
        for _ in 0..10_000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn gaussian_has_unit_variance() {
        let mut r = Pcg32::new(3);
        let n = 50_000;
        let (mut sum, mut sq) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let (a, b) = r.gaussian_pair();
            for v in [a as f64, b as f64] {
                assert!(v.is_finite());
                sum += v;
                sq += v * v;
            }
        }
        let mean = sum / (2 * n) as f64;
        let var = sq / (2 * n) as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }
}
