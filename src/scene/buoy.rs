//! A moored navigation buoy: procedural mesh and a five-point spring-damper float model.

use glam::{Mat4, Quat, Vec2, Vec3};

use crate::waves::query::WaterSample;

/// Where the water is sampled, relative to the buoy's waterline centre (world-aligned).
pub const SAMPLE_OFFSETS: [Vec3; 5] = [
    Vec3::ZERO,
    Vec3::new(0.9, 0.0, 0.0),
    Vec3::new(-0.9, 0.0, 0.0),
    Vec3::new(0.0, 0.0, 0.9),
    Vec3::new(0.0, 0.0, -0.9),
];

/// Heave natural frequency (rad/s) and damping ratio: a small buoy rides the swell closely.
const HEAVE_OMEGA: f32 = 2.5;
const HEAVE_ZETA: f32 = 0.6;
const ROLL_ZETA: f32 = 0.7;
const MAX_SUBSTEP_S: f32 = 1.0 / 120.0;

pub struct BuoyMesh {
    /// Position (xyz) and normal (xyz), metres; y = 0 is the waterline.
    pub vertices: Vec<[f32; 6]>,
    pub indices: Vec<u32>,
}

impl BuoyMesh {
    fn add_cylinder(&mut self, center: Vec2, radius: f32, y0: f32, y1: f32, segments: u32) {
        let ring = |y: f32, i: u32| {
            let a = std::f32::consts::TAU * i as f32 / segments as f32;
            (Vec3::new(center.x + radius * a.cos(), y, center.y + radius * a.sin()), Vec3::new(a.cos(), 0.0, a.sin()))
        };
        // Side.
        let base = self.vertices.len() as u32;
        for i in 0..segments {
            for y in [y0, y1] {
                let (p, n) = ring(y, i);
                self.vertices.push([p.x, p.y, p.z, n.x, n.y, n.z]);
            }
        }
        for i in 0..segments {
            let (a, b) = (base + 2 * i, base + 2 * ((i + 1) % segments));
            self.indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
        // Caps, with their own vertices for sharp edges.
        for (y, ny) in [(y0, -1.0f32), (y1, 1.0f32)] {
            let c = self.vertices.len() as u32;
            self.vertices.push([center.x, y, center.y, 0.0, ny, 0.0]);
            for i in 0..segments {
                let (p, _) = ring(y, i);
                self.vertices.push([p.x, p.y, p.z, 0.0, ny, 0.0]);
            }
            for i in 0..segments {
                let (a, b) = (c + 1 + i, c + 1 + (i + 1) % segments);
                if ny > 0.0 {
                    self.indices.extend_from_slice(&[c, b, a]);
                } else {
                    self.indices.extend_from_slice(&[c, a, b]);
                }
            }
        }
    }
}

/// Float drum (Ø1.8 m, half submerged), deck, four cage posts, top ring and lantern: 4 m tall.
pub fn build_buoy_mesh() -> BuoyMesh {
    let mut m = BuoyMesh { vertices: Vec::new(), indices: Vec::new() };
    m.add_cylinder(Vec2::ZERO, 0.9, -0.5, 0.4, 40);
    m.add_cylinder(Vec2::ZERO, 0.95, 0.4, 0.47, 40);
    for k in 0..4 {
        let a = std::f32::consts::FRAC_PI_2 * k as f32 + std::f32::consts::FRAC_PI_4;
        m.add_cylinder(Vec2::new(0.5 * a.cos(), 0.5 * a.sin()), 0.045, 0.47, 3.2, 10);
    }
    m.add_cylinder(Vec2::ZERO, 0.62, 3.2, 3.26, 32);
    m.add_cylinder(Vec2::ZERO, 0.14, 3.26, 3.5, 16);
    m
}

pub struct BuoyBody {
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat,
    pub angular_velocity: Vec3,
}

impl BuoyBody {
    pub fn at(position: Vec3) -> BuoyBody {
        BuoyBody { position, velocity: Vec3::ZERO, rotation: Quat::IDENTITY, angular_velocity: Vec3::ZERO }
    }

    /// World xz points whose water the next `step` needs (same order as SAMPLE_OFFSETS).
    pub fn sample_points(&self) -> [Vec2; 5] {
        SAMPLE_OFFSETS.map(|o| Vec2::new(self.position.x + o.x, self.position.z + o.z))
    }

    pub fn step(&mut self, samples: &[WaterSample; 5], dt: f32) {
        if !(dt.is_finite() && dt > 0.0) {
            return;
        }
        let n = (dt / MAX_SUBSTEP_S).ceil().clamp(1.0, 240.0) as u32;
        let h = dt / n as f32;
        let inv_inertia = 1.0 / (SAMPLE_OFFSETS.iter().map(|o| o.length_squared()).sum::<f32>() / SAMPLE_OFFSETS.len() as f32);
        for _ in 0..n {
            let mut force = 0.0;
            let mut torque = Vec3::ZERO;
            let mut count = 0.0;
            for (s, o) in samples.iter().zip(SAMPLE_OFFSETS) {
                if !(s.height.is_finite() && s.velocity.is_finite()) {
                    continue;
                }
                let r = self.rotation * o;
                let point_vy = self.velocity.y + self.angular_velocity.cross(r).y;
                let f = HEAVE_OMEGA * HEAVE_OMEGA * (s.height - (self.position.y + r.y))
                    + 2.0 * HEAVE_ZETA * HEAVE_OMEGA * (s.velocity.y - point_vy);
                force += f;
                torque += r.cross(Vec3::new(0.0, f, 0.0));
                count += 1.0;
            }
            if count > 0.0 {
                force /= count;
                torque /= count;
            }
            // Roll damping (on top of the per-point heave damping) keeps it from ringing.
            let ang_acc = torque * inv_inertia - 2.0 * ROLL_ZETA * HEAVE_OMEGA * self.angular_velocity;
            self.velocity.y += force * h;
            self.angular_velocity += ang_acc * h;
            // Moored: no drift and no spin about the vertical.
            self.velocity.x = 0.0;
            self.velocity.z = 0.0;
            self.angular_velocity.y = 0.0;
            self.position += self.velocity * h;
            let w = self.angular_velocity;
            if w.length_squared() > 0.0 {
                self.rotation = (Quat::from_scaled_axis(w * h) * self.rotation).normalize();
            }
        }
    }

    pub fn transform(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waves::query::WaterSample;
    use glam::Vec3;

    fn flat(height: f32) -> [WaterSample; 5] {
        [WaterSample { height, normal: Vec3::Y, velocity: Vec3::ZERO }; 5]
    }

    #[test]
    fn buoy_settles_on_flat_water() {
        let mut b = BuoyBody::at(Vec3::new(0.0, 0.8, 0.0));
        for _ in 0..(20 * 60) {
            b.step(&flat(0.0), 1.0 / 60.0);
        }
        assert!(b.position.y.abs() < 0.02, "y {}", b.position.y);
        assert!(b.velocity.length() < 0.01, "v {}", b.velocity);
    }

    #[test]
    fn buoy_follows_swell() {
        // Heave on a 10 s swell of 0.5 m: the buoy must track with < 0.5 s lag.
        let mut b = BuoyBody::at(Vec3::ZERO);
        let dt = 1.0 / 60.0;
        let (w, a) = (std::f32::consts::TAU / 10.0, 0.5);
        let mut best = (f32::MAX, 0.0);
        let mut trace = Vec::new();
        for i in 0..(40 * 60) {
            let t = i as f32 * dt;
            let mut s = flat(a * (w * t).sin());
            s.iter_mut().for_each(|x| x.velocity.y = a * w * (w * t).cos());
            b.step(&s, dt);
            if t > 20.0 {
                trace.push((t, b.position.y));
            }
        }
        // Find the lag that best aligns the buoy with the water.
        for lag_i in 0..60 {
            let lag = lag_i as f32 * dt;
            let err: f32 = trace.iter().map(|(t, y)| (y - a * (w * (t - lag)).sin()).powi(2)).sum();
            if err < best.0 {
                best = (err, lag);
            }
        }
        assert!(best.1 < 0.5, "lag {} s", best.1);
        let peak = trace.iter().map(|(_, y)| y.abs()).fold(0.0, f32::max);
        assert!((peak - a).abs() < 0.15, "peak heave {peak}");
    }

    #[test]
    fn buoy_tilts_with_the_surface() {
        let mut b = BuoyBody::at(Vec3::ZERO);
        // Water higher on the +x side.
        let mut s = flat(0.0);
        for (i, o) in SAMPLE_OFFSETS.iter().enumerate() {
            s[i].height = 0.2 * o.x;
        }
        for _ in 0..600 {
            b.step(&s, 1.0 / 60.0);
        }
        let up = b.rotation * Vec3::Y;
        assert!(up.x < -0.05, "buoy should lean away from the high side: up {up}");
    }

    #[test]
    fn buoy_stays_finite_at_extremes() {
        let mut seed = 1u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        for dt in [1.0 / 240.0, 1.0 / 30.0, 0.25] {
            let mut b = BuoyBody::at(Vec3::ZERO);
            for _ in 0..1000 {
                let mut s = flat(0.0);
                for x in s.iter_mut() {
                    x.height = rnd() * 5.0;
                    x.normal = Vec3::new(rnd(), 1.0, rnd()).normalize();
                }
                b.step(&s, dt);
                assert!(b.position.is_finite() && b.velocity.is_finite() && b.angular_velocity.is_finite(), "dt {dt}");
                assert!(b.rotation.is_finite() && (b.rotation.length() - 1.0).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn non_finite_samples_are_ignored() {
        let mut b = BuoyBody::at(Vec3::ZERO);
        let mut s = flat(0.0);
        s[2].height = f32::NAN;
        b.step(&s, 1.0 / 60.0);
        assert!(b.position.is_finite());
    }

    #[test]
    fn buoy_mesh_is_closed_and_sane() {
        let m = build_buoy_mesh();
        let n = m.vertices.len() as u32;
        assert!(m.indices.len().is_multiple_of(3) && !m.indices.is_empty());
        assert!(m.indices.iter().all(|&i| i < n));
        for t in m.indices.chunks_exact(3) {
            let p = |i: u32| Vec3::from_slice(&m.vertices[i as usize][0..3]);
            let area = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).length();
            assert!(area > 1e-8, "degenerate triangle {t:?}");
        }
        let (lo, hi) = m.vertices.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v[1]), hi.max(v[1])));
        assert!((3.5..=4.5).contains(&(hi - lo)), "height {}", hi - lo);
        assert!(lo < -0.2, "drum must reach below the waterline");
        assert!((m.indices.len() / 3) < 8000, "{} triangles", m.indices.len() / 3);
        for v in &m.vertices {
            assert!((Vec3::from_slice(&v[3..6]).length() - 1.0).abs() < 1e-3, "unit normals");
        }
    }
}
