//! The driftwood raft: buoyancy on nine water samples, a self-trimming square sail, a keel-like
//! hull, a paddle, gentle steering, and grounding on island terrain. All accelerations are per
//! unit mass.

use glam::{Mat4, Quat, Vec2, Vec3};

use crate::waves::query::WaterSample;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RaftControls {
    /// −1 (left) … 1 (right).
    pub steer: f32,
    /// −1 (backwards) … 1 (forwards).
    pub paddle: f32,
    pub sail_up: bool,
}

/// 3×3 grid on the 3.0 m (along) × 2.4 m (across) deck; local, y = 0 at the waterline.
pub const RAFT_SAMPLE_OFFSETS: [Vec3; 9] = [
    Vec3::new(-1.2, 0.0, -1.5),
    Vec3::new(0.0, 0.0, -1.5),
    Vec3::new(1.2, 0.0, -1.5),
    Vec3::new(-1.2, 0.0, 0.0),
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.2, 0.0, 0.0),
    Vec3::new(-1.2, 0.0, 1.5),
    Vec3::new(0.0, 0.0, 1.5),
    Vec3::new(1.2, 0.0, 1.5),
];
/// How far the logs reach below the waterline.
pub const RAFT_DRAFT_M: f32 = 0.15;

const HEAVE_OMEGA: f32 = 2.2;
const HEAVE_ZETA: f32 = 0.6;
const ROLL_ZETA: f32 = 0.8;
/// Quadratic hull drag along and across the logs (1/m): the raft goes where it points.
const DRAG_FORWARD: f32 = 0.25;
const DRAG_LATERAL: f32 = 3.75;
const DRAG_LINEAR: f32 = 0.05;
/// Sail: force = SAIL_K·|apparent wind|²·facing² along the sail normal.
const SAIL_K: f32 = 0.11;
/// A square sail can be braced at most this far from the hull axis.
const SAIL_MAX_BRACE: f32 = 70.0 * std::f32::consts::PI / 180.0;
/// Wind on the raft itself, sail down (drift).
const WINDAGE_K: f32 = 0.000_6;
const PADDLE_THRUST: f32 = 0.105;
/// Yaw: ω̇ = TURN_K·steer·(TURN_BASE + |v_fwd|) − YAW_DAMPING·ω.
const TURN_K: f32 = 0.15;
const TURN_BASE: f32 = 0.3;
const YAW_DAMPING: f32 = 1.5;
const MAX_YAW_RATE: f32 = 15.0 * std::f32::consts::PI / 180.0;
const MAX_SUBSTEP_S: f32 = 1.0 / 120.0;
/// Ground contact: stiff spring along the terrain normal, damping, Coulomb friction.
const CONTACT_K: f32 = 400.0;
const CONTACT_DAMPING: f32 = 40.0;
const GROUND_MU: f32 = 1.2;
/// Logs ploughing into sand: extra drag per metre of penetration.
const PLOUGH_DRAG: f32 = 30.0;
/// Aground, the paddle becomes a pole pushed against the sand.
const PUSH_OFF_GAIN: f32 = 30.0;

/// Force of a self-trimming square sail in the apparent wind (both world xz).
pub fn sail_force(apparent_wind: Vec2, forward: Vec2, sail_up: bool) -> Vec2 {
    let speed = apparent_wind.length();
    if !sail_up || speed < 1e-4 {
        return Vec2::ZERO;
    }
    let a = apparent_wind / speed;
    // Sail normal halfway between the hull axis and the wind, braced at most SAIL_MAX_BRACE.
    let bisector = (forward + a).normalize_or_zero();
    let mut normal = if bisector == Vec2::ZERO { forward } else { bisector };
    let angle = forward.perp_dot(normal).atan2(forward.dot(normal));
    if angle.abs() > SAIL_MAX_BRACE {
        let b = SAIL_MAX_BRACE * angle.signum();
        normal = Vec2::new(
            forward.x * b.cos() - forward.y * b.sin(),
            forward.x * b.sin() + forward.y * b.cos(),
        );
    }
    let facing = normal.dot(a).max(0.0);
    normal * (SAIL_K * speed * speed * facing * facing)
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaftBody {
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat,
    pub angular_velocity: Vec3,
    /// Compass heading, rad: 0 = +z, clockwise from above (like the sun's azimuth).
    pub heading: f32,
    /// Any part of the hull is resting on terrain this step.
    pub grounded: bool,
    /// Seaward direction (down the terrain slope) while aground.
    pub downhill: Vec2,
}

impl RaftBody {
    pub fn at(position: Vec3, heading: f32) -> RaftBody {
        RaftBody {
            position,
            velocity: Vec3::ZERO,
            rotation: Quat::from_rotation_y(heading),
            angular_velocity: Vec3::ZERO,
            heading,
            grounded: false,
            downhill: Vec2::ZERO,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.heading.sin(), 0.0, self.heading.cos())
    }

    pub fn speed(&self) -> f32 {
        Vec2::new(self.velocity.x, self.velocity.z).length()
    }

    pub fn transform(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.position)
    }

    /// World xz where the water must be sampled (same order as RAFT_SAMPLE_OFFSETS).
    pub fn sample_points(&self) -> [Vec2; 9] {
        let yaw = Quat::from_rotation_y(self.heading);
        RAFT_SAMPLE_OFFSETS.map(|o| {
            let r = yaw * o;
            Vec2::new(self.position.x + r.x, self.position.z + r.z)
        })
    }

    pub fn step(
        &mut self,
        water: &[WaterSample; 9],
        wind: Vec2,
        c: &RaftControls,
        terrain: &dyn Fn(Vec2) -> f32,
        dt: f32,
    ) {
        if !(dt.is_finite() && dt > 0.0) {
            return;
        }
        let wind = if wind.is_finite() { wind } else { Vec2::ZERO };
        let steer = c.steer.clamp(-1.0, 1.0);
        let paddle = c.paddle.clamp(-1.0, 1.0);
        let n = (dt / MAX_SUBSTEP_S).ceil().clamp(1.0, 240.0) as u32;
        let h = dt / n as f32;
        let inv_inertia =
            RAFT_SAMPLE_OFFSETS.len() as f32 / RAFT_SAMPLE_OFFSETS.iter().map(|o| o.length_squared()).sum::<f32>();
        for _ in 0..n {
            // Vertical: per-point spring-damper towards the water surface.
            let mut lift = 0.0;
            let mut torque = Vec3::ZERO;
            let mut count = 0.0;
            for (s, o) in water.iter().zip(RAFT_SAMPLE_OFFSETS) {
                if !(s.height.is_finite() && s.velocity.is_finite()) {
                    continue;
                }
                let r = self.rotation * o;
                let vy = self.velocity.y + self.angular_velocity.cross(r).y;
                let f = HEAVE_OMEGA * HEAVE_OMEGA * (s.height - (self.position.y + r.y))
                    + 2.0 * HEAVE_ZETA * HEAVE_OMEGA * (s.velocity.y - vy);
                lift += f;
                torque += r.cross(Vec3::new(0.0, f, 0.0));
                count += 1.0;
            }
            if count > 0.0 {
                lift /= count;
                torque /= count;
            }

            // Horizontal: sail, windage, paddle, hull drag in the hull frame.
            let fwd = Vec2::new(self.heading.sin(), self.heading.cos());
            let side = Vec2::new(fwd.y, -fwd.x);
            let v = Vec2::new(self.velocity.x, self.velocity.z);
            let apparent = wind - v;
            // Aground, a paddle stroke towards the sea is a push-off against the sand.
            let pole = if self.grounded && (fwd * paddle).dot(self.downhill) > 0.0 {
                PUSH_OFF_GAIN
            } else {
                1.0
            };
            let mut force = sail_force(apparent, fwd, c.sail_up)
                + apparent * apparent.length() * WINDAGE_K
                + fwd * (paddle * PADDLE_THRUST * pole);
            let (vf, vl) = (v.dot(fwd), v.dot(side));
            force -= fwd * (DRAG_FORWARD * vf * vf.abs()) + side * (DRAG_LATERAL * vl * vl.abs()) + v * DRAG_LINEAR;

            let (contact, contact_torque) = self.contact(terrain, h);
            // Yaw.
            // Aground you pivot by poling, about as fast as turning under way.
            let turn_speed = if self.grounded { vf.abs().max(1.0) } else { vf.abs() };
            let yaw_acc = TURN_K * steer * (TURN_BASE + turn_speed) - YAW_DAMPING * self.angular_velocity.y;
            let pitch_roll = (torque + contact_torque) * inv_inertia
                - 2.0 * ROLL_ZETA * HEAVE_OMEGA * Vec3::new(self.angular_velocity.x, 0.0, self.angular_velocity.z);

            self.velocity += (Vec3::new(force.x, lift, force.y) + contact) * h;
            self.angular_velocity += Vec3::new(pitch_roll.x, yaw_acc, pitch_roll.z) * h;
            self.angular_velocity.y = self.angular_velocity.y.clamp(-MAX_YAW_RATE, MAX_YAW_RATE);
            self.position += self.velocity * h;
            self.heading = (self.heading + self.angular_velocity.y * h).rem_euclid(std::f32::consts::TAU);
            // Yaw is tracked by `heading`; the quaternion integrates only pitch and roll.
            let tilt_w = Vec3::new(self.angular_velocity.x, 0.0, self.angular_velocity.z);
            let tilt = self.tilt();
            let tilt = if tilt_w.length_squared() > 0.0 {
                (Quat::from_scaled_axis(tilt_w * h) * tilt).normalize()
            } else {
                tilt
            };
            self.rotation = (Quat::from_rotation_y(self.heading) * tilt).normalize();
        }
    }

    /// Pitch/roll part of the rotation (yaw removed), in world axes.
    fn tilt(&self) -> Quat {
        (Quat::from_rotation_y(self.heading).inverse() * self.rotation).normalize()
    }

    /// Ground contact at every hull point touching terrain: a spring-damper along the terrain
    /// normal plus Coulomb friction (static: can stop sliding within the substep). Returns
    /// (acceleration, torque) per unit mass, and sets `grounded` and `downhill`.
    fn contact(&mut self, terrain: &dyn Fn(Vec2) -> f32, h: f32) -> (Vec3, Vec3) {
        self.grounded = false;
        self.downhill = Vec2::ZERO;
        let points = self.sample_points();
        let mut hits = Vec::new();
        for (p, o) in points.iter().zip(RAFT_SAMPLE_OFFSETS) {
            let ground = terrain(*p);
            if !ground.is_finite() {
                continue;
            }
            let r = self.rotation * o;
            let depth = ground - (self.position.y + r.y - RAFT_DRAFT_M);
            if depth <= 0.0 {
                continue;
            }
            let e = 0.25;
            let grad = Vec2::new(
                terrain(*p + Vec2::X * e) - terrain(*p - Vec2::X * e),
                terrain(*p + Vec2::Y * e) - terrain(*p - Vec2::Y * e),
            ) / (2.0 * e);
            let grad = if grad.is_finite() { grad } else { Vec2::ZERO };
            hits.push((r, depth, grad));
        }
        if hits.is_empty() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        self.grounded = true;
        let share = RAFT_SAMPLE_OFFSETS.len() as f32 / hits.len() as f32;
        let mut acc = Vec3::ZERO;
        let mut torque = Vec3::ZERO;
        for (r, depth, grad) in hits {
            self.downhill -= grad;
            let normal = Vec3::new(-grad.x, 1.0, -grad.y).normalize();
            let v = self.velocity + self.angular_velocity.cross(r);
            let vn = v.dot(normal);
            let push = (CONTACT_K * depth - CONTACT_DAMPING * vn).max(0.0);
            let tangential = v - normal * vn;
            let t = tangential.length();
            // Each contact may cancel its share of the sliding velocity, no more (no reversal).
            let friction = if t > 1e-5 {
                -tangential / t * (GROUND_MU * push + PLOUGH_DRAG * depth * t).min(t / h * share)
            } else {
                Vec3::ZERO
            };
            let f = (normal * push + friction) / RAFT_SAMPLE_OFFSETS.len() as f32;
            acc += f;
            torque += r.cross(f);
        }
        self.downhill = self.downhill.normalize_or_zero();
        (acc, torque)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waves::query::WaterSample;
    use glam::{Vec2, Vec3};

    const DT: f32 = 1.0 / 60.0;

    fn flat() -> [WaterSample; 9] {
        [WaterSample {
            height: 0.0,
            normal: Vec3::Y,
            velocity: Vec3::ZERO,
        }; 9]
    }

    fn deep(_: Vec2) -> f32 {
        -25.0
    }

    fn controls(paddle: f32, steer: f32, sail_up: bool) -> RaftControls {
        RaftControls { steer, paddle, sail_up }
    }

    fn run(r: &mut RaftBody, wind: Vec2, c: &RaftControls, secs: f32, terrain: &dyn Fn(Vec2) -> f32) {
        for _ in 0..(secs / DT) as usize {
            r.step(&flat(), wind, c, terrain, DT);
        }
    }

    fn fwd_speed(r: &RaftBody) -> f32 {
        Vec2::new(r.velocity.x, r.velocity.z).dot(Vec2::new(r.forward().x, r.forward().z))
    }

    #[test]
    fn floats_flat_on_calm_water() {
        let mut r = RaftBody::at(Vec3::new(0.0, 0.5, 0.0), 0.0);
        run(&mut r, Vec2::ZERO, &controls(0.0, 0.0, false), 20.0, &deep);
        assert!(r.position.y.abs() < 0.02, "heave {}", r.position.y);
        let up = r.rotation * Vec3::Y;
        assert!(up.angle_between(Vec3::Y).to_degrees() < 1.0, "tilt {up}");
    }

    #[test]
    fn downwind_speed_is_leisurely() {
        let mut r = RaftBody::at(Vec3::ZERO, 0.0);
        run(&mut r, Vec2::new(0.0, 5.0), &controls(0.0, 0.0, true), 120.0, &deep);
        let v = fwd_speed(&r);
        assert!((1.0..=3.0).contains(&v), "downwind {v} m/s");
    }

    #[test]
    fn upwind_barely_moves() {
        let mut r = RaftBody::at(Vec3::ZERO, std::f32::consts::PI);
        run(&mut r, Vec2::new(0.0, 5.0), &controls(0.0, 0.0, true), 60.0, &deep);
        assert!(fwd_speed(&r) < 0.3, "upwind {} m/s", fwd_speed(&r));
    }

    #[test]
    fn reaching_moves_across_the_wind() {
        let mut r = RaftBody::at(Vec3::ZERO, std::f32::consts::FRAC_PI_2);
        run(&mut r, Vec2::new(0.0, 5.0), &controls(0.0, 0.0, true), 90.0, &deep);
        let f = fwd_speed(&r);
        let total = Vec2::new(r.velocity.x, r.velocity.z).length();
        let lateral = (total * total - f * f).max(0.0).sqrt();
        assert!(f > 0.4, "reach speed {f}");
        assert!(lateral < 0.4 * f, "lateral {lateral} vs forward {f}");
    }

    #[test]
    fn paddle_speed_in_any_direction() {
        for heading in [0.0f32, 90.0, 180.0] {
            let mut r = RaftBody::at(Vec3::ZERO, heading.to_radians());
            run(&mut r, Vec2::new(0.0, 5.0), &controls(1.0, 0.0, false), 60.0, &deep);
            let f = fwd_speed(&r);
            assert!((0.25..=1.05).contains(&f), "heading {heading}: {f}");
            let mut calm = RaftBody::at(Vec3::ZERO, heading.to_radians());
            run(&mut calm, Vec2::ZERO, &controls(1.0, 0.0, false), 60.0, &deep);
            assert!(
                (0.5..=0.8).contains(&fwd_speed(&calm)),
                "calm paddle {}",
                fwd_speed(&calm)
            );
        }
    }

    #[test]
    fn turning_is_gentle() {
        let mut r = RaftBody::at(Vec3::ZERO, 0.0);
        run(&mut r, Vec2::new(0.0, 5.0), &controls(0.0, 0.0, true), 30.0, &deep);
        let mut turned = 0.0f32;
        let mut prev = r.heading;
        for _ in 0..(24.0 / DT) as usize {
            r.step(&flat(), Vec2::new(0.0, 5.0), &controls(0.0, 1.0, true), &deep, DT);
            let mut d = r.heading - prev;
            d = (d + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
            assert!(
                d.abs() / DT <= 15f32.to_radians() + 1e-3,
                "yaw rate {} deg/s",
                (d / DT).to_degrees()
            );
            turned += d;
            prev = r.heading;
        }
        assert!(turned.abs() < std::f32::consts::TAU, "full turn in under 24 s");
        assert!(turned > 0.5, "steering right should turn clockwise: {turned}");
    }

    fn beach(p: Vec2) -> f32 {
        // Rises from -3 m at z = 10 to +1 m at z = 20.
        (-3.0 + (p.y - 10.0).clamp(0.0, 10.0) * 0.4).max(-25.0)
    }

    #[test]
    fn cannot_pass_through_terrain() {
        let mut r = RaftBody::at(Vec3::ZERO, 0.0);
        for _ in 0..(90.0 / DT) as usize {
            r.step(&flat(), Vec2::new(0.0, 5.0), &controls(0.0, 0.0, true), &beach, DT);
            for (p, o) in r.sample_points().iter().zip(RAFT_SAMPLE_OFFSETS) {
                let bottom = r.position.y + (r.rotation * o).y - RAFT_DRAFT_M;
                assert!(beach(*p) <= bottom + 0.05, "sank into the beach at {p}");
            }
        }
        assert!((12.0..19.0).contains(&r.position.z), "stopped at z {}", r.position.z);
        assert!(r.grounded, "should be aground");
    }

    #[test]
    fn paddle_frees_a_beached_raft() {
        // Poling straight back off the beach is quick; turning round first takes a gentle
        // 180° (~14 s) before paddling away.
        for (c, secs) in [(controls(-1.0, 0.0, false), 12.0), (controls(1.0, 1.0, false), 30.0)] {
            let mut r = RaftBody::at(Vec3::ZERO, 0.0);
            run(&mut r, Vec2::new(0.0, 5.0), &controls(0.0, 0.0, true), 90.0, &beach);
            assert!(r.grounded);
            let mut freed = false;
            for _ in 0..(secs / DT) as usize {
                r.step(&flat(), Vec2::new(0.0, 5.0), &c, &beach, DT);
                if -beach(Vec2::new(r.position.x, r.position.z)) >= 0.5 && !r.grounded {
                    freed = true;
                    break;
                }
            }
            assert!(freed, "{c:?} did not free the raft (z {})", r.position.z);
        }
    }

    #[test]
    fn stays_finite_at_extremes() {
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        for dt in [1.0 / 240.0, 1.0 / 30.0, 0.25] {
            let mut r = RaftBody::at(Vec3::ZERO, 0.3);
            for i in 0..1000 {
                let mut w = flat();
                for s in w.iter_mut() {
                    s.height = rnd() * 5.0;
                    s.velocity.y = rnd() * 3.0;
                }
                if i % 97 == 0 {
                    w[3].height = f32::NAN;
                }
                let c = controls(rnd(), rnd(), i % 2 == 0);
                r.step(&w, Vec2::new(rnd(), rnd()) * 25.0, &c, &beach, dt);
                assert!(
                    r.position.is_finite() && r.velocity.is_finite() && r.rotation.is_finite(),
                    "dt {dt}"
                );
                assert!((r.rotation.length() - 1.0).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn sample_points_follow_heading() {
        let r = RaftBody::at(Vec3::new(10.0, 0.0, 5.0), std::f32::consts::FRAC_PI_2);
        let pts = r.sample_points();
        assert!((pts[4] - Vec2::new(10.0, 5.0)).length() < 1e-4, "centre sample");
        // The bow-centre sample (local +z) points along +x at heading 90°.
        let bow = pts
            .iter()
            .cloned()
            .fold(Vec2::new(f32::MIN, 0.0), |a, b| if b.x > a.x { b } else { a });
        assert!(bow.x > 11.0);
    }
}
