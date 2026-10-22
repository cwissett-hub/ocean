//! The driftwood raft: buoyancy on nine water samples, a self-trimming square sail, a keel-like
//! hull, a paddle, gentle steering, and grounding on island terrain. All accelerations are per
//! unit mass.

use glam::{Mat4, Quat, Vec2, Vec3};

use crate::render::particles::{KIND_SPRAY, SpawnRequest};
use crate::waves::query::WaterSample;
use crate::waves::spectrum::G;

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

/// A light raft drawing ~15 cm bobs at about sqrt(g / draft) ≈ 8 rad/s: it rides the surface.
const HEAVE_OMEGA: f32 = 7.0;
const HEAVE_ZETA: f32 = 0.5;
const ROLL_ZETA: f32 = 0.5;
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
const GROUND_MU: f32 = 0.6;
/// Logs ploughing into sand: extra drag per metre of penetration.
const PLOUGH_DRAG: f32 = 30.0;
/// Aground, the paddle becomes a pole pushed against the sand.
const PUSH_OFF_GAIN: f32 = 80.0;

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
    /// Distance sailed over the ground, m (the wake is laid by it).
    pub travelled: f32,
    /// How much of `travelled` the last `step` added.
    pub last_step_travel: f32,
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
            travelled: 0.0,
            last_step_travel: 0.0,
        }
    }

    /// Follows an origin rebase (world::origin): local positions move by −shift.
    pub fn rebase(&mut self, shift: Vec2) {
        self.position -= Vec3::new(shift.x, 0.0, shift.y);
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
        self.last_step_travel = 0.0;
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
            let mut afloat = 0.0;
            for (s, o) in water.iter().zip(RAFT_SAMPLE_OFFSETS) {
                if !(s.height.is_finite() && s.velocity.is_finite()) {
                    continue;
                }
                let r = self.rotation * o;
                let vy = self.velocity.y + self.angular_velocity.cross(r).y;
                let immersion = s.height - (self.position.y + r.y);
                // Out of the water a log only weighs g; the spring must not pull it down harder
                // (that pinned a beached raft to the sand).
                let f = (HEAVE_OMEGA * HEAVE_OMEGA * immersion + 2.0 * HEAVE_ZETA * HEAVE_OMEGA * (s.velocity.y - vy))
                    .max(-G);
                if immersion > -RAFT_DRAFT_M {
                    afloat += 1.0;
                }
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
            // A beached raft is driven only as much as it still floats: the wind must not
            // carry it up the dunes.
            let floating = (afloat / RAFT_SAMPLE_OFFSETS.len() as f32).max(if self.grounded { 0.0 } else { 1.0 });
            let mut force = sail_force(apparent, fwd, c.sail_up) * floating
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
            let travel = Vec2::new(self.velocity.x, self.velocity.z).length() * h;
            self.travelled += travel;
            self.last_step_travel += travel;
            // Yaw is tracked by `heading`; the quaternion integrates only pitch and roll. The
            // tilt lives in the yaw frame (rotation = yaw · tilt): split it off with the heading
            // the rotation was built from, before turning, and turn the world-axis angular
            // velocity into that frame.
            let tilt = self.tilt();
            self.heading = (self.heading + self.angular_velocity.y * h).rem_euclid(std::f32::consts::TAU);
            let tilt_w = Quat::from_rotation_y(self.heading).inverse()
                * Vec3::new(self.angular_velocity.x, 0.0, self.angular_velocity.z);
            let tilt = if tilt_w.length_squared() > 0.0 {
                (Quat::from_scaled_axis(tilt_w * h) * tilt).normalize()
            } else {
                tilt
            };
            // Keep only the swing (where "up" points): composing small pitch and roll steps
            // leaves a net twist about the vertical, and yaw belongs to `heading` alone.
            let tilt = Quat::from_rotation_arc(Vec3::Y, (tilt * Vec3::Y).normalize());
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

/// The bow throws spray when it slams down into the water while under way.
const SPRAY_MIN_SPEED: f32 = 0.6;
const SPRAY_MIN_PLUNGE: f32 = 0.4;
/// Particles per second per (m/s of plunge beyond the minimum × m/s forward), per bow point.
const SPRAY_RATE: f32 = 40.0;
const SPRAY_MAX_PER_POINT: u32 = 2048;
const BOW_POINTS: [usize; 3] = [6, 7, 8];

/// What the raft throws into the particle system after a `step` of `dt`: bow spray. (Its wake
/// and the foam the wake breaks into come from the wake simulation, render::wake.)
pub fn raft_emissions(body: &RaftBody, water: &[WaterSample; 9], dt: f32) -> Vec<SpawnRequest> {
    let mut out = Vec::new();
    if !(dt.is_finite() && dt > 0.0) {
        return out;
    }
    let fwd = body.forward();
    let side = Vec3::new(fwd.z, 0.0, -fwd.x);
    let speed = body.velocity.dot(fwd);
    if speed <= SPRAY_MIN_SPEED {
        return out;
    }
    for i in BOW_POINTS {
        let r = body.rotation * RAFT_SAMPLE_OFFSETS[i];
        let p = body.position + r;
        let w = &water[i];
        let submerged = w.height > p.y - RAFT_DRAFT_M;
        let plunge = -(body.velocity.y + body.angular_velocity.cross(r).y - w.velocity.y) - SPRAY_MIN_PLUNGE;
        if !(submerged && plunge > 0.0) {
            continue;
        }
        let count = (plunge * speed * SPRAY_RATE * dt)
            .ceil()
            .min(SPRAY_MAX_PER_POINT as f32) as u32;
        let x = RAFT_SAMPLE_OFFSETS[i].x;
        let outward = if x == 0.0 { 0.0 } else { x.signum() };
        let v = fwd * (0.7 * speed) + side * (1.2 * outward) + Vec3::Y * (1.2 + 1.5 * plunge);
        out.push(SpawnRequest {
            position: (p + fwd * 0.2 + Vec3::Y * 0.05).to_array(),
            kind: KIND_SPRAY,
            velocity: v.to_array(),
            count,
        });
    }
    out
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
    fn beached_on_the_real_island_in_any_wind_it_can_be_got_off() {
        // Sail onto the start island's windward beach in an onshore wind, then get off it:
        // poling back (sail down) or turning round and paddling. Seen by review: in 10 m/s
        // the raft was stranded for good, in 25 m/s it sailed up the dunes.
        use crate::world::islands::{WORLD_SEED, seabed_height, start_island};
        let isle = start_island(WORLD_SEED);
        let ground = |p: Vec2| seabed_height(WORLD_SEED, std::slice::from_ref(&isle), p.as_dvec2());
        let dir = isle.center.normalize().as_vec2();
        let start = isle.center.as_vec2() - dir * 120.0;
        let heading = dir.x.atan2(dir.y);
        for wind_speed in [5.0f32, 10.0, 25.0] {
            let wind = dir * wind_speed;
            let strategies = [
                ("pole back, sail down", controls(-1.0, 0.0, false), 20.0),
                ("turn and paddle, sail down", controls(1.0, 1.0, false), 45.0),
            ];
            for (name, c, secs) in strategies {
                let mut r = RaftBody::at(Vec3::new(start.x, 0.0, start.y), heading);
                run(&mut r, wind, &controls(0.0, 0.0, true), 150.0, &ground);
                let under = ground(Vec2::new(r.position.x, r.position.z));
                assert!(r.grounded, "{wind_speed} m/s: never reached the beach");
                assert!(under < 0.6, "{wind_speed} m/s: sailed {under} m up the island");
                let mut freed_after = None;
                for i in 0..(secs / DT) as usize {
                    r.step(&flat(), wind, &c, &ground, DT);
                    // Afloat over water deeper than twice the draft. (With the sail up in an
                    // onshore wind nobody claws off a lee shore: drop it first.)
                    if ground(Vec2::new(r.position.x, r.position.z)) < -0.3 && !r.grounded {
                        freed_after = Some(i as f32 * DT);
                        break;
                    }
                }
                assert!(
                    freed_after.is_some(),
                    "{wind_speed} m/s, {name}: still aground after {secs} s"
                );
            }
        }
    }

    #[test]
    fn hull_keeps_its_heading_over_a_long_rough_sail() {
        // Composing many small pitch and roll steps leaves a net twist about the vertical;
        // unless it is removed, the drawn hull drifts off the heading (review: +20–40°/hour).
        let waves: Vec<(Vec2, f32, f32)> = (0..16)
            .map(|i| {
                let a = i as f32 * 2.399;
                let k = 0.05 + 0.03 * (i % 5) as f32;
                (Vec2::new(a.sin(), a.cos()) * k, (G * k).sqrt(), i as f32 * 1.7)
            })
            .collect();
        let water_at = |p: Vec2, t: f32| -> (f32, f32) {
            waves.iter().fold((0.0, 0.0), |(h, v), (k, w, ph)| {
                let phase = k.dot(p) - w * t + ph;
                (h + 0.18 * phase.cos(), v + 0.18 * w * phase.sin())
            })
        };
        for heading in [0.0f32, 2.4] {
            let mut r = RaftBody::at(Vec3::ZERO, heading);
            let mut worst = 0.0f32;
            let mut worst_twist = 0.0f32;
            for i in 0..(3600.0 / DT) as usize {
                let t = i as f32 * DT;
                let mut w = flat();
                for (s, p) in w.iter_mut().zip(r.sample_points()) {
                    let (h, v) = water_at(p, t);
                    s.height = h;
                    s.velocity.y = v;
                }
                r.step(&w, Vec2::ZERO, &controls(0.0, 0.0, false), &deep, DT);
                let f = r.rotation * Vec3::Z;
                let drawn = f.x.atan2(f.z);
                let off =
                    (drawn - r.heading + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
                worst = worst.max(off.abs().to_degrees());
                // Twist of the tilt about the vertical (swing-twist split): should stay zero.
                let q = r.tilt();
                worst_twist = worst_twist.max((2.0 * q.y.atan2(q.w)).abs().to_degrees());
            }
            assert!(worst < 2.0, "heading {heading}: hull drifted {worst}° off its heading");
            assert!(
                worst_twist < 0.5,
                "heading {heading}: tilt twisted {worst_twist}° about the vertical"
            );
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
    fn follows_wave_slope_quickly() {
        // A wide flat raft is stiff in roll: it takes the slope of the water within a second.
        let mut r = RaftBody::at(Vec3::ZERO, 0.7);
        for _ in 0..60 {
            let mut w = flat();
            for (s, p) in w.iter_mut().zip(r.sample_points()) {
                s.height = 0.2 * p.x;
            }
            r.step(&w, Vec2::ZERO, &controls(0.0, 0.0, false), &deep, DT);
        }
        let tilt = (r.rotation * Vec3::Y).angle_between(Vec3::Y).to_degrees();
        assert!(tilt > 0.8 * 0.2f32.atan().to_degrees(), "tilt {tilt}° after 1 s");
    }

    #[test]
    fn settles_onto_the_water_slope_at_any_heading() {
        // A static slope rising along +x: the deck must come to lie on it, whichever way the
        // raft points (a frame mix-up made the torque push the wrong way off heading 0).
        let normal = Vec3::new(-0.2, 1.0, 0.0).normalize();
        for heading in [0.0f32, 0.7, 1.6, 3.5, 5.0] {
            let mut r = RaftBody::at(Vec3::ZERO, heading);
            for _ in 0..(5.0 / DT) as usize {
                let mut w = flat();
                for (s, p) in w.iter_mut().zip(r.sample_points()) {
                    s.height = 0.2 * p.x;
                    s.normal = normal;
                }
                r.step(&w, Vec2::ZERO, &controls(0.0, 0.0, false), &deep, DT);
            }
            let off = (r.rotation * Vec3::Y).angle_between(normal).to_degrees();
            assert!(off < 1.5, "heading {heading}: deck {off}° off the water slope");
        }
    }

    #[test]
    fn hull_turns_with_its_heading() {
        // The drawn hull (rotation) must point where the physics says (heading) after a turn.
        let mut r = RaftBody::at(Vec3::ZERO, 0.0);
        r.velocity = r.forward() * 2.0;
        run(&mut r, Vec2::ZERO, &controls(1.0, 1.0, false), 10.0, &deep);
        assert!(r.heading > 0.5, "barely turned: {}", r.heading);
        let off = (r.rotation * Vec3::Z).angle_between(r.forward()).to_degrees();
        assert!(off < 1.0, "hull points {off}° away from the heading");
    }

    #[test]
    fn heave_tracks_a_passing_wave_closely() {
        // 0.3 m, 4 s waves: a light raft rides on the surface, not behind it.
        let mut r = RaftBody::at(Vec3::ZERO, 0.0);
        let w = std::f32::consts::TAU / 4.0;
        let mut worst = 0.0f32;
        for i in 0..(20.0 / DT) as usize {
            let t = i as f32 * DT;
            let mut s = flat();
            for x in s.iter_mut() {
                x.height = 0.3 * (w * t).sin();
                x.velocity.y = 0.3 * w * (w * t).cos();
            }
            r.step(&s, Vec2::ZERO, &controls(0.0, 0.0, false), &deep, DT);
            if t > 8.0 {
                worst = worst.max((r.position.y - 0.3 * (w * t).sin()).abs());
            }
        }
        assert!(worst < 0.04, "raft strays {worst} m from the surface");
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

    fn spray(reqs: &[SpawnRequest]) -> u32 {
        reqs.iter().filter(|r| r.kind == KIND_SPRAY).map(|r| r.count).sum()
    }

    fn marks(reqs: &[SpawnRequest]) -> usize {
        reqs.iter()
            .filter(|r| r.kind == crate::render::particles::KIND_FOAM_MARK)
            .count()
    }

    fn moving(speed: f32, vy: f32) -> RaftBody {
        let mut r = RaftBody::at(Vec3::ZERO, 0.3);
        r.velocity = r.forward() * speed + Vec3::new(0.0, vy, 0.0);
        r
    }

    #[test]
    fn bow_spray_only_when_plunging_at_speed() {
        assert!(
            spray(&raft_emissions(&moving(2.5, -1.2), &flat(), DT)) > 0,
            "fast, bow plunging"
        );
        assert_eq!(spray(&raft_emissions(&moving(0.3, -1.2), &flat(), DT)), 0, "too slow");
        assert_eq!(spray(&raft_emissions(&moving(2.5, 0.6), &flat(), DT)), 0, "bow rising");
        assert_eq!(
            spray(&raft_emissions(&moving(2.5, -0.2), &flat(), DT)),
            0,
            "gentle pitch"
        );
        // Out of the water (riding over a trough), the bow throws nothing.
        let mut high = moving(2.5, -1.2);
        high.position.y = 0.6;
        assert_eq!(spray(&raft_emissions(&high, &flat(), DT)), 0, "bow in the air");
        // Harder impacts throw more.
        let soft = spray(&raft_emissions(&moving(2.5, -0.7), &flat(), DT));
        let hard = spray(&raft_emissions(&moving(2.5, -2.0), &flat(), DT));
        assert!(hard > soft, "soft {soft} hard {hard}");
        for r in raft_emissions(&moving(2.5, -1.2), &flat(), DT) {
            if r.kind == KIND_SPRAY {
                let p = Vec3::from(r.position);
                assert!(p.dot(moving(1.0, 0.0).forward()) > 1.0, "spray from the bow, not {p}");
                assert!(r.velocity[1] > 0.5, "thrown upwards");
            }
        }
    }

    #[test]
    fn raft_lays_no_foam_lanes_the_wake_simulation_makes_its_foam() {
        // The straight three-lane foam marks doubled up with the simulated Kelvin wake.
        let mut r = RaftBody::at(Vec3::ZERO, 1.0);
        r.velocity = r.forward() * 2.0;
        for _ in 0..(8.0 / DT) as usize {
            r.step(&flat(), Vec2::ZERO, &controls(1.0, 0.0, false), &deep, DT);
            assert_eq!(marks(&raft_emissions(&r, &flat(), DT)), 0);
        }
    }

    #[test]
    fn raft_state_survives_rebase() {
        use crate::scene::buoy::BuoyBody;
        use crate::scene::cameras::{CameraController, FreeFlyInput};
        let s = Vec2::new(4096.0, -1024.0);
        let sea = |_: Vec2| 0.3;
        let c = controls(0.6, 0.3, true);
        let wind = Vec2::new(3.0, 5.0);
        let make = || {
            let mut r = RaftBody::at(Vec3::new(4200.0, 0.1, -900.0), 0.7);
            r.velocity = Vec3::new(0.8, -0.1, 1.2);
            let cam = CameraController::new(&r);
            let buoy = BuoyBody::at(Vec3::new(4180.0, 0.0, -880.0));
            (r, cam, buoy)
        };
        let step = |r: &mut RaftBody, cam: &mut CameraController, buoy: &mut BuoyBody| {
            for _ in 0..30 {
                r.step(&flat(), wind, &c, &deep, DT);
                buoy.step(&[flat()[0]; 5], DT);
                cam.update(DT, r, 30.0, &FreeFlyInput::default(), &sea);
            }
        };
        let shift3 = Vec3::new(s.x, 0.0, s.y);
        // A: rebase, then step.
        let (mut ra, mut ca, mut ba) = make();
        ca.update(DT, &ra, 30.0, &FreeFlyInput::default(), &sea);
        ra.rebase(s);
        ca.rebase(s);
        ba.position -= shift3;
        step(&mut ra, &mut ca, &mut ba);
        let cam_a = ca.update(DT, &ra, 30.0, &FreeFlyInput::default(), &sea);
        // B: step, then rebase.
        let (mut rb, mut cb, mut bb) = make();
        cb.update(DT, &rb, 30.0, &FreeFlyInput::default(), &sea);
        step(&mut rb, &mut cb, &mut bb);
        rb.rebase(s);
        cb.rebase(s);
        bb.position -= shift3;
        let cam_b = cb.update(DT, &rb, 30.0, &FreeFlyInput::default(), &sea);
        assert!(
            (ra.position - rb.position).length() < 1e-3,
            "{} vs {}",
            ra.position,
            rb.position
        );
        assert!((ra.velocity - rb.velocity).length() < 1e-4);
        assert!((ra.heading - rb.heading).abs() < 1e-5);
        assert!((ba.position - bb.position).length() < 1e-3);
        assert!(
            (cam_a.position - cam_b.position).length() < 1e-3,
            "{} vs {}",
            cam_a.position,
            cam_b.position
        );
    }

    #[test]
    fn emissions_respect_request_budget() {
        use crate::render::particles::{REQUEST_BUDGET, pack_requests};
        // A violent ten-second step: far more spray than one frame may spawn.
        let mut r = moving(6.0, -5.0);
        r.last_step_travel = 60.0;
        r.travelled = 60.0;
        let reqs = raft_emissions(&r, &flat(), 10.0);
        let wanted: u32 = reqs.iter().map(|q| q.count).sum();
        assert!(wanted > REQUEST_BUDGET, "only {wanted} requested");
        let packed = pack_requests(&reqs, REQUEST_BUDGET);
        assert!(packed.iter().map(|q| q.count).sum::<u32>() <= REQUEST_BUDGET);
    }
}
