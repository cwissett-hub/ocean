//! The four cameras: chase (behind and above, smoothly following the heading), deck (riding
//! the raft at eye height), high (the v1 sunset framing, following the raft) and free fly.
//! None of them goes below the water or into terrain.

use glam::{Vec2, Vec3};

use crate::scene::camera::Camera;
use crate::scene::raft::RaftBody;

pub const CAMERA_CLEARANCE_M: f32 = 0.6;
/// Top of the logs above the raft's waterline.
pub const DECK_HEIGHT_M: f32 = 0.15;
const EYE_HEIGHT_M: f32 = 1.65;
const CHASE_BACK_M: f32 = 10.0;
const CHASE_UP_M: f32 = 4.0;
const CHASE_OMEGA: f32 = 3.0;
const HIGH_BACK_M: f32 = 30.0;
const HIGH_UP_M: f32 = 18.0;
const HIGH_PITCH_DEG: f32 = -25.0;
const FREE_SPEED: f32 = 8.0;
const FREE_FAST: f32 = 4.0;
const FOV_Y_DEG: f32 = 60.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    Chase,
    Deck,
    High,
    FreeFly,
}

impl CameraMode {
    pub fn next(self) -> CameraMode {
        match self {
            CameraMode::Chase => CameraMode::Deck,
            CameraMode::Deck => CameraMode::High,
            CameraMode::High => CameraMode::FreeFly,
            CameraMode::FreeFly => CameraMode::Chase,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            CameraMode::Chase => "chase",
            CameraMode::Deck => "deck",
            CameraMode::High => "high",
            CameraMode::FreeFly => "free fly",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FreeFlyInput {
    /// x right, y up, z forward; each −1..1.
    pub move_local: Vec3,
    /// Yaw (x) and pitch (y) change, radians.
    pub look_delta: Vec2,
    pub fast: bool,
}

pub struct CameraController {
    pub mode: CameraMode,
    chase_yaw: f32,
    chase_yaw_vel: f32,
    free_pos: Vec3,
    free_yaw: f32,
    free_pitch: f32,
    last_mode: CameraMode,
    last: Option<Camera>,
}

fn dir(yaw: f32, pitch: f32) -> Vec3 {
    Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos())
}

fn wrap(a: f32) -> f32 {
    (a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

pub fn clamp_above(pos: Vec3, ground_height: f32, clearance: f32) -> Vec3 {
    if ground_height.is_finite() && pos.y < ground_height + clearance {
        Vec3::new(pos.x, ground_height + clearance, pos.z)
    } else {
        pos
    }
}

impl CameraController {
    pub fn new(raft: &RaftBody) -> CameraController {
        CameraController {
            mode: CameraMode::Chase,
            chase_yaw: raft.heading,
            chase_yaw_vel: 0.0,
            free_pos: Vec3::ZERO,
            free_yaw: raft.heading,
            free_pitch: 0.0,
            last_mode: CameraMode::Chase,
            last: None,
        }
    }

    /// Shifts every stored position by −`shift` (world origin rebase).
    pub fn rebase(&mut self, shift: Vec2) {
        let s = Vec3::new(shift.x, 0.0, shift.y);
        self.free_pos -= s;
        if let Some(l) = &mut self.last {
            l.position -= s;
        }
    }

    /// `ground(xz)` is the highest of the water estimate and the terrain there.
    pub fn update(
        &mut self,
        dt: f32,
        raft: &RaftBody,
        sun_azimuth_deg: f32,
        free: &FreeFlyInput,
        ground: &dyn Fn(Vec2) -> f32,
    ) -> Camera {
        let dt = if dt.is_finite() { dt.clamp(0.0, 0.25) } else { 0.0 };
        // The chase spring runs in every mode so switching back never swings.
        let err = wrap(raft.heading - self.chase_yaw);
        self.chase_yaw_vel += (CHASE_OMEGA * CHASE_OMEGA * err - 2.0 * CHASE_OMEGA * self.chase_yaw_vel) * dt;
        self.chase_yaw = wrap(self.chase_yaw + self.chase_yaw_vel * dt);

        if self.mode == CameraMode::FreeFly && self.last_mode != CameraMode::FreeFly {
            // Take off from wherever the previous camera was.
            if let Some(l) = self.last {
                self.free_pos = l.position;
                self.free_yaw = l.forward.x.atan2(l.forward.z);
                self.free_pitch = l.forward.y.clamp(-1.0, 1.0).asin();
            }
        }
        self.last_mode = self.mode;

        let (mut position, forward) = match self.mode {
            CameraMode::Chase => {
                let back = dir(self.chase_yaw, 0.0);
                let pos = raft.position - back * CHASE_BACK_M + Vec3::Y * CHASE_UP_M;
                (pos, (raft.position + Vec3::Y - pos).normalize())
            }
            CameraMode::Deck => {
                let eye = raft.position + raft.rotation * Vec3::new(0.0, DECK_HEIGHT_M + EYE_HEIGHT_M, -0.6);
                (eye, raft.rotation * Vec3::Z)
            }
            CameraMode::High => {
                let yaw = sun_azimuth_deg.to_radians();
                let pos = raft.position - dir(yaw, 0.0) * HIGH_BACK_M;
                (
                    Vec3::new(pos.x, HIGH_UP_M, pos.z),
                    dir(yaw, HIGH_PITCH_DEG.to_radians()),
                )
            }
            CameraMode::FreeFly => {
                self.free_yaw = wrap(self.free_yaw + free.look_delta.x);
                self.free_pitch = (self.free_pitch + free.look_delta.y).clamp(-1.48, 1.48);
                let f = dir(self.free_yaw, self.free_pitch);
                let right = Vec3::new(f.z, 0.0, -f.x).normalize_or_zero();
                let speed = FREE_SPEED * if free.fast { FREE_FAST } else { 1.0 };
                let m = free.move_local.clamp(Vec3::splat(-1.0), Vec3::ONE);
                self.free_pos += (right * m.x + Vec3::Y * m.y + f * m.z) * speed * dt;
                (self.free_pos, f)
            }
        };
        position = clamp_above(position, ground(Vec2::new(position.x, position.z)), CAMERA_CLEARANCE_M);
        if self.mode == CameraMode::FreeFly {
            self.free_pos = position;
        }
        let cam = Camera {
            position,
            forward: forward.normalize(),
            fov_y: FOV_Y_DEG.to_radians(),
        };
        self.last = Some(cam);
        cam
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::raft::RaftBody;
    use glam::{Vec2, Vec3};

    fn sea(_: Vec2) -> f32 {
        0.0
    }

    fn settle(c: &mut CameraController, r: &RaftBody, secs: f32) -> Camera {
        let mut cam = c.update(0.0, r, 30.0, &FreeFlyInput::default(), &sea);
        for _ in 0..(secs * 60.0) as usize {
            cam = c.update(1.0 / 60.0, r, 30.0, &FreeFlyInput::default(), &sea);
        }
        cam
    }

    fn yaw_of(v: Vec3) -> f32 {
        v.x.atan2(v.z)
    }

    #[test]
    fn chase_sits_behind_and_above() {
        let r = RaftBody::at(Vec3::new(5.0, 0.0, 7.0), 0.7);
        let mut c = CameraController::new(&r);
        let cam = settle(&mut c, &r, 3.0);
        let rel = cam.position - r.position;
        let behind = -rel.dot(r.forward());
        assert!((8.0..=12.0).contains(&behind), "behind {behind}");
        assert!((3.0..=5.0).contains(&rel.y), "height {}", rel.y);
        let to_raft = (r.position - cam.position).normalize();
        assert!(cam.forward.dot(to_raft) > 0.98, "not looking at the raft");
    }

    #[test]
    fn chase_follows_heading_smoothly() {
        let mut r = RaftBody::at(Vec3::ZERO, 0.0);
        let mut c = CameraController::new(&r);
        settle(&mut c, &r, 3.0);
        r.heading = std::f32::consts::FRAC_PI_2;
        let mut cam = c.update(1.0 / 60.0, &r, 30.0, &FreeFlyInput::default(), &sea);
        for _ in 0..14 {
            cam = c.update(1.0 / 60.0, &r, 30.0, &FreeFlyInput::default(), &sea);
        }
        let early = yaw_of(cam.forward).to_degrees();
        assert!(early.abs() < 30.0, "snapped: {early}° after 0.25 s");
        let late = yaw_of(settle(&mut c, &r, 3.0).forward).to_degrees();
        assert!(late > 80.0, "not following: {late}°");
    }

    #[test]
    fn deck_rides_the_raft() {
        let mut r = RaftBody::at(Vec3::new(0.0, 0.3, 0.0), 1.0);
        r.rotation = glam::Quat::from_rotation_y(1.0) * glam::Quat::from_rotation_z(0.1);
        let mut c = CameraController::new(&r);
        c.mode = CameraMode::Deck;
        let cam = c.update(1.0 / 60.0, &r, 30.0, &FreeFlyInput::default(), &sea);
        let above_deck = cam.position.y - (r.position.y + DECK_HEIGHT_M);
        assert!((1.4..=1.9).contains(&above_deck), "eye {above_deck} above deck");
        let fwd = r.rotation * Vec3::Z;
        assert!(cam.forward.dot(fwd) > 0.99, "deck camera should look along the raft");
    }

    #[test]
    fn high_view_keeps_raft_in_frame() {
        for az in [0.0, 30.0, 200.0] {
            let r = RaftBody::at(Vec3::new(100.0, 0.0, -40.0), 2.0);
            let mut c = CameraController::new(&r);
            c.mode = CameraMode::High;
            let cam = c.update(1.0 / 60.0, &r, az, &FreeFlyInput::default(), &sea);
            let p = cam.view_proj(16.0 / 9.0).project_point3(r.position);
            assert!(p.x.abs() < 0.8 && p.y.abs() < 0.8, "az {az}: raft at {p}");
            assert!((cam.position.y - 18.0).abs() < 0.5);
        }
    }

    #[test]
    fn free_fly_moves_and_looks() {
        let r = RaftBody::at(Vec3::ZERO, 0.0);
        let mut c = CameraController::new(&r);
        c.mode = CameraMode::FreeFly;
        let start = c.update(0.0, &r, 30.0, &FreeFlyInput::default(), &sea);
        let go = FreeFlyInput {
            move_local: Vec3::Z,
            ..Default::default()
        };
        let mut cam = start;
        for _ in 0..60 {
            cam = c.update(1.0 / 60.0, &r, 30.0, &go, &sea);
        }
        let moved = (cam.position - start.position).length();
        assert!((7.0..=9.0).contains(&moved), "moved {moved}");
        let fast = FreeFlyInput {
            move_local: Vec3::Z,
            fast: true,
            ..Default::default()
        };
        let before = cam.position;
        for _ in 0..60 {
            cam = c.update(1.0 / 60.0, &r, 30.0, &fast, &sea);
        }
        assert!((cam.position - before).length() > 25.0, "fast should be ~4x");
        let look = FreeFlyInput {
            look_delta: Vec2::new(0.5, 0.0),
            ..Default::default()
        };
        let turned = c.update(1.0 / 60.0, &r, 30.0, &look, &sea);
        assert!(turned.forward.dot(cam.forward) < 0.95, "look should turn the view");
    }

    #[test]
    fn no_camera_mode_goes_below_water_or_terrain() {
        let r = RaftBody::at(Vec3::ZERO, 0.0);
        let high_ground = |_: Vec2| 5.0;
        for mode in [
            CameraMode::Chase,
            CameraMode::Deck,
            CameraMode::High,
            CameraMode::FreeFly,
        ] {
            let mut c = CameraController::new(&r);
            c.mode = mode;
            let dive = FreeFlyInput {
                move_local: Vec3::new(0.0, -1.0, 0.0),
                fast: true,
                ..Default::default()
            };
            let mut cam = c.update(0.0, &r, 30.0, &dive, &high_ground);
            for _ in 0..240 {
                cam = c.update(1.0 / 60.0, &r, 30.0, &dive, &high_ground);
            }
            assert!(
                cam.position.y >= 5.0 + CAMERA_CLEARANCE_M - 1e-4,
                "{mode:?} at {}",
                cam.position.y
            );
        }
    }

    #[test]
    fn mode_cycles_in_order() {
        let order = [
            CameraMode::Chase,
            CameraMode::Deck,
            CameraMode::High,
            CameraMode::FreeFly,
        ];
        for (i, m) in order.iter().enumerate() {
            assert_eq!(m.next(), order[(i + 1) % 4]);
        }
    }
}
