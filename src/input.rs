//! Keyboard and mouse state → raft controls and free-fly input (pure; the window just feeds it).

use std::collections::HashSet;

use glam::{Vec2, Vec3};
use winit::keyboard::KeyCode;

use crate::scene::cameras::FreeFlyInput;
use crate::scene::raft::RaftControls;

/// Radians of look per pixel of mouse drag.
const LOOK_PER_PIXEL: f32 = 0.003;

#[derive(Debug, Default)]
pub struct KeyState {
    pressed: HashSet<KeyCode>,
    pub right_mouse: bool,
    mouse_delta: Vec2,
}

impl KeyState {
    pub fn press(&mut self, k: KeyCode) {
        self.pressed.insert(k);
    }

    pub fn release(&mut self, k: KeyCode) {
        self.pressed.remove(&k);
    }

    pub fn mouse_moved(&mut self, delta_px: Vec2) {
        if self.right_mouse {
            self.mouse_delta += delta_px;
        }
    }

    fn axis(&self, neg: KeyCode, pos: KeyCode) -> f32 {
        (self.pressed.contains(&pos) as i32 - self.pressed.contains(&neg) as i32) as f32
    }

    pub fn raft_controls(&self, sail_up: bool) -> RaftControls {
        RaftControls {
            steer: self.axis(KeyCode::KeyA, KeyCode::KeyD),
            paddle: self.axis(KeyCode::KeyS, KeyCode::KeyW),
            sail_up,
        }
    }

    /// Free-fly movement and look; consumes the accumulated mouse drag.
    pub fn free_fly(&mut self) -> FreeFlyInput {
        let d = std::mem::take(&mut self.mouse_delta);
        FreeFlyInput {
            move_local: Vec3::new(
                self.axis(KeyCode::KeyA, KeyCode::KeyD),
                self.axis(KeyCode::KeyQ, KeyCode::KeyE),
                self.axis(KeyCode::KeyS, KeyCode::KeyW),
            ),
            // Drag right turns right (clockwise yaw); drag up looks up.
            look_delta: Vec2::new(d.x, -d.y) * LOOK_PER_PIXEL,
            fast: self.pressed.contains(&KeyCode::ShiftLeft) || self.pressed.contains(&KeyCode::ShiftRight),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::KeyCode;

    #[test]
    fn key_state_maps_raft_controls() {
        let mut k = KeyState::default();
        k.press(KeyCode::KeyA);
        assert_eq!(k.raft_controls(true).steer, -1.0);
        k.press(KeyCode::KeyD);
        assert_eq!(k.raft_controls(true).steer, 0.0, "A+D cancel");
        k.release(KeyCode::KeyA);
        assert_eq!(k.raft_controls(true).steer, 1.0);
        k.press(KeyCode::KeyW);
        assert_eq!(k.raft_controls(false).paddle, 1.0);
        k.press(KeyCode::KeyS);
        assert_eq!(k.raft_controls(false).paddle, 0.0);
        assert!(!k.raft_controls(false).sail_up);
    }

    #[test]
    fn free_fly_input_consumes_mouse_only_while_dragging() {
        let mut k = KeyState::default();
        k.mouse_moved(glam::Vec2::new(100.0, 0.0));
        assert_eq!(k.free_fly().look_delta, glam::Vec2::ZERO, "no drag, no look");
        k.right_mouse = true;
        k.mouse_moved(glam::Vec2::new(100.0, -50.0));
        let f = k.free_fly();
        // Dragging right turns right; dragging up (negative screen y) looks up.
        assert!(f.look_delta.x > 0.0 && f.look_delta.y > 0.0);
        assert_eq!(k.free_fly().look_delta, glam::Vec2::ZERO, "delta consumed");
        k.press(KeyCode::KeyE);
        k.press(KeyCode::ShiftLeft);
        let f = k.free_fly();
        assert_eq!(f.move_local.y, 1.0);
        assert!(f.fast);
    }
}
