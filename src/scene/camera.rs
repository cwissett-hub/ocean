use glam::{Mat4, Vec3};

pub const NEAR_M: f32 = 0.1;

/// Where the eye is and where it looks. y is up; azimuth 0 looks along +z.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    /// Vertical field of view, radians.
    pub fov_y: f32,
}

impl Camera {
    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_to_mat4(self.position, self.forward, Vec3::Y)
    }

    /// Reversed-Z infinite projection: depth 1 at the near plane, 0 at infinity.
    pub fn proj(&self, aspect: f32) -> Mat4 {
        glam::camera::rh::proj::directx::perspective_infinite_reverse(self.fov_y, aspect, NEAR_M)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect) * self.view()
    }
}

/// Anything that can place the camera each frame. v1 uses `FixedCamera`; free flight later
/// implements this too.
pub trait CameraRig {
    fn pose(&self) -> Camera;
}

#[derive(Debug, Clone, Copy)]
pub struct FixedCamera {
    pub height: f32,
    pub pitch_deg: f32,
    pub yaw_deg: f32,
    pub fov_y_deg: f32,
}

impl Default for FixedCamera {
    fn default() -> Self {
        FixedCamera {
            height: 18.0,
            pitch_deg: -25.0,
            yaw_deg: 30.0,
            fov_y_deg: 60.0,
        }
    }
}

impl CameraRig for FixedCamera {
    fn pose(&self) -> Camera {
        let (p, y) = (self.pitch_deg.to_radians(), self.yaw_deg.to_radians());
        Camera {
            position: Vec3::new(0.0, self.height, 0.0),
            forward: Vec3::new(y.sin() * p.cos(), p.sin(), y.cos() * p.cos()),
            fov_y: self.fov_y_deg.to_radians(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_camera_sits_above_water_looking_down() {
        let c = FixedCamera {
            yaw_deg: 30.0,
            ..Default::default()
        }
        .pose();
        assert_eq!(c.position.y, 18.0);
        let pitch = c.forward.y.asin().to_degrees();
        assert!((pitch + 25.0).abs() < 1e-3, "pitch {pitch}");
        assert!((c.forward.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn yaw_follows_azimuth_convention() {
        // Azimuth 0 looks along +z, 90 along +x (clockwise from above).
        let f0 = FixedCamera {
            yaw_deg: 0.0,
            ..Default::default()
        }
        .pose()
        .forward;
        let f90 = FixedCamera {
            yaw_deg: 90.0,
            ..Default::default()
        }
        .pose()
        .forward;
        assert!(f0.z > 0.0 && f0.x.abs() < 1e-5);
        assert!(f90.x > 0.0 && f90.z.abs() < 1e-5);
    }

    #[test]
    fn horizon_strip_is_visible_at_top_of_frame() {
        let c = FixedCamera::default().pose();
        let inv = c.view_proj(16.0 / 9.0).inverse();
        // Reversed-Z: NDC z = 1 is the near plane.
        let near = inv.project_point3(Vec3::new(0.0, 1.0, 1.0));
        let far = inv.project_point3(Vec3::new(0.0, 1.0, 0.5));
        let top_ray = (far - near).normalize();
        assert!(top_ray.y > 0.0, "top edge looks below the horizon: {top_ray}");
        let mid =
            (inv.project_point3(Vec3::new(0.0, 0.0, 0.5)) - inv.project_point3(Vec3::new(0.0, 0.0, 1.0))).normalize();
        assert!(mid.y < 0.0);
    }

    #[test]
    fn reversed_z_maps_near_to_one() {
        let c = FixedCamera::default().pose();
        let p = c.view_proj(1.0).project_point3(c.position + c.forward * NEAR_M);
        assert!((p.z - 1.0).abs() < 1e-4, "z {}", p.z);
    }
}
