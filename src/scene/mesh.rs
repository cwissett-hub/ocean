//! The vertex format every object shares (position, normal, material id, per-vertex param) and a
//! few primitives to build meshes from.

use glam::{Vec2, Vec3};

/// Material ids, shared by the G-buffer (`material.g = id / 8`) and the TLAS custom data.
pub mod material {
    pub const WATER: u32 = 0;
    pub const BUOY: u32 = 1;
    pub const RAFT_WOOD: u32 = 2;
    pub const SAIL: u32 = 3;
    pub const SAND: u32 = 4;
    pub const PALM_TRUNK: u32 = 5;
    pub const PALM_FROND: u32 = 6;
}

#[derive(Debug, Clone, Default)]
pub struct MeshData {
    /// pos (3), normal (3), material id, param (0..1, material-specific).
    pub vertices: Vec<[f32; 8]>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn vertex(&mut self, p: Vec3, n: Vec3, material: u32, param: f32) -> u32 {
        self.vertices
            .push([p.x, p.y, p.z, n.x, n.y, n.z, material as f32, param]);
        self.vertices.len() as u32 - 1
    }

    pub fn append(&mut self, other: &MeshData) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        self.vertices
            .iter()
            .fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), v| {
                let p = Vec3::from_slice(&v[0..3]);
                (lo.min(p), hi.max(p))
            })
    }

    /// A capped tube from `a` to `b`.
    pub fn add_tube(&mut self, a: Vec3, b: Vec3, radius: f32, segments: u32, material: u32, param: f32) {
        let axis = (b - a).normalize();
        let helper = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let u = axis.cross(helper).normalize();
        let v = axis.cross(u);
        let ring = |i: u32| {
            let t = std::f32::consts::TAU * i as f32 / segments as f32;
            u * t.cos() + v * t.sin()
        };
        let base = self.vertices.len() as u32;
        for i in 0..segments {
            let n = ring(i);
            self.vertex(a + n * radius, n, material, param);
            self.vertex(b + n * radius, n, material, param);
        }
        for i in 0..segments {
            let (p, q) = (base + 2 * i, base + 2 * ((i + 1) % segments));
            self.indices.extend_from_slice(&[p, q, p + 1, p + 1, q, q + 1]);
        }
        for (centre, n) in [(a, -axis), (b, axis)] {
            let c = self.vertex(centre, n, material, param);
            for i in 0..segments {
                self.vertex(centre + ring(i) * radius, n, material, param);
            }
            for i in 0..segments {
                let (p, q) = (c + 1 + i, c + 1 + (i + 1) % segments);
                if n == axis {
                    self.indices.extend_from_slice(&[c, p, q]);
                } else {
                    self.indices.extend_from_slice(&[c, q, p]);
                }
            }
        }
    }

    /// A vertical capped cylinder centred at `center` (xz) from `y0` to `y1`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_cylinder(
        &mut self,
        center: Vec2,
        radius: f32,
        y0: f32,
        y1: f32,
        segments: u32,
        material: u32,
        param: f32,
    ) {
        self.add_tube(
            Vec3::new(center.x, y0, center.y),
            Vec3::new(center.x, y1, center.y),
            radius,
            segments,
            material,
            param,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};

    #[test]
    fn append_offsets_indices() {
        let mut a = MeshData::default();
        a.add_cylinder(Vec2::ZERO, 1.0, 0.0, 1.0, 8, material::BUOY, 0.0);
        let (va, ia) = (a.vertices.len(), a.indices.len());
        let mut b = MeshData::default();
        b.add_cylinder(Vec2::new(5.0, 0.0), 0.5, 0.0, 2.0, 6, material::SAIL, 1.0);
        a.append(&b);
        assert_eq!(a.vertices.len(), va + b.vertices.len());
        assert!(a.indices[ia..].iter().all(|&i| i as usize >= va));
        assert!(a.indices.iter().all(|&i| (i as usize) < a.vertices.len()));
    }

    #[test]
    fn bounds_cover_all_vertices() {
        let mut m = MeshData::default();
        m.add_cylinder(Vec2::new(1.0, 2.0), 0.5, -1.0, 3.0, 12, material::BUOY, 0.0);
        let (lo, hi) = m.bounds();
        assert!((lo - Vec3::new(0.5, -1.0, 1.5)).abs().max_element() < 1e-4, "{lo}");
        assert!((hi - Vec3::new(1.5, 3.0, 2.5)).abs().max_element() < 1e-4, "{hi}");
    }

    #[test]
    fn vertices_carry_material_and_param() {
        let mut m = MeshData::default();
        m.add_cylinder(Vec2::ZERO, 1.0, 0.0, 1.0, 8, material::RAFT_WOOD, 0.25);
        assert!(
            m.vertices
                .iter()
                .all(|v| v[6] == material::RAFT_WOOD as f32 && v[7] == 0.25)
        );
    }
}
