//! Island meshes (heightfield grids in island-local coordinates, two LODs) and palm trees.

use glam::{DVec2, Mat4, Quat, Vec3};

use crate::scene::mesh::{MeshData, material};
use crate::waves::rng::Pcg32;
use crate::world::islands::{IslandDesc, island_height};

/// The mesh reaches this many radii from the centre: the beach and the first metres of the
/// shallows; the deep seabed is drawn by the water shading, not geometry.
const EXTENT_RADII: f32 = 1.6;
const LOD0_CELL_M: f32 = 0.75;
const LOD0_MAX_QUADS: u32 = 160;
const LOD1_CELL_M: f32 = 4.0;
/// Sand param = height / this (materials.wgsl darkens wet sand below 0.05, i.e. 0.4 m).
const SAND_PARAM_SCALE: f32 = 8.0;
const MIN_PALM_GROUND_M: f32 = 1.5;

/// Grid over the island's square, vertices exactly on `island_height`, normals from central
/// differences. Island-local: (0, 0) is the island centre, y is height above sea level.
pub fn island_mesh(d: &IslandDesc, lod: u32) -> MeshData {
    let half = EXTENT_RADII * d.radius;
    let quads = if lod == 0 {
        ((2.0 * half / LOD0_CELL_M).ceil() as u32).min(LOD0_MAX_QUADS)
    } else {
        ((2.0 * half / LOD1_CELL_M).ceil() as u32).max(4)
    };
    let cell = 2.0 * half / quads as f32;
    let height = |x: f32, z: f32| island_height(d, d.center + DVec2::new(x as f64, z as f64));
    let mut m = MeshData::default();
    for j in 0..=quads {
        for i in 0..=quads {
            let (x, z) = (-half + i as f32 * cell, -half + j as f32 * cell);
            let y = height(x, z);
            let e = cell * 0.5;
            let dx = (height(x + e, z) - height(x - e, z)) / (2.0 * e);
            let dz = (height(x, z + e) - height(x, z - e)) / (2.0 * e);
            let n = Vec3::new(-dx, 1.0, -dz).normalize();
            m.vertex(
                Vec3::new(x, y, z),
                n,
                material::SAND,
                (y / SAND_PARAM_SCALE).clamp(0.0, 1.0),
            );
        }
    }
    let v = quads + 1;
    for j in 0..quads {
        for i in 0..quads {
            let a = j * v + i;
            m.indices.extend_from_slice(&[a, a + v, a + v + 1, a, a + v + 1, a + 1]);
        }
    }
    m
}

/// A coconut palm: a gently curving 7 m trunk and eight drooping fronds. Origin at the base.
pub fn palm_mesh() -> MeshData {
    let mut m = MeshData::default();
    let segments = 7;
    let trunk = |t: f32| Vec3::new(0.9 * t * t, 7.0 * t, 0.0);
    for s in 0..segments {
        let (t0, t1) = (s as f32 / segments as f32, (s + 1) as f32 / segments as f32);
        let r = 0.2 - 0.08 * t0;
        m.add_tube(trunk(t0), trunk(t1), r, 8, material::PALM_TRUNK, t0);
    }
    let top = trunk(1.0);
    for f in 0..8 {
        let a = std::f32::consts::TAU * f as f32 / 8.0 + 0.2;
        let dir = Vec3::new(a.cos(), 0.0, a.sin());
        let side = Vec3::new(-a.sin(), 0.0, a.cos());
        let steps = 6;
        let base = m.vertices.len() as u32;
        for k in 0..=steps {
            let t = k as f32 / steps as f32;
            // Out 3 m, rising then drooping.
            let p = top + dir * (3.0 * t) + Vec3::Y * (0.6 * t - 1.6 * t * t);
            let width = 0.55 * (std::f32::consts::PI * t).sin() + 0.05;
            let n = side.cross(dir + Vec3::Y * (0.6 - 3.2 * t)).normalize();
            let n = if n.y < 0.0 { -n } else { n };
            m.vertex(p - side * width, n, material::PALM_FROND, t);
            m.vertex(p + side * width, n, material::PALM_FROND, t);
        }
        for k in 0..steps {
            let q = base + 2 * k;
            m.indices.extend_from_slice(&[q, q + 2, q + 1, q + 1, q + 2, q + 3]);
        }
    }
    m
}

/// Where this island's palms stand (island-local), on dune ground at least 1.5 m high.
pub fn palm_transforms(d: &IslandDesc) -> Vec<Mat4> {
    let mut rng = Pcg32::new(d.seed ^ 0xBA1A);
    let mut out = Vec::new();
    let mut tries = 0;
    while (out.len() as u32) < d.palms && tries < 400 {
        tries += 1;
        let a = rng.next_f32() * std::f32::consts::TAU;
        let r = rng.next_f32().sqrt() * 0.7 * d.radius;
        let (x, z) = (r * a.cos(), r * a.sin());
        let ground = island_height(d, d.center + DVec2::new(x as f64, z as f64));
        if ground < MIN_PALM_GROUND_M {
            continue;
        }
        let yaw = rng.next_f32() * std::f32::consts::TAU;
        let lean = (rng.next_f32() - 0.5) * 0.28;
        let rot = Quat::from_rotation_y(yaw) * Quat::from_rotation_z(lean);
        out.push(Mat4::from_rotation_translation(rot, Vec3::new(x, ground - 0.2, z)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::mesh::material;
    use crate::world::islands::{WORLD_SEED, island_height, start_island};
    use glam::{DVec2, Vec3};

    #[test]
    fn island_mesh_matches_height_fn() {
        let d = start_island(WORLD_SEED);
        let m = island_mesh(&d, 0);
        assert!(!m.indices.is_empty());
        for v in m.vertices.iter().step_by(7) {
            let w = d.center + DVec2::new(v[0] as f64, v[2] as f64);
            let h = island_height(&d, w);
            assert!((v[1] - h).abs() < 0.01, "vertex {} vs height {h}", v[1]);
            assert_eq!(v[6] as u32, material::SAND);
        }
    }

    #[test]
    fn island_mesh_covers_beach_and_shallows() {
        let d = start_island(WORLD_SEED);
        let (lo, hi) = island_mesh(&d, 0).bounds();
        assert!(hi.y > 1.0, "dunes above water");
        assert!(lo.y < -5.0, "reaches into the shallows");
        assert!(hi.x - lo.x >= 2.5 * d.radius, "extent {}", hi.x - lo.x);
    }

    #[test]
    fn lod1_is_much_smaller() {
        let d = start_island(WORLD_SEED);
        let (a, b) = (island_mesh(&d, 0).indices.len(), island_mesh(&d, 1).indices.len());
        assert!(b * 10 < a, "lod0 {a} lod1 {b}");
    }

    #[test]
    fn palms_stand_on_dunes() {
        let d = start_island(WORLD_SEED);
        let t = palm_transforms(&d);
        assert_eq!(t.len() as u32, d.palms);
        for m in t {
            let base = m.transform_point3(Vec3::ZERO);
            let ground = island_height(&d, d.center + DVec2::new(base.x as f64, base.z as f64));
            assert!(ground >= 1.5, "palm on ground {ground}");
            assert!((base.y - ground).abs() < 0.4, "palm base {} vs ground {ground}", base.y);
        }
    }

    #[test]
    fn palm_mesh_is_sane() {
        let m = palm_mesh();
        let n = m.vertices.len() as u32;
        assert!(m.indices.iter().all(|&i| i < n));
        let (lo, hi) = m.bounds();
        assert!((6.0..=10.0).contains(&(hi.y - lo.y)), "height {}", hi.y - lo.y);
        assert!(m.vertices.iter().any(|v| v[6] as u32 == material::PALM_FROND));
        assert!(m.vertices.iter().any(|v| v[6] as u32 == material::PALM_TRUNK));
    }
}
