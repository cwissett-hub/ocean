//! A driftwood raft: seven uneven logs lashed side by side, a short mast and yard, and a square
//! sail with some billow. Local frame: +z forward, y = 0 the waterline.

use glam::Vec3;

use crate::scene::mesh::{MeshData, material};

fn jitter(i: u32, salt: u32) -> f32 {
    let mut x = i
        .wrapping_mul(747_796_405)
        .wrapping_add(salt.wrapping_mul(2_891_336_453));
    x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277_803_737);
    ((x >> 22) ^ x) as f32 / u32::MAX as f32
}

pub fn build_raft_mesh() -> MeshData {
    let mut m = MeshData::default();
    for i in 0..7u32 {
        let x = -1.08 + 0.36 * i as f32;
        let r = 0.14 + 0.06 * jitter(i, 1);
        let len = 2.8 + 0.4 * jitter(i, 2);
        let off = (jitter(i, 3) - 0.5) * 0.25;
        let tilt = (jitter(i, 4) - 0.5) * 0.08;
        let a = Vec3::new(x, -0.03, -len / 2.0 + off);
        let b = Vec3::new(x + tilt, -0.03 + tilt * 0.3, len / 2.0 + off);
        m.add_tube(a, b, r, 12, material::RAFT_WOOD, jitter(i, 5));
    }
    // Cross lashings.
    for z in [-1.0f32, 1.0] {
        m.add_tube(
            Vec3::new(-1.25, 0.12, z),
            Vec3::new(1.25, 0.12, z),
            0.035,
            6,
            material::RAFT_WOOD,
            0.9,
        );
    }
    // Mast and yard.
    m.add_tube(
        Vec3::new(0.0, 0.1, 0.2),
        Vec3::new(0.0, 2.7, 0.2),
        0.06,
        8,
        material::RAFT_WOOD,
        0.6,
    );
    m.add_tube(
        Vec3::new(-0.95, 2.55, 0.25),
        Vec3::new(0.95, 2.55, 0.25),
        0.04,
        6,
        material::RAFT_WOOD,
        0.6,
    );
    // Square sail hanging from the yard, billowing forward; param = height down the sail.
    let (cols, rows) = (8u32, 8u32);
    let at = |u: f32, v: f32| {
        let billow = 0.28 * (std::f32::consts::PI * u).sin() * (std::f32::consts::PI * v).sin();
        Vec3::new(-0.9 + 1.8 * u, 2.5 - 1.6 * v, 0.3 + billow)
    };
    let base = m.vertices.len() as u32;
    for j in 0..=rows {
        for i in 0..=cols {
            let (u, v) = (i as f32 / cols as f32, j as f32 / rows as f32);
            let e = 1e-3;
            let du = at((u + e).min(1.0), v) - at((u - e).max(0.0), v);
            let dv = at(u, (v + e).min(1.0)) - at(u, (v - e).max(0.0));
            let n = du.cross(dv).normalize();
            m.vertex(at(u, v), n, material::SAIL, v);
        }
    }
    for j in 0..rows {
        for i in 0..cols {
            let a = base + j * (cols + 1) + i;
            let c = a + cols + 1;
            m.indices.extend_from_slice(&[a, c, a + 1, a + 1, c, c + 1]);
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::mesh::material;
    use glam::Vec3;

    #[test]
    fn raft_mesh_is_sane() {
        let m = build_raft_mesh();
        let n = m.vertices.len() as u32;
        assert!(!m.indices.is_empty() && m.indices.len().is_multiple_of(3));
        assert!(m.indices.iter().all(|&i| i < n));
        for t in m.indices.chunks_exact(3) {
            let p = |i: u32| Vec3::from_slice(&m.vertices[i as usize][0..3]);
            assert!(
                (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).length() > 1e-9,
                "degenerate {t:?}"
            );
        }
        for v in &m.vertices {
            assert!((Vec3::from_slice(&v[3..6]).length() - 1.0).abs() < 1e-3, "unit normals");
            let id = v[6] as u32;
            assert!(id == material::RAFT_WOOD || id == material::SAIL, "material {id}");
        }
        let (lo, hi) = m.bounds();
        let footprint = hi - lo;
        assert!((2.6..=3.4).contains(&footprint.z), "length {}", footprint.z);
        assert!((2.0..=2.8).contains(&footprint.x), "beam {}", footprint.x);
        assert!((2.4..=3.0).contains(&hi.y), "mast top {}", hi.y);
        assert!(lo.y < 0.0, "logs sit in the water");
        assert!(m.indices.len() / 3 < 8000, "{} triangles", m.indices.len() / 3);
        assert!(m.vertices.iter().any(|v| v[6] as u32 == material::SAIL));
    }
}
