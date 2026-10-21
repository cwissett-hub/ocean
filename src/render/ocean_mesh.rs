//! Camera-centred nested grids (geometry-clipmap style). Every level is the same
//! (GRID+1)² vertex grid, scaled by 2^level and snapped to twice its cell size so vertices
//! never swim. The shader discards each level where the next finer level covers it and
//! geomorphs the outer band of each level onto the coarser grid, so edges meet watertight.
//! `ocean_gbuffer.wgsl` mirrors `level_cell` and `level_center`.

use glam::Vec2;

pub const GRID: u32 = 128;
pub const LEVELS: u32 = 11;
pub const BASE_CELL_M: f32 = 0.25;

pub fn level_cell(level: u32) -> f32 {
    BASE_CELL_M * (1u32 << level) as f32
}

pub fn level_center(cam_xz: Vec2, level: u32) -> Vec2 {
    let snap = 2.0 * level_cell(level);
    (cam_xz / snap).round() * snap
}

/// Triangle list for one level; quads split along (i, j)–(i+1, j+1).
pub fn grid_indices() -> Vec<u32> {
    let v = GRID + 1;
    let mut out = Vec::with_capacity((GRID * GRID * 6) as usize);
    for j in 0..GRID {
        for i in 0..GRID {
            let a = j * v + i;
            out.extend_from_slice(&[a, a + 1, a + v + 1, a, a + v + 1, a + v]);
        }
    }
    out
}

pub struct OceanMesh {
    pub indices: wgpu::Buffer,
    pub index_count: u32,
}

impl OceanMesh {
    pub fn new(device: &wgpu::Device) -> OceanMesh {
        use wgpu::util::DeviceExt;
        let idx = grid_indices();
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ocean grid indices"),
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::INDEX,
        });
        OceanMesh { indices, index_count: idx.len() as u32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn indices_cover_grid_with_one_diagonal() {
        let idx = grid_indices();
        let v = GRID + 1;
        assert_eq!(idx.len() as u32, GRID * GRID * 6);
        assert!(idx.iter().all(|&i| i < v * v));
        // Every quad is split along (i, j)–(i+1, j+1); morphing relies on that diagonal.
        for q in idx.chunks_exact(6) {
            let a = q[0];
            assert!(q.contains(&(a + v + 1)), "quad at {a} not split on the 00-11 diagonal");
        }
    }

    #[test]
    fn finer_level_edges_lie_on_coarser_grid_lines() {
        for cam in [Vec2::ZERO, Vec2::new(3.3, -7.9), Vec2::new(1234.56, 98.7)] {
            for l in 1..LEVELS {
                let (s, c, cf) = (level_cell(l), level_center(cam, l), level_center(cam, l - 1));
                let half_fine = GRID as f32 / 2.0 * level_cell(l - 1);
                for edge in [cf - half_fine, cf + half_fine] {
                    let g = (edge - c) / s;
                    assert!((g - g.round()).abs().max_element() < 1e-3, "level {l} cam {cam}: {g}");
                }
            }
        }
    }

    #[test]
    fn each_level_encloses_the_finer_one() {
        for cam in [Vec2::ZERO, Vec2::new(-55.5, 17.25)] {
            for l in 1..LEVELS {
                let half = GRID as f32 / 2.0 * level_cell(l);
                let half_fine = GRID as f32 / 2.0 * level_cell(l - 1);
                let d = (level_center(cam, l - 1) - level_center(cam, l)).abs();
                assert!((d + half_fine).max_element() <= half - level_cell(l) + 1e-3, "level {l}");
            }
        }
    }

    #[test]
    fn coarsest_level_reaches_past_the_horizon() {
        // Horizon distance from 18 m is ≈ 15 km.
        assert!(GRID as f32 / 2.0 * level_cell(LEVELS - 1) > 15_000.0);
    }
}
