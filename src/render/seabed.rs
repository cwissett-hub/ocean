//! The seabed around the camera as a heightmap the shaders sample: wave attenuation in the
//! shallows, the visible sandy bottom and the shore foam band. Baked on the CPU from exactly
//! the height functions the raft physics uses (world::islands), on a worker thread, and
//! re-baked when the map's snapped centre or the set of nearby islands changes.

use std::sync::mpsc;

use glam::DVec2;

use crate::world::islands::{IslandDesc, island_height, sea_floor};
use crate::world::origin::WorldOrigin;

pub const SEABED_RES: usize = 1024;
pub const SEABED_TEXEL_M: f64 = 4.0;
/// The map recentres in steps of this, so texel centres always land on the same world points.
pub const SEABED_SNAP_M: f64 = 64.0;
pub const SEABED_EXTENT_M: f64 = SEABED_RES as f64 * SEABED_TEXEL_M;
pub const SEABED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// World position of the map's (0, 0) corner when it is centred near `focus`.
pub fn map_corner(focus: DVec2) -> DVec2 {
    (focus / SEABED_SNAP_M).round() * SEABED_SNAP_M - DVec2::splat(SEABED_EXTENT_M / 2.0)
}

/// World position of texel (i, j)'s centre.
pub fn texel_center(corner: DVec2, i: usize, j: usize) -> DVec2 {
    corner + (DVec2::new(i as f64, j as f64) + 0.5) * SEABED_TEXEL_M
}

/// Heights at every texel centre (row-major, j = z rows): world::islands::seabed_height, but
/// each island is only evaluated inside its own 3-radius footprint.
pub fn bake_seabed(seed: u64, islands: &[IslandDesc], corner: DVec2) -> Vec<f32> {
    let mut h = Vec::with_capacity(SEABED_RES * SEABED_RES);
    for j in 0..SEABED_RES {
        for i in 0..SEABED_RES {
            h.push(sea_floor(seed, texel_center(corner, i, j)));
        }
    }
    for d in islands {
        let reach = 3.0 * d.radius as f64;
        let lo = ((d.center - reach - corner) / SEABED_TEXEL_M).floor().max(DVec2::ZERO);
        let hi = ((d.center + reach - corner) / SEABED_TEXEL_M)
            .ceil()
            .min(DVec2::splat(SEABED_RES as f64));
        for j in lo.y as usize..hi.y.max(lo.y) as usize {
            for i in lo.x as usize..hi.x.max(lo.x) as usize {
                let v = &mut h[j * SEABED_RES + i];
                *v = v.max(island_height(d, texel_center(corner, i, j)));
            }
        }
    }
    h
}

/// What a bake depends on: where the map sits and which islands are in it.
#[derive(Debug, Clone, PartialEq)]
struct BakeKey {
    corner: DVec2,
    islands: Vec<u64>,
}

fn bake_key(corner: DVec2, islands: &[IslandDesc]) -> BakeKey {
    let mut ids: Vec<u64> = islands.iter().map(|d| d.id).collect();
    ids.sort_unstable();
    BakeKey { corner, islands: ids }
}

pub struct SeabedMap {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// World corner of the uploaded map; None until the first upload.
    corner: Option<DVec2>,
    /// Key of the newest bake requested (uploaded or still on the worker).
    wanted: Option<BakeKey>,
    jobs: mpsc::Sender<(u64, Vec<IslandDesc>, DVec2)>,
    done: mpsc::Receiver<(DVec2, Vec<f32>)>,
}

impl SeabedMap {
    pub fn new(device: &wgpu::Device) -> SeabedMap {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("seabed"),
            size: wgpu::Extent3d {
                width: SEABED_RES as u32,
                height: SEABED_RES as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SEABED_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let (jobs, rx) = mpsc::channel::<(u64, Vec<IslandDesc>, DVec2)>();
        let (tx, done) = mpsc::channel();
        std::thread::Builder::new()
            .name("seabed baker".into())
            .spawn(move || {
                while let Ok(mut job) = rx.recv() {
                    // Only the newest request matters.
                    while let Ok(newer) = rx.try_recv() {
                        job = newer;
                    }
                    let (seed, islands, corner) = job;
                    if tx.send((corner, bake_seabed(seed, &islands, corner))).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn seabed baker");
        SeabedMap {
            texture,
            view,
            corner: None,
            wanted: None,
            jobs,
            done,
        }
    }

    /// Replaces the map with `heights` (SEABED_RES², row-major) whose corner is at `corner`.
    pub fn upload(&mut self, queue: &wgpu::Queue, corner: DVec2, heights: &[f32]) {
        assert_eq!(heights.len(), SEABED_RES * SEABED_RES);
        queue.write_texture(
            self.texture.as_image_copy(),
            bytemuck::cast_slice(heights),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SEABED_RES as u32 * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: SEABED_RES as u32,
                height: SEABED_RES as u32,
                depth_or_array_layers: 1,
            },
        );
        self.corner = Some(corner);
    }

    /// Keeps the map centred near `focus` (world) over `islands`. The first bake runs here so
    /// the shallows are right from the first frame; later ones run on the worker and are
    /// uploaded when they finish.
    pub fn update(&mut self, queue: &wgpu::Queue, seed: u64, focus: DVec2, islands: &[IslandDesc]) {
        let key = bake_key(map_corner(focus), islands);
        if self.corner.is_none() {
            let heights = bake_seabed(seed, islands, key.corner);
            self.upload(queue, key.corner, &heights);
            self.wanted = Some(key);
        } else if self.wanted.as_ref() != Some(&key) {
            let _ = self.jobs.send((seed, islands.to_vec(), key.corner));
            self.wanted = Some(key);
        }
        if let Some((corner, heights)) = self.done.try_iter().last() {
            self.upload(queue, corner, &heights);
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Shader parameters: local corner x, z; texel size m; resolution (0 = no map yet, which
    /// the shaders read as open sea everywhere).
    pub fn params(&self, origin: &WorldOrigin) -> [f32; 4] {
        match self.corner {
            Some(c) => {
                let l = origin.to_local(c);
                [l.x, l.y, SEABED_TEXEL_M as f32, SEABED_RES as f32]
            }
            None => [0.0; 4],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::islands::{WORLD_SEED, seabed_height, start_island};

    #[test]
    fn map_corner_snaps_and_centres() {
        let c = map_corner(DVec2::new(100.0, -31.0));
        assert_eq!(c, DVec2::new(128.0, 0.0) - SEABED_EXTENT_M / 2.0);
        let texels = (c - map_corner(DVec2::new(1234.5, 77.0))) / SEABED_TEXEL_M;
        assert_eq!(texels, texels.round(), "recentring moves by whole texels");
    }

    #[test]
    fn bake_matches_seabed_height_at_texel_centres() {
        let isle = start_island(WORLD_SEED);
        let islands = vec![isle.clone()];
        let corner = map_corner(isle.center);
        let h = bake_seabed(WORLD_SEED, &islands, corner);
        assert_eq!(h.len(), SEABED_RES * SEABED_RES);
        for k in 0..500usize {
            let (i, j) = ((k * 7919) % SEABED_RES, 400 + (k * 31) % 220);
            let want = seabed_height(WORLD_SEED, &islands, texel_center(corner, i, j));
            assert_eq!(h[j * SEABED_RES + i], want, "texel ({i}, {j})");
        }
        let mid = ((isle.center - corner) / SEABED_TEXEL_M).floor();
        assert!(
            h[mid.y as usize * SEABED_RES + mid.x as usize] > 1.0,
            "island above water"
        );
    }
}
