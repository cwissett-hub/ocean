//! Which islands exist near the raft, which have meshes, and the worker thread that builds
//! them, so a whole archipelago coming into range never stalls a frame.

use std::collections::HashMap;
use std::sync::mpsc;

use glam::{DVec2, Mat4};

use crate::scene::mesh::MeshData;
use crate::world::islands::{IslandDesc, islands_near, seabed_height};
use crate::world::terrain_mesh::{island_mesh, palm_transforms};

pub const LOAD_M: f64 = 4000.0;
pub const UNLOAD_M: f64 = 4500.0;
pub const LOD0_IN_M: f64 = 1500.0;
pub const LOD1_OUT_M: f64 = 1700.0;

#[derive(Debug, Default, PartialEq)]
pub struct StreamPlan {
    pub load: Vec<(IslandDesc, u32)>,
    pub unload: Vec<u64>,
    pub relod: Vec<(u64, u32)>,
}

fn wanted_lod(dist: f64, current: Option<u32>) -> u32 {
    match current {
        Some(0) if dist <= LOD1_OUT_M => 0,
        Some(1) if dist >= LOD0_IN_M => 1,
        _ => {
            if dist < (LOD0_IN_M + LOD1_OUT_M) / 2.0 {
                0
            } else {
                1
            }
        }
    }
}

/// Mesh work for this frame: loads (nearest first) and LOD changes, at most `max_new` of them;
/// unloads are free. Hysteresis between load/unload and between LODs prevents thrash.
pub fn plan_streaming(loaded: &HashMap<u64, u32>, visible: &[IslandDesc], focus: DVec2, max_new: usize) -> StreamPlan {
    let dist = |d: &IslandDesc| (d.center - focus).length();
    let mut plan = StreamPlan::default();
    let mut work: Vec<(f64, Option<IslandDesc>, u64, u32)> = Vec::new();
    for d in visible {
        let r = dist(d);
        match loaded.get(&d.id) {
            None if r < LOAD_M => work.push((r, Some(d.clone()), d.id, wanted_lod(r, None))),
            Some(&lod) => {
                let want = wanted_lod(r, Some(lod));
                if want != lod {
                    work.push((r, None, d.id, want));
                }
            }
            None => {}
        }
    }
    for id in loaded.keys() {
        let far = visible.iter().find(|d| d.id == *id).is_none_or(|d| dist(d) > UNLOAD_M);
        if far {
            plan.unload.push(*id);
        }
    }
    plan.unload.sort_unstable();
    work.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, desc, id, lod) in work.into_iter().take(max_new) {
        match desc {
            Some(d) => plan.load.push((d, lod)),
            None => plan.relod.push((id, lod)),
        }
    }
    plan
}

/// A finished island mesh, island-local.
pub struct ReadyIsland {
    pub desc: IslandDesc,
    pub lod: u32,
    pub mesh: MeshData,
    pub palms: Vec<Mat4>,
}

pub struct IslandStreamer {
    seed: u64,
    jobs: mpsc::Sender<(IslandDesc, u32)>,
    done: mpsc::Receiver<ReadyIsland>,
    /// Islands with a mesh built or being built, and that mesh's LOD.
    requested: HashMap<u64, u32>,
    /// Every island within UNLOAD_M of the focus (for terrain height, with or without a mesh).
    known: Vec<IslandDesc>,
    unloads: Vec<u64>,
    max_new: usize,
}

impl IslandStreamer {
    pub fn new(seed: u64) -> IslandStreamer {
        let (jobs, rx) = mpsc::channel::<(IslandDesc, u32)>();
        let (tx, done) = mpsc::channel();
        std::thread::Builder::new()
            .name("island mesher".into())
            .spawn(move || {
                for (desc, lod) in rx {
                    let mesh = island_mesh(&desc, lod);
                    let palms = palm_transforms(&desc);
                    if tx.send(ReadyIsland { desc, lod, mesh, palms }).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn island mesher");
        IslandStreamer {
            seed,
            jobs,
            done,
            requested: HashMap::new(),
            known: Vec::new(),
            unloads: Vec::new(),
            max_new: 2,
        }
    }

    /// Re-plans around `focus` (world): queues mesh jobs and collects unloads.
    pub fn update(&mut self, focus: DVec2) {
        self.known = islands_near(self.seed, focus, UNLOAD_M);
        let plan = plan_streaming(&self.requested, &self.known, focus, self.max_new);
        for id in plan.unload {
            self.requested.remove(&id);
            self.unloads.push(id);
        }
        for (d, lod) in plan.load {
            self.requested.insert(d.id, lod);
            let _ = self.jobs.send((d, lod));
        }
        for (id, lod) in plan.relod {
            if let Some(d) = self.known.iter().find(|d| d.id == id) {
                self.requested.insert(id, lod);
                let _ = self.jobs.send((d.clone(), lod));
            }
        }
    }

    /// Meshes finished since the last call (drop any for islands unloaded meanwhile).
    pub fn take_ready(&mut self) -> Vec<ReadyIsland> {
        self.done
            .try_iter()
            .filter(|r| self.requested.get(&r.desc.id) == Some(&r.lod))
            .collect()
    }

    pub fn take_unloads(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.unloads)
    }

    /// Ground height at a world point: sea floor or island, whether or not its mesh exists.
    pub fn terrain_height(&self, world: DVec2) -> f32 {
        seabed_height(self.seed, &self.known, world)
    }

    pub fn known(&self) -> &[IslandDesc] {
        &self.known
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::islands::IslandDesc;
    use glam::DVec2;
    use std::collections::HashMap;

    fn isle(id: u64, x: f64) -> IslandDesc {
        IslandDesc {
            id,
            center: DVec2::new(x, 0.0),
            radius: 40.0,
            height: 5.0,
            palms: 1,
            seed: id,
        }
    }

    #[test]
    fn streaming_loads_nearest_first_and_unloads_far() {
        let visible = vec![isle(1, 3000.0), isle(2, 500.0), isle(3, 2000.0)];
        let mut loaded = HashMap::new();
        loaded.insert(9u64, 0u32); // no longer visible
        let p = plan_streaming(&loaded, &visible, DVec2::ZERO, 2);
        assert_eq!(p.load.iter().map(|(d, _)| d.id).collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(p.load[0].1, 0, "near island at full detail");
        assert_eq!(p.load[1].1, 1, "2 km island at the coarse LOD");
        assert_eq!(p.unload, vec![9]);
    }

    #[test]
    fn streaming_plan_limits_work_per_frame() {
        let visible: Vec<_> = (0..6).map(|i| isle(i, 300.0 + 120.0 * i as f64)).collect();
        let mut loaded = HashMap::new();
        let mut rounds = 0;
        while loaded.len() < 6 {
            let p = plan_streaming(&loaded, &visible, DVec2::ZERO, 2);
            assert!(p.load.len() + p.relod.len() <= 2);
            for (d, lod) in p.load {
                loaded.insert(d.id, lod);
            }
            rounds += 1;
            assert!(rounds < 10);
        }
        assert_eq!(rounds, 3);
    }

    #[test]
    fn hysteresis_prevents_thrash() {
        let visible = vec![isle(1, 4000.0)];
        let mut loaded = HashMap::new();
        loaded.insert(1u64, 1u32);
        for x in [-50.0, 50.0, -50.0, 50.0] {
            let p = plan_streaming(&loaded, &visible, DVec2::new(x, 0.0), 4);
            assert!(p.load.is_empty() && p.unload.is_empty(), "churn at offset {x}");
        }
        loaded.insert(1, 0);
        for x in [2400.0, 2350.0] {
            // ~1.6 km away: between the LOD thresholds, keep whatever it has.
            assert!(
                plan_streaming(&loaded, &visible, DVec2::new(x, 0.0), 4)
                    .relod
                    .is_empty()
            );
        }
    }

    #[test]
    fn lod_switches_with_distance() {
        let visible = vec![isle(1, 1000.0)];
        let mut loaded = HashMap::new();
        loaded.insert(1u64, 1u32);
        let p = plan_streaming(&loaded, &visible, DVec2::ZERO, 4);
        assert_eq!(p.relod, vec![(1, 0)]);
    }
}
