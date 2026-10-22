//! Procedural desert islands over an unbounded world: deterministic placement per 2 km cell
//! (lone islands, sometimes an archipelago), a dune-and-beach height profile, and the sandy
//! sea floor 20–30 m down everywhere.

use glam::{DVec2, IVec2};

use crate::waves::rng::Pcg32;

pub const CELL_M: f64 = 2048.0;
pub const WORLD_SEED: u64 = 0x0CEA_2026;
pub const SEA_DEPTH_M: f32 = 25.0;
/// Probability that a cell holds an island site, and that a site is an archipelago.
const SITE_P: f32 = 0.6;
const ARCHIPELAGO_P: f32 = 0.15;
/// Other sites keep this far from the start island so it stands alone at the start.
const START_CLEARANCE_M: f64 = 1200.0;

#[derive(Debug, Clone, PartialEq)]
pub struct IslandDesc {
    pub id: u64,
    pub center: DVec2,
    pub radius: f32,
    pub height: f32,
    pub palms: u32,
    pub seed: u64,
}

fn mix(mut z: u64) -> u64 {
    // splitmix64 finaliser.
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn cell_hash(seed: u64, cell: IVec2) -> u64 {
    mix(seed ^ mix((cell.x as i64 as u64) ^ mix(cell.y as i64 as u64).rotate_left(17)))
}

fn hash01(seed: u64, i: i64, j: i64) -> f64 {
    (mix(seed ^ mix(i as u64 ^ mix(j as u64).rotate_left(29))) >> 11) as f64 / (1u64 << 53) as f64
}

/// Smooth value noise in [-1, 1].
fn value_noise(seed: u64, p: DVec2) -> f64 {
    let (i, j) = (p.x.floor(), p.y.floor());
    let (fx, fy) = (p.x - i, p.y - j);
    let (ux, uy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let h = |a: f64, b: f64| hash01(seed, (i + a) as i64, (j + b) as i64) * 2.0 - 1.0;
    let top = h(0.0, 0.0) + (h(1.0, 0.0) - h(0.0, 0.0)) * ux;
    let bottom = h(0.0, 1.0) + (h(1.0, 1.0) - h(0.0, 1.0)) * ux;
    top + (bottom - top) * uy
}

pub fn start_island(seed: u64) -> IslandDesc {
    let dir = DVec2::new(30f64.to_radians().sin(), 30f64.to_radians().cos());
    IslandDesc {
        id: mix(seed ^ 0x5747),
        center: dir * 400.0,
        radius: 45.0,
        height: 6.0,
        palms: 2,
        seed: mix(seed ^ 0x51),
    }
}

fn island(rng: &mut Pcg32, id: u64, center: DVec2, radius: (f32, f32), height: (f32, f32)) -> IslandDesc {
    let r = radius.0 + rng.next_f32() * (radius.1 - radius.0);
    let palms = if r > 25.0 {
        (rng.next_f32() * 4.0) as u32
    } else {
        (rng.next_f32() * 1.5) as u32
    };
    IslandDesc {
        id,
        center,
        radius: r,
        height: height.0 + rng.next_f32() * (height.1 - height.0),
        palms,
        seed: mix(id ^ 0xd00d),
    }
}

/// All islands whose site lies in `cell` (deterministic).
pub fn islands_in_cell(seed: u64, cell: IVec2) -> Vec<IslandDesc> {
    let start = start_island(seed);
    if cell == IVec2::ZERO {
        return vec![start];
    }
    let h = cell_hash(seed, cell);
    let mut rng = Pcg32::new(h);
    if rng.next_f32() >= SITE_P {
        return Vec::new();
    }
    let origin = cell.as_dvec2() * CELL_M;
    let site = origin + DVec2::new(0.2 + 0.6 * rng.next_f32() as f64, 0.2 + 0.6 * rng.next_f32() as f64) * CELL_M;
    if (site - start.center).length() < START_CLEARANCE_M {
        return Vec::new();
    }
    if rng.next_f32() >= ARCHIPELAGO_P {
        return vec![island(&mut rng, mix(h), site, (25.0, 75.0), (3.0, 8.0))];
    }
    let count = 3 + (rng.next_f32() * 4.0) as usize;
    let mut members = vec![island(&mut rng, mix(h), site, (15.0, 45.0), (2.0, 6.0))];
    let mut tries = 0;
    while members.len() < count && tries < 200 {
        tries += 1;
        let parent = members[(rng.next_f32() * members.len() as f32) as usize % members.len()].center;
        let a = rng.next_f32() as f64 * std::f64::consts::TAU;
        let c = parent + DVec2::new(a.cos(), a.sin()) * (150.0 + 230.0 * rng.next_f32() as f64);
        if members.iter().all(|m| (m.center - c).length() >= 120.0) {
            let id = mix(h ^ members.len() as u64);
            members.push(island(&mut rng, id, c, (15.0, 45.0), (2.0, 6.0)));
        }
    }
    members
}

/// Islands within `radius_m` of `world` (centre distance, plus their own extent).
pub fn islands_near(seed: u64, world: DVec2, radius_m: f64) -> Vec<IslandDesc> {
    let reach = radius_m + CELL_M;
    let lo = ((world - DVec2::splat(reach)) / CELL_M).floor().as_ivec2();
    let hi = ((world + DVec2::splat(reach)) / CELL_M).floor().as_ivec2();
    let mut out = Vec::new();
    for y in lo.y..=hi.y {
        for x in lo.x..=hi.x {
            out.extend(
                islands_in_cell(seed, IVec2::new(x, y))
                    .into_iter()
                    .filter(|d| (d.center - world).length() <= radius_m + 3.0 * d.radius as f64),
            );
        }
    }
    out
}

/// Ground height of one island: dunes above sea level inside its radius, a beach sloping to the
/// sea floor by 2.5 radii, no contribution beyond 3 radii.
pub fn island_height(d: &IslandDesc, world: DVec2) -> f32 {
    let r = (world - d.center).length() / d.radius as f64;
    if r > 3.0 {
        return f32::NEG_INFINITY;
    }
    if r <= 1.0 {
        let base = d.height as f64 * (1.0 - r * r);
        let t = ((1.0 - r) / 0.5).clamp(0.0, 1.0);
        let fade = t * t * (3.0 - 2.0 * t);
        let local = world - d.center;
        let dunes = 0.6 * value_noise(d.seed, local / 11.0) + 0.4 * value_noise(d.seed ^ 7, local / 4.5);
        return (base + 0.3 * d.height as f64 * dunes * fade).max(0.0) as f32;
    }
    let beach = ((r - 1.0) / 1.5).min(1.0);
    -(SEA_DEPTH_M * beach as f32)
}

/// The open sea floor: 25 m down with a few metres of smooth sandy relief.
pub fn sea_floor(seed: u64, world: DVec2) -> f32 {
    let n = 3.2 * value_noise(seed ^ 0xf100, world / 400.0) + 1.0 * value_noise(seed ^ 0xf200, world / 120.0);
    (-(SEA_DEPTH_M as f64) + n) as f32
}

/// How much of the open-sea wave displacement survives over water `depth` metres deep (the
/// shaders' `wave_attenuation` in seabed_common.wgsl is the same formula): waves die out over
/// the last 10 m to the shore and never ride up onto dry land.
pub fn wave_attenuation(depth: f32) -> f32 {
    let t = (depth / 10.0).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Seabed (or island ground) height at a world point.
pub fn seabed_height(seed: u64, islands: &[IslandDesc], world: DVec2) -> f32 {
    islands
        .iter()
        .map(|d| island_height(d, world))
        .fold(sea_floor(seed, world), f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec2, IVec2};

    fn sites(seed: u64) -> Vec<Vec<IslandDesc>> {
        let mut out = Vec::new();
        for y in -8..=8 {
            for x in -8..=8 {
                let v = islands_in_cell(seed, IVec2::new(x, y));
                if !v.is_empty() {
                    out.push(v);
                }
            }
        }
        out
    }

    fn centroid(s: &[IslandDesc]) -> DVec2 {
        s.iter().map(|d| d.center).sum::<DVec2>() / s.len() as f64
    }

    #[test]
    fn attenuation_is_monotonic_and_bounded() {
        assert_eq!(wave_attenuation(-3.0), 0.0, "no waves over dry land");
        assert_eq!(wave_attenuation(0.0), 0.0);
        assert_eq!(wave_attenuation(10.0), 1.0);
        assert_eq!(wave_attenuation(SEA_DEPTH_M), 1.0, "open sea is untouched");
        assert!(wave_attenuation(1.0) < 0.05, "a 1 m shoal all but stops waves");
        let mut prev = 0.0;
        for i in 0..=200 {
            let a = wave_attenuation(i as f32 * 0.06);
            assert!(
                (0.0..=1.0).contains(&a) && a >= prev,
                "depth {} -> {a}",
                i as f32 * 0.06
            );
            prev = a;
        }
    }

    #[test]
    fn placement_is_deterministic() {
        let c = IVec2::new(3, -2);
        assert_eq!(islands_in_cell(WORLD_SEED, c), islands_in_cell(WORLD_SEED, c));
        let a: Vec<_> = (0..20).map(|i| islands_in_cell(1, IVec2::new(i, 0))).collect();
        let b: Vec<_> = (0..20).map(|i| islands_in_cell(2, IVec2::new(i, 0))).collect();
        assert_ne!(a, b);
    }

    #[test]
    fn start_island_is_ahead_under_the_sun() {
        let s = start_island(WORLD_SEED);
        let want = DVec2::new(30f64.to_radians().sin(), 30f64.to_radians().cos()) * 400.0;
        assert!((s.center - want).length() < 1.0, "{}", s.center);
        assert!(islands_in_cell(WORLD_SEED, IVec2::ZERO).contains(&s));
    }

    #[test]
    fn lone_islands_are_about_two_km_apart() {
        let c: Vec<DVec2> = sites(WORLD_SEED).iter().map(|s| centroid(s)).collect();
        let nn: Vec<f64> = c
            .iter()
            .enumerate()
            .map(|(i, a)| {
                c.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, b)| (*a - *b).length())
                    .fold(f64::MAX, f64::min)
            })
            .collect();
        let mean = nn.iter().sum::<f64>() / nn.len() as f64;
        assert!((1500.0..3000.0).contains(&mean), "mean nearest site {mean}");
        assert!(
            nn.iter().all(|&d| d > 700.0),
            "min {}",
            nn.iter().cloned().fold(f64::MAX, f64::min)
        );
    }

    #[test]
    fn archipelagos_exist_with_close_members() {
        let s = sites(WORLD_SEED);
        let arch: Vec<_> = s.iter().filter(|v| v.len() > 1).collect();
        let frac = arch.len() as f64 / s.len() as f64;
        assert!((0.08..=0.25).contains(&frac), "archipelago fraction {frac}");
        for a in arch {
            assert!((3..=6).contains(&a.len()), "{} members", a.len());
            for (i, m) in a.iter().enumerate() {
                let nearest = a
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, o)| (m.center - o.center).length())
                    .fold(f64::MAX, f64::min);
                assert!((100.0..=400.0).contains(&nearest), "member spacing {nearest}");
            }
        }
    }

    #[test]
    fn island_sizes_in_range() {
        for d in sites(WORLD_SEED).iter().flatten() {
            assert!(
                (15.0..=75.0).contains(&d.radius) && d.height <= 8.0 && d.height > 1.0,
                "{d:?}"
            );
        }
    }

    #[test]
    fn height_profile_is_a_beach() {
        let d = start_island(WORLD_SEED);
        assert!(island_height(&d, d.center) > 1.0);
        let dir = DVec2::new(0.6, 0.8);
        let at = |r: f64| island_height(&d, d.center + dir * r);
        assert!(at(d.radius as f64).abs() < 0.5, "shore height {}", at(d.radius as f64));
        let mut prev = at(d.radius as f64);
        let mut r = d.radius as f64;
        while r < 2.5 * d.radius as f64 {
            r += 0.5;
            assert!(at(r) <= prev + 1e-4, "not descending at {r}");
            prev = at(r);
        }
        assert!(at(2.5 * d.radius as f64) <= -SEA_DEPTH_M + 1.0);
        assert_eq!(at(3.1 * d.radius as f64), f32::NEG_INFINITY);
    }

    #[test]
    fn sea_floor_is_20_to_30_m_deep_everywhere() {
        let step = 50.0;
        for j in 0..200 {
            for i in 0..200 {
                let p = DVec2::new(i as f64 * step - 5000.0, j as f64 * step - 5000.0);
                let h = sea_floor(WORLD_SEED, p);
                assert!((-30.0..=-20.0).contains(&h), "{h} at {p}");
                let slope = (sea_floor(WORLD_SEED, p + DVec2::X) - h).abs();
                assert!(slope < 0.15, "slope {slope} at {p}");
            }
        }
    }

    #[test]
    fn seabed_is_the_highest_of_floor_and_islands() {
        let d = start_island(WORLD_SEED);
        assert_eq!(
            seabed_height(WORLD_SEED, std::slice::from_ref(&d), d.center),
            island_height(&d, d.center)
        );
        let far = d.center + DVec2::new(5000.0, 0.0);
        assert_eq!(seabed_height(WORLD_SEED, &[d], far), sea_floor(WORLD_SEED, far));
    }

    #[test]
    fn islands_near_finds_the_start_island() {
        let v = islands_near(WORLD_SEED, DVec2::ZERO, 1000.0);
        assert!(v.iter().any(|d| d.id == start_island(WORLD_SEED).id));
        assert!(v.iter().all(|d| d.center.length() <= 1000.0 + 3.0 * d.radius as f64));
    }
}
