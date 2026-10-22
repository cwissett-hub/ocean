// #include frame_common.wgsl
// #include seabed_common.wgsl
// Spray and mist simulation. Entry points: clear_deposits, emit (crest spray + mist), spawn
// (CPU SpawnRequests), update (physics; landing spray deposits foam for the next frame).

struct Particle {
    pos: vec3<f32>,
    life: f32,
    vel: vec3<f32>,
    kind: u32,
    size: f32,
    max_life: f32,
    seed: f32,
    _pad: f32,
};

struct SpawnRequest {
    position: vec3<f32>,
    kind: u32,
    velocity: vec3<f32>,
    count: u32,
};

struct SimParams {
    dt: f32,
    spray_rate: f32,
    mist_rate: f32,
    threshold: f32,
    wind: vec2<f32>,
    frame_index: u32,
    request_count: u32,
    area_origin: vec2<f32>,
    area_size: f32,
    candidates: u32,
};

@group(0) @binding(1) var displacement: texture_2d_array<f32>;
@group(0) @binding(2) var derivatives: texture_2d_array<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(5) var<storage, read_write> head: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> deposits: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> sp: SimParams;
@group(0) @binding(8) var<storage, read> requests: array<SpawnRequest>;
@group(0) @binding(9) var seabed: texture_2d<f32>;

// Waves (and so crest spray) fade out over the shallows like the rendered surface.
fn shallows(p: vec2<f32>) -> f32 {
    return wave_attenuation(-seabed_height(seabed, frame.seabed, p));
}

const MAX_PARTICLES: u32 = 65536u;
const DEPOSIT_RES: u32 = 512u;
const DEPOSIT_CELL_M: f32 = 2.0;
const FOAM_CASCADES: u32 = 3u;
const KIND_SPRAY: u32 = 0u;
const KIND_MIST: u32 = 1u;
const G: f32 = 9.81;
const SPRAY_BURST: u32 = 6u;

fn hash(n: u32) -> f32 {
    var x = n * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return f32(x) / 4294967295.0;
}

fn disp(p: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(0.0);
    for (var c = 0u; c < FOAM_CASCADES; c++) {
        d += textureSampleLevel(displacement, samp, cascade_uv(p, c), c, 0.0).xyz;
    }
    return d * shallows(p);
}

fn jacobian(p: vec2<f32>) -> f32 {
    var d = vec3<f32>(0.0);
    for (var c = 0u; c < FOAM_CASCADES; c++) {
        let der = textureSampleLevel(derivatives, samp, cascade_uv(p, c), c, 0.0);
        let jxz = textureSampleLevel(displacement, samp, cascade_uv(p, c), c, 0.0).w;
        d += vec3<f32>(der.z, der.w, jxz);
    }
    d *= shallows(p);
    return (1.0 + d.x) * (1.0 + d.y) - d.z * d.z;
}

fn allocate() -> u32 {
    return atomicAdd(&head[0], 1u) % MAX_PARTICLES;
}

fn spawn(kind: u32, pos: vec3<f32>, vel: vec3<f32>, seed: u32) {
    var q: Particle;
    q.pos = pos;
    q.vel = vel;
    q.kind = kind;
    q.seed = hash(seed ^ 0x9e3779b9u);
    if (kind == KIND_SPRAY) {
        q.max_life = 1.2 + 1.6 * hash(seed + 7u);
        q.size = 0.25 + 0.5 * hash(seed + 11u);
    } else {
        q.max_life = 5.0 + 5.0 * hash(seed + 7u);
        q.size = 4.0 + 6.0 * hash(seed + 11u);
    }
    q.life = q.max_life;
    particles[allocate()] = q;
}

@compute @workgroup_size(64, 1, 1)
fn clear_deposits(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.y * DEPOSIT_RES * 64u + id.x;
    if (i < DEPOSIT_RES * DEPOSIT_RES) {
        atomicStore(&deposits[i], 0u);
    }
}

@compute @workgroup_size(64, 1, 1)
fn emit(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= sp.candidates) {
        return;
    }
    let seed = id.x * 9781u + sp.frame_index * 6271u;
    let p = sp.area_origin + vec2<f32>(hash(seed), hash(seed + 1u)) * sp.area_size;
    let j = jacobian(p);
    if (j >= sp.threshold) {
        return;
    }
    let cell_area = sp.area_size * sp.area_size / f32(sp.candidates);
    let strength = clamp((sp.threshold - j) * 8.0, 0.0, 1.0);
    let surface = vec3<f32>(p.x, 0.0, p.y) + disp(p);
    let wind = vec3<f32>(sp.wind.x, 0.0, sp.wind.y);
    // Spray leaves a crest as a plume: a burst of clumps sharing a launch with some spread.
    if (hash(seed + 2u) < sp.spray_rate * strength * cell_area * sp.dt / f32(SPRAY_BURST)) {
        let up = 2.0 + 3.5 * hash(seed + 5u);
        for (var b = 0u; b < SPRAY_BURST; b++) {
            let s = seed + 101u * (b + 1u);
            let jitter = vec3<f32>(hash(s + 3u) - 0.5, (hash(s + 8u) - 0.5) * 0.6, hash(s + 4u) - 0.5) * 1.6;
            let offset = vec3<f32>(hash(s + 9u) - 0.5, 0.0, hash(s + 10u) - 0.5) * 1.2;
            spawn(KIND_SPRAY, surface + offset + vec3<f32>(0.0, 0.15, 0.0), wind * 0.35 + jitter + vec3<f32>(0.0, up, 0.0), s);
        }
    }
    if (hash(seed + 6u) < sp.mist_rate * strength * cell_area * sp.dt) {
        spawn(KIND_MIST, surface + vec3<f32>(0.0, 0.4, 0.0), wind * 0.5 + vec3<f32>(0.0, 0.25, 0.0), seed + 13u);
    }
}

@compute @workgroup_size(64, 1, 1)
fn spawn_requests(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= sp.request_count) {
        return;
    }
    let r = requests[id.x];
    for (var k = 0u; k < r.count; k++) {
        let s = id.x * 131u + k * 17u + sp.frame_index * 7919u;
        let jitter = (vec3<f32>(hash(s), hash(s + 1u), hash(s + 2u)) - 0.5) * 0.6;
        spawn(r.kind, r.position, r.velocity + jitter * length(r.velocity), s);
    }
}

@compute @workgroup_size(64, 1, 1)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= MAX_PARTICLES) {
        return;
    }
    var q = particles[i];
    if (q.life <= 0.0) {
        return;
    }
    let dt = sp.dt;
    q.life -= dt;
    let wind = vec3<f32>(sp.wind.x, 0.0, sp.wind.y);
    if (q.kind == KIND_SPRAY) {
        q.vel.y -= G * dt;
        // Droplet clumps are dragged towards the wind velocity.
        q.vel += (wind - q.vel) * (1.0 - exp(-dt * 0.9));
        q.pos += q.vel * dt;
        // Spray lands on the water, or on the beach where that is higher.
        let water = max(disp(q.pos.xz).y, seabed_height(seabed, frame.seabed, q.pos.xz));
        if (q.vel.y < 0.0 && q.pos.y < water) {
            // Landed: leave foam where it fell, for the foam pass next frame.
            let cell = vec2<i32>(floor(q.pos.xz / DEPOSIT_CELL_M));
            let c = vec2<u32>(vec2<i32>(i32(DEPOSIT_RES)) + cell % i32(DEPOSIT_RES)) % DEPOSIT_RES;
            atomicAdd(&deposits[c.y * DEPOSIT_RES + c.x], 64u);
            q.life = 0.0;
        }
    } else {
        // Mist drifts with the wind, rises slowly and spreads.
        q.vel += (wind * 0.6 + vec3<f32>(0.0, 0.15, 0.0) - q.vel) * (1.0 - exp(-dt * 0.5));
        q.pos += q.vel * dt;
        q.size += dt * 0.4;
    }
    particles[i] = q;
}
