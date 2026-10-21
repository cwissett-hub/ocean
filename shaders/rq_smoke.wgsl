enable wgpu_ray_query;

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>, 2>;

fn trace(dir: vec3<f32>) -> vec4<f32> {
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, RayDesc(RAY_FLAG_NONE, 0xFFu, 0.001, 100.0, vec3<f32>(0.0), dir));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    return vec4<f32>(f32(hit.kind), hit.t, 0.0, 0.0);
}

@compute @workgroup_size(1)
fn main() {
    out[0] = trace(vec3<f32>(0.0, 0.0, 1.0));
    out[1] = trace(vec3<f32>(0.0, 0.0, -1.0));
}
