struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let horizon = vec3<f32>(1.0, 0.45, 0.15);
    let top = vec3<f32>(0.25, 0.12, 0.35);
    return vec4<f32>(mix(horizon, top, pow(1.0 - in.uv.y, 0.7)), 1.0);
}
