// #include frame_common.wgsl
// #include buoy_common.wgsl
// The buoy into the G-buffer (fs_gbuffer) and into the opaque pre-pass that water refraction
// reads (fs_albedo).

struct Model {
    transform: mat4x4<f32>,
    prev_transform: mat4x4<f32>,
};

@group(1) @binding(0) var<uniform> model: Model;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) local_y: f32,
    @location(3) prev_world: vec3<f32>,
};

@vertex
fn vs_buoy(v: VsIn) -> VsOut {
    var o: VsOut;
    let w = model.transform * vec4<f32>(v.pos, 1.0);
    o.world = w.xyz;
    o.clip = frame.view_proj * w;
    o.normal = normalize((model.transform * vec4<f32>(v.normal, 0.0)).xyz);
    o.local_y = v.pos.y;
    o.prev_world = (model.prev_transform * vec4<f32>(v.pos, 1.0)).xyz;
    return o;
}

struct GBufferOut {
    @location(0) normal_var: vec4<f32>,
    @location(1) material: vec4<f32>,
    @location(2) motion: vec2<f32>,
};

// Material id 1 = buoy; b carries the height above the waterline for the paint bands.
fn encode_local_y(y: f32) -> f32 {
    return clamp((y + 0.5) / 4.0, 0.0, 1.0);
}

@fragment
fn fs_gbuffer(in: VsOut, @builtin(front_facing) front: bool) -> GBufferOut {
    var n = normalize(in.normal);
    if (!front) {
        n = -n;
    }
    let prev = frame.prev_view_proj * vec4<f32>(in.prev_world, 1.0);
    let prev_px = (vec2<f32>(prev.x, -prev.y) / prev.w * 0.5 + 0.5) * frame.resolution.xy;
    var o: GBufferOut;
    o.normal_var = vec4<f32>(n, BUOY_PAINT_VARIANCE);
    o.material = vec4<f32>(0.0, 1.0, encode_local_y(in.local_y), 0.0);
    o.motion = prev_px - in.clip.xy;
    return o;
}

@fragment
fn fs_albedo(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(buoy_albedo(in.local_y), 1.0);
}
