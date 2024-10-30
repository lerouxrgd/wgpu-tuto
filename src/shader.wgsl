// Vertex shader

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
};

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) tangent_light_position: vec3<f32>,
    @location(2) tangent_view_position: vec3<f32>,
};

@group(1) @binding(0)
var<uniform> camera: Camera;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}
@group(2) @binding(0)
var<uniform> light: Light;

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    // Construct the tangent matrix
    let world_normal = normalize(normal_matrix * model.normal);
    let world_tangent = normalize(normal_matrix * model.tangent);
    let world_bitangent = normalize(normal_matrix * model.bitangent);
    let tangent_matrix = transpose(mat3x3<f32>(
        world_tangent,
        world_bitangent,
        world_normal,
    ));

    let world_position = model_matrix * vec4<f32>(model.position, 1.0);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_position;
    out.tex_coords = model.tex_coords;
    out.tangent_view_position = tangent_matrix * (camera.view_pos.xyz - world_position.xyz);
    out.tangent_light_position = tangent_matrix * (light.position - world_position.xyz);
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0)@binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample textures
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    var object_normal: vec3<f32> = textureSample(t_normal, s_normal, in.tex_coords).xyz;
    object_normal = pow(object_normal, vec3<f32>(1.0 / 2.2)); // cheap sRGB to linear, reversed ;)

    // We don't need (or want) much ambient light, so 0.1 is fine
    let ambient_strength = 0.012;
    let ambient_color = object_color.rgb * ambient_strength;

    // Create the lighting vectors
    var tangent_normal = normalize(object_normal * 2.0 - 1.0);
    var light_dir = in.tangent_light_position;
    let light_dir_length = length(light_dir);
    light_dir /= light_dir_length;
    let view_dir = normalize(in.tangent_view_position);
    let half_dir = normalize(view_dir + light_dir);

    let diffuse_strength = max(dot(tangent_normal, light_dir), 0.0);
    let diffuse_color = object_color.rgb * diffuse_strength;

    let specular_strength = 0.1 * pow(max(dot(tangent_normal, half_dir), 0.0), 16.0);

    let attenuation_factor = 10.0 / (light_dir_length * light_dir_length);
    let incoming_light = attenuation_factor * light.color;

    var result = incoming_light * (diffuse_color + specular_strength) + ambient_color;
    result = pow(result, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(result, object_color.a);
}
