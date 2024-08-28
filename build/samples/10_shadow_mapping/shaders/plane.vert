#version 450

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_normal;
layout(location = 2) in vec4 in_uv;
layout(location = 3) in vec4 in_tangent;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_pos;
layout(location = 3) out vec3 frag_tangent;

layout(binding = 0) uniform MVP
{
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 view_pos;
} mvp;

void main()
{
    vec4 world_position = mvp.model * in_position;  // World space position
    gl_Position = mvp.projection * mvp.view * world_position;

    frag_pos = world_position.xyz;  // Pass world position to fragment shader
    frag_uv = vec2(in_uv.xy);
    frag_normal = normalize(mat3(mvp.model) * in_normal.xyz);
    frag_tangent = normalize(mat3(mvp.model) * in_tangent.xyz);
}
