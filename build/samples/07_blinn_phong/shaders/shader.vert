#version 450

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_normal;
layout(location = 4) in mat4 in_transform;
layout(location = 8) in vec4 in_color;
layout(location = 9) in vec4 in_ao;
layout(location = 10) in vec4 in_roughness;
layout(location = 11) in vec4 in_metallic;

layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_pos;
layout(location = 3) out vec3 frag_color;
layout(location = 4) out vec3 view_pos;
layout(location = 5) out float ao;
layout(location = 6) out float roughness;
layout(location = 7) out float metallic;

layout(binding = 0) uniform VP
{
    mat4 view;
    mat4 projection;
    vec4 view_pos;
}vp;

layout(binding = 1) uniform Light
{
    vec4 position;
    vec4 color;
}light;


void main()
{
    gl_Position = vp.projection * vp.view * in_transform * in_position;

    frag_pos = vec3(in_transform * in_position);
    frag_normal = mat3(transpose(inverse(in_transform))) * in_normal.xyz;
    frag_color = in_color.xyz;
    view_pos = vp.view_pos.xyz;
}
