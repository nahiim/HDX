#version 450

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_normal;
layout(location = 4) in mat4 in_transform;


layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_pos;
layout(location = 4) out vec3 view_pos;

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
    view_pos = vp.view_pos.xyz;
}
