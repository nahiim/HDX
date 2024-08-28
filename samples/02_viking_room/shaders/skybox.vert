#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;
layout(location = 2) in vec2 in_texcoord;

layout(location = 0) out vec3 tex_coord;

layout(set = 0, binding = 0) uniform VPMatrices
{
    mat4 model;
    mat4 projection;
    mat4 view;
}vp;

void main()
{
    tex_coord = in_position;
    vec4 pos = vp.projection * vp.view * vec4(in_position, 1.0);
    gl_Position = pos.xyww;
}