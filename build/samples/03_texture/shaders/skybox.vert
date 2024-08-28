#version 450

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_uv;

layout(location = 0) out vec3 frag_uv;

layout(set = 0, binding = 0) uniform MVP
{
    mat4 model;
    mat4 view;
    mat4 projection;
}mvp;

void main()
{
    frag_uv = in_position.xyz;
    vec4 pos = mvp.projection * mvp.view * in_position;
    gl_Position = pos.xyww;
}