#version 450

layout(location = 0) in vec4 in_position;

layout(location = 0) out vec3 frag_uv;

layout(set = 0, binding = 0) uniform VP
{
    mat4 view;
    mat4 projection;
    vec4 cam_pos;
}vp;

void main()
{
    frag_uv = in_position.xyz;
    mat4 view = mat4(mat3(vp.view));
    vec4 pos = vp.projection * view * in_position;
    gl_Position = pos.xyww;
}