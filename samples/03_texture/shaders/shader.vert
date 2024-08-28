#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_texcoord;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec2 frag_texcoord;

layout(binding = 0) uniform MVP
{
    mat4 model; 
    mat4 view;
    mat4 projection;
} mvp;


void main()
{
    frag_texcoord = vec2(in_texcoord.x, 1.0 - in_texcoord.y);
    gl_Position = mvp.projection * mvp.view * mvp.model * vec4(in_position, 1.0);
//    frag_texcoord = in_texcoord;
}
