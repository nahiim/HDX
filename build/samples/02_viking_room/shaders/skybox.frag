#version 450

layout(set = 0, binding = 1) uniform samplerCube skybox;

layout(location = 0) in vec3 tex_coord;
layout(location = 0) out vec4 frag_color;

void main()
{
    frag_color = texture(skybox, tex_coord);
}