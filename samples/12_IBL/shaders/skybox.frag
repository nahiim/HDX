#version 450

layout(set = 0, binding = 1) uniform samplerCube cubemap;

layout(location = 0) in vec3 frag_uv;

layout(location = 0) out vec4 frag_color;

void main()
{
    frag_color = texture(cubemap, frag_uv);
}