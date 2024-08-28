#version 450

layout(binding = 1) uniform sampler2D sampler1;

layout(location = 0) in vec2 tex_coord;
layout(location = 0) out vec4 out_color;

void main()
{
    vec3 frag_color = texture(sampler1, tex_coord).rgb;
    out_color = vec4(frag_color, 1.0);
}
