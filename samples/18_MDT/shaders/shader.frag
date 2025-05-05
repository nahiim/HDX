#version 450

layout(location = 0) in vec2 frag_texcoord;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform sampler2D sampler0;

void main()
{
    vec3 frag_color = texture(sampler0, frag_texcoord).rgb;
    out_color = vec4(frag_color, 1.0);
}
