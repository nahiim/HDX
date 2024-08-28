#version 450

layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec3 frag_pos;
layout(location = 4) in vec3 view_pos;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform Light
{
    vec4 position; // Light position (xyz) and intensity (w)
    vec4 color;    // Light color (xyz)
} light;

layout(binding = 2) uniform samplerCube skybox;



void main()
{
    vec3 I = normalize(frag_pos - view_pos);
    vec3 R = reflect(I, normalize(frag_normal));
    out_color = texture(skybox, R) * vec4(1.0, 0.7, 0.3, 1.0);
}
