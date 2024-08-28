#version 450

layout(location = 0) in vec3 in_position;

layout(binding = 0) uniform Light
{
    vec4 position;
    vec4 color;
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 view_pos;
};

void main() {
    // Transform the vertex position to light space
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}
