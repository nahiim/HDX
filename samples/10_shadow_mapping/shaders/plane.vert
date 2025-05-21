#version 450

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_normal;
layout(location = 2) in vec4 in_color;

layout(location = 0) out vec3 frag_pos;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_color;
layout(location = 3) out vec4 light_space;

layout(binding = 0) uniform MVP
{
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 view_pos;
} mvp;

layout(binding = 1) uniform Light
{
    vec4 position;
    vec4 color;
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 view_pos;
}light;

void main()
{
    vec4 world_position = mvp.model * in_position;  // World space position

    mat3 normal_matrix = transpose(inverse(mat3(mvp.model)));
    frag_normal = normalize(normal_matrix * in_normal.xyz);

    frag_pos = world_position.xyz;  // Pass world position to fragment shader
//    frag_normal = normalize(mat3(mvp.model) * in_normal.xyz);
    frag_color = in_color.rgb;
    
    light_space = light.projection * light.view * world_position;

    gl_Position = mvp.projection * mvp.view * world_position;
}
