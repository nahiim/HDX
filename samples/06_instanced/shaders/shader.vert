#version 450

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_normal;
layout(location = 2) in vec4 in_uv;
layout(location = 3) in vec4 in_tangent;
layout(location = 4) in mat4 in_transform;
layout(location = 8) in float in_texture_id;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_pos;
layout(location = 3) out vec3 TangentLightPos;
layout(location = 4) out vec3 TangentViewPos;
layout(location = 5) out vec3 TangentFragPos;
layout(location = 6) out float frag_texture_id;

layout(binding = 0) uniform VP
{
    mat4 view;
    mat4 projection;
    vec4 view_pos;
}vp;

layout(binding = 1) uniform Light
{
    vec4 position;
    vec4 color;
}light;


void main()
{
    gl_Position = vp.projection * vp.view * in_transform * in_position;

    frag_pos = vec3(in_transform * in_position);
    frag_normal = transpose(inverse(mat3(in_transform))) * in_normal.xyz;
    frag_uv = vec2(in_uv.xy);
    frag_normal = vec3(in_normal.xyz);

    mat3 normal_matrix = transpose(inverse(mat3(in_transform)));
    vec3 T = normalize(normal_matrix * in_tangent.xyz);
    vec3 N = normalize(normal_matrix * in_normal.xyz);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    
    mat3 TBN = transpose(mat3(T, B, N));    
    TangentLightPos = TBN * light.position.xyz;
    TangentViewPos  = TBN * vp.view_pos.xyz;
    TangentFragPos  = TBN * frag_pos;

    frag_texture_id = in_texture_id;
}
