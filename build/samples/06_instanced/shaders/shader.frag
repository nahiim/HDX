#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec3 frag_pos;
layout(location = 3) in vec3 TangentLightPos;
layout(location = 4) in vec3 TangentViewPos;
layout(location = 5) in vec3 TangentFragPos;
layout(location = 6) in float frag_texture_id;

layout(location = 0) out vec4 out_color;

layout(binding = 2) uniform sampler2DArray texture_array;
//layout(binding = 3) uniform sampler2D normal_map;

layout(binding = 1) uniform Light
{
    vec4 position;
    vec4 color;
}light;

float ambience = 0.0;


void main()
{
    vec3 diffuse_uv = vec3(frag_uv, frag_texture_id*2);
    vec3 normal_uv = vec3(frag_uv, frag_texture_id*2 + 1);
     // obtain normal from normal map in range [0,1]
    vec3 normal = texture(texture_array, normal_uv).rgb;

    // transform normal vector to range [-1,1]
    normal = normalize(normal * 2.0 - 1.0);  // this normal is in tangent space

    // get diffuse color
    vec3 color = texture(texture_array, diffuse_uv).rgb;

    // ambient
    vec3 ambient = ambience * color;

    vec3 light_direction = normalize(TangentLightPos - TangentFragPos);
    float diff = max(dot(light_direction, normal), 0.0);
    vec3 diffuse = diff * color;

    vec3 view_direction = normalize(TangentViewPos - TangentFragPos);
    vec3 reflection_direction = reflect(-light_direction, normal);
    vec3 halfway_dir = normalize(view_direction + view_direction);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), 32.0);
    vec3 specular = vec3(0.5, 0.5, 0.5) * spec * light.color.xyz;

    out_color = vec4(ambient + diffuse + specular, 1.0);

//    out_color = texture(texture_array, diffuse_uv);
}
