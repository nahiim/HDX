#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec3 frag_pos;
layout(location = 3) in vec3 frag_tangent;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform sampler2DArray t_map;

layout(binding = 2) uniform Light
{
    vec4 position;
    vec4 color;
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 view_pos;
}light;
layout(binding = 3) uniform sampler2DShadow shadow_map;

void main()
{
    vec3 normal_uv = vec3(frag_uv, 1.0);  // normal map is at layer 1
    vec3 diffuse_uv = vec3(frag_uv, 0.0); // albedo map is at layer 0

    // Obtain normal from normal map in range [0,1]
    vec3 normal = texture(t_map, normal_uv).rgb;

    // Transform normal vector to range [-1,1]
    normal = normalize(normal * 2.0 - 1.0);  // This normal is in tangent space

    // Compute tangent space basis
    vec3 bitangent = cross(frag_normal, frag_tangent);
    mat3 TBN = mat3(frag_tangent, bitangent, frag_normal);  // Tangent space basis matrix

    // Transform normal to world space
    vec3 world_normal = normalize(TBN * normal);

    // Lighting calculations (simple diffuse lighting)
    vec3 lightDir = normalize(vec3(light.position) - frag_pos);
    float diff = max(dot(world_normal, lightDir), 0.0);

    // Sample albedo map
    vec3 albedo = texture(t_map, diffuse_uv).rgb;

    // Calculate shadow
    vec4 light_space_pos = light.projection * light.view * vec4(frag_pos, 1.0);
    vec3 shadow_coord = light_space_pos.xyz / light_space_pos.w;
    shadow_coord = shadow_coord * 0.5 + 0.5; // Transform to [0,1] range

    float shadow = texture(shadow_map, shadow_coord);
    shadow = shadow < shadow_coord.z ? 0.5 : 1.0; // Simple shadow test

    // Debugging output
    out_color = vec4(diff * albedo * shadow, 1.0);
    // Uncomment the following line to visualize normals
    // out_color = vec4((world_normal + 1.0) * 0.5, 1.0);  // For visualizing world normals
    // Uncomment the following line to visualize light direction
    // out_color = vec4((lightDir + 1.0) * 0.5, 1.0);  // For visualizing light direction

//out_color = vec4(shadow_coord, 1.0); // For debugging

}
