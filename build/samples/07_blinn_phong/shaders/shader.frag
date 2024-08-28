#version 450

layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec3 frag_pos;
layout(location = 3) in vec3 frag_color;
layout(location = 4) in vec3 view_pos;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform Light
{
    vec4 position; // Light position (xyz) and intensity (w)
    vec4 color;    // Light color (xyz)
} light;

float ambient_strength = 0.1;

// Attenuation factors
float constant = 1.0;
float linear = 0.09;
float quadratic = 0.032;

void main()
{
    vec3 color = frag_color;
    vec3 ambient = ambient_strength * light.color.xyz;

    vec3 normal = normalize(frag_normal);
    vec3 light_direction = normalize(light.position.xyz - frag_pos);
    float distance = length(light.position.xyz - frag_pos);

    // Calculate attenuation
    float attenuation = 1.0 / (constant + linear * distance + quadratic * distance * distance);

    // Apply attenuation to the light color (radiance)
    vec3 radiance = light.color.xyz * attenuation;

    // Diffuse component
    float diff = max(dot(light_direction, normal), 0.0);
    vec3 diffuse = diff * radiance;

    // Specular component
    vec3 view_direction = normalize(view_pos - frag_pos);
    vec3 halfway_dir = normalize(light_direction + view_direction);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), 32.0);
    vec3 specular = vec3(0.5, 0.5, 0.5) * spec * radiance;

    // Final color with ambient, diffuse, and specular components
    out_color = vec4((ambient + diffuse + specular) * color, 1.0);
}
