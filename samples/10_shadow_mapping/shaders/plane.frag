#version 450

layout(location = 0) in vec3 frag_pos;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec3 frag_color;
layout(location = 3) in vec4 light_space;   // Light-space position

layout(location = 0) out vec4 out_color;

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

layout(binding = 2) uniform sampler2D shadow_map;

vec3 ambient  = vec3(0.1, 0.1, 0.1);
vec3 diffuse  = frag_color;
vec3 specular = vec3(0.2);
float shininess = 32.0;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadow_map, projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // calculate bias (based on depth map resolution and slope)
    vec3 normal = normalize(frag_normal);
    vec3 lightDir = normalize(light.position.xyz - frag_pos);
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    // check whether current frag pos is in shadow
    // float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadow_map, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadow_map, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth  ? 1.0 : 0.0;        
        }
    }
    shadow /= 9.0;
    
    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(projCoords.z > 1.0)
        shadow = 0.0;
        
    return shadow;
}

void main()
{
    vec3 norm = normalize(frag_normal);
    vec3 light_dir = normalize(light.position.xyz - frag_pos);
    vec3 view_dir = normalize(mvp.view_pos.xyz - frag_pos);
    vec3 reflect_dir = reflect(-light_dir, norm);

    // Ambient
    vec3 ambient_comp = ambient * light.color.rgb;

    // Diffuse
    float diff = max(dot(norm, light_dir), 0.1);
    vec3 diffuse_comp = diff * diffuse * light.color.rgb;

    // Specular
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    vec3 specular_comp = spec * specular * light.color.rgb;

    // Sample the shadow mask
    float shadow = ShadowCalculation(light_space);

    // Correct shadowing: apply shadow only to diffuse and specular
    vec3 lit_color = ambient_comp + (1-shadow) * (diffuse_comp + specular_comp);

    out_color = vec4(lit_color, 1.0);


}
