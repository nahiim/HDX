
#version 460

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

struct Ray
{
    vec3 origin;
    vec3 direction;
};
struct Sphere
{
    vec3 center;
    float radius;
    vec3 color;
    float roughness;
};
const int NUM_SPHERES = 2;
Sphere spheres[NUM_SPHERES] = Sphere[](
    Sphere(vec3(0.0, 1.0, 0.0), 1.0, vec3(1.0, 0.0, 1.0), 0.0), // Metal
    Sphere(vec3(0.0, -50.0, 0.0), 50, vec3(0.0, 1.0, 0.0), 0.5) // Floor
);

layout(std430, binding = 2) buffer AccumulatedColorSSB {
    vec4 accumulated_colors[]; // store float RGB + alpha (for blending or sample count)
};


layout(binding = 3) uniform UBO
{
    uint frame_index;
    uint Nx;    // Image width
    uint Ny;    // Image height

    vec3 cam_position;
    vec3 cam_direction;
    vec3 cam_up;
    vec3 cam_right;
};

layout(std430, binding = 4) buffer RaySSB
{
    Ray rays[];
};

float rand(vec2 co, int seed)
{
    return fract(sin(dot(co.xy + float(seed), vec2(12.9898,78.233))) * 43758.5453123);
}
vec3 randomHemisphereDirection(vec3 normal) 
{
    // Generate random angles for hemisphere sampling
    float r1 = fract(sin(gl_GlobalInvocationID.x * 12.9898 + gl_GlobalInvocationID.y * 78.233) * 43758.5453);
    float r2 = fract(sin(gl_GlobalInvocationID.y * 43.1212 + gl_GlobalInvocationID.x * 93.2121) * 43758.5453);
    
    float phi = 2.0 * 3.14159265359 * r1;
    float cosTheta = sqrt(1.0 - r2);
    float sinTheta = sqrt(r2);

    vec3 tangent = normalize(cross(normal, vec3(0.0, 1.0, 0.0)));
    vec3 bitangent = cross(normal, tangent);
    
    return normalize(sinTheta * cos(phi) * tangent + sinTheta * sin(phi) * bitangent + cosTheta * normal);
}


bool hitSphere(Ray ray, Sphere sphere, out float t, out vec3 normal)
{
    vec3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0)
        return false;

    float t1 = (-b - sqrt(discriminant)) / (2.0 * a);
    float t2 = (-b + sqrt(discriminant)) / (2.0 * a);

    t = (t1 > 0.001) ? t1 : ((t2 > 0.001) ? t2 : -1.0);

    if (t < 0.0)
        return false;

    vec3 hit_point = ray.origin + t * ray.direction;
    normal = normalize(hit_point - sphere.center);

    return true;
}


bool isInShadow(vec3 point, vec3 lightDir)
{
    Ray shadowRay;
    shadowRay.origin = point + lightDir * 1e-4;  // Avoid self-shadowing
    shadowRay.direction = lightDir;

    for (int j = 0; j < NUM_SPHERES; j++)
    {
        float t;
        vec3 n;
        if (hitSphere(shadowRay, spheres[j], t, n))
        {
            return true; // Something blocks the light
        }
    }
    return false; // Light is visible
}

vec3 traceRay(Ray ray)
{
    vec3 final_color = vec3(0.0);
    vec3 throughput = vec3(1.0);
    int max_bounces = 2;
    vec3 light_dir = normalize(vec3(-1.0, -1.0, -1.0));

    for (int i = 0; i < max_bounces; i++)
    {
        float closestT = 1e9;
        vec3 hit_normal, hit_color;
        bool hit = false;
        float hit_roughness;

        for (int j = 0; j < NUM_SPHERES; j++)
        {
            float t;
            vec3 normal;
            if (hitSphere(ray, spheres[j], t, normal) && t < closestT)
            {
                hit = true;
                closestT = t;
                hit_normal = normal;
                hit_color = spheres[j].color;
                hit_roughness = spheres[j].roughness;
            }
        }

        if (hit)
        {
            vec3 hit_point = ray.origin + closestT * ray.direction;

            // ✅ Direct lighting — modulate by hit color
            float diffuse = max(dot(hit_normal, -light_dir), 0.0);
            final_color += throughput * hit_color * diffuse;

            // ✅ For pure reflection, don't modulate by hit color:
            // This lets the reflection carry the real color of what's hit
            throughput *= mix(vec3(1.0), hit_color, hit_roughness); // for tinted reflection

            ray.origin = hit_point + hit_normal * 1e-4;
            ray.direction = reflect(normalize(ray.direction), normalize(hit_normal + hit_roughness * randomHemisphereDirection(hit_normal)));
            if (!isInShadow(hit_point, -light_dir))
            {
                float diffuse = max(dot(hit_normal, -light_dir), 0.0);
                final_color += throughput * hit_color * diffuse;
            }

        }
        else
        {
            final_color += throughput * vec3(0.5, 0.7, 1.0) * (0.5 + 0.5 * ray.direction.y);
            break;
        }
    }

    return final_color;
}







// packing function (RGBA order)
uint packUint8To32(uint R, uint G, uint B, uint A)
{
    return (A << 24) | (B << 16) | (G << 8) | R;
}



void main()
{
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= Nx || y >= Ny) return;

    uint idx = Nx * y + x;

    // --- trace 1 ray per frame per pixel ---
    float jitterX = (fract(sin(float(x + y * 17 + frame_index * 23)) * 43758.5453) - 0.5) / float(Nx);
    float jitterY = (fract(sin(float(y + x * 13 + frame_index * 31)) * 43758.5453) - 0.5) / float(Ny);
    float u = ((float(x) + 0.5 + jitterX) / float(Nx)) * 2.0 - 1.0;
    float v = ((float(y) + 0.5 + jitterY) / float(Ny)) * 2.0 - 1.0;

    Ray ray;
    ray.origin = cam_position;
    ray.direction = normalize(cam_direction + u * cam_right + v * cam_up);

    vec3 new_sample = traceRay(ray); // <--- one sample this frame

    // --- Accumulate with previous value ---
    vec3 prev = accumulated_colors[idx].rgb;
    float count = float(frame_index);

    vec3 blended = mix(prev, new_sample, 1.0 / (count + 1.0)); // running average

    accumulated_colors[idx] = vec4(blended, 1.0); // alpha unused
}
