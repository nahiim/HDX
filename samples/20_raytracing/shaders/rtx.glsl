
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
    float transparency;
    float ior;
    float emission;
};

const int NUM_SPHERES = 2;
Sphere spheres[NUM_SPHERES] = Sphere[](
    Sphere(vec3(0.0, 1.0, 0.0), 1.0, vec3(1.0, 1.0, 1.0), 0.0, 1.0, 1.0, .0), // Red Sphere
    Sphere(vec3(0.0, -50.0, 0.0), 50, vec3(0.0, 1.0, 0.0), 0.2, 0.0, 1.0, .0) // Green Sphere
);

layout(std430, binding = 2) buffer PixelSSB
{
    uint pixels[];
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
float random_float() {
    return fract(sin(dot(gl_GlobalInvocationID.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}
float reflectance(float cosine, float refraction_index)
{
    // Use Schlick's approximation for reflectance.
    float r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

bool scatter(Ray incoming, out vec3 attenuation, out Ray scattered, float refraction_index, vec3 hit_point, vec3 hit_normal)
{
    attenuation = vec3(1.0);
    bool front_face = dot(incoming.direction, hit_normal) < 0;
    float ri = front_face ? (1.0/refraction_index) : refraction_index;

    vec3 unit_direction = normalize(incoming.direction);
    float cos_theta = min(dot(-unit_direction, hit_normal), 1.0);
    float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0;
    vec3 direction;

    vec3 refracted = refract(unit_direction, hit_normal, ri);
//    scattered = Ray(hit_point, refracted);

    if (cannot_refract)
        direction = reflect(unit_direction, hit_normal);
    else
        direction = refract(unit_direction, hit_normal, ri);

    scattered = Ray(hit_point, direction);

    return true;
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




vec3 traceRay(Ray ray)
{
    vec3 final_color = vec3(0.34); // Default background color
    int max_bounces = 2;          // Max reflection depth
    vec3 light_dir = normalize(vec3(-1.0, -1.0, -1.0));
    vec3 attenuation;
    vec3 accumulated_color = vec3(1.0);

    float eta_air = 1.0; // IOR of air

    for (int i = 0; i < max_bounces; i++)
    {
        float closestT = 1e9;
        vec3 hit_normal, hit_color;
        bool hit = false;
        float hit_roughness, hit_transparency, hit_ior;
        
        // Find the closest sphere hit
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
                hit_transparency = spheres[j].transparency;
                hit_ior = spheres[j].ior;
            }
        }

        if (hit)
        {
            vec3 hit_point = ray.origin + closestT * ray.direction;


            
            if (hit_transparency == 1.0)
                scatter(ray, attenuation, ray, hit_ior, hit_point, hit_normal);
            else{
                // **Lighting Calculation**
                float diffuse = max(dot(hit_normal, -light_dir), 0.2); // Prevent pure black shadows
                final_color = hit_color * diffuse;

                // Update ray direction with rough reflection
                ray.origin = hit_point + hit_normal * 4.77e-4;
                ray.direction = reflect(normalize(ray.direction), normalize(hit_normal + hit_roughness * randomHemisphereDirection(hit_normal)));
            }
        }
        else
        {
            // Background
            vec3 unit_direction = normalize(ray.direction);
            float a = 0.5 * (unit_direction.y + 1.0);
            vec3 background = mix(vec3(1.0), vec3(0.5, 0.7, 1.0), a);
            return accumulated_color * background;
        }
    }

    return vec3(0.0);
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

    if (x >= Nx || y >= Ny) return; // Bounds check

    uint idx = Nx * y + x;

    int num_samples = 64;  // Number of samples per pixel
    vec3 color = vec3(0.0);

    for (int i = 0; i < num_samples; i++)
    {
        // Random jittering in the range [-0.5, 0.5]
        float jitterX = (fract(sin(float(x + y * 17 + i * 23)) * 43758.5453) - 0.5) / float(Nx);
        float jitterY = (fract(sin(float(y + x * 13 + i * 31)) * 43758.5453) - 0.5) / float(Ny);

        float u = ((float(x) + 0.5 + jitterX) / float(Nx)) * 2.0 - 1.0;
        float v = ((float(y) + 0.5 + jitterY) / float(Ny)) * 2.0 - 1.0;

        Ray ray;
        ray.origin = cam_position;
        ray.direction = normalize(cam_direction + u * cam_right + v * cam_up);

        color += traceRay(ray);
    }

    // Average the color
    color /= float(num_samples);

    // Clamp and pack the final color
    ivec3 i_color = ivec3(clamp(color * 255.0, 0.0, 255.0));
    pixels[idx] = packUint8To32(i_color.r, i_color.g, i_color.b, 255);
}
