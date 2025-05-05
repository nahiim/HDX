
#version 460
#extension GL_ARB_gpu_shader_fp64 : enable

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

float pos_inf = 1.0 / 0.0;
struct Ray
{
    vec3 origin;
    vec3 direction;
};
struct HitRecord
{
    vec3 p;
    vec3 normal;
    float t;
    bool front_face;
    vec3 albedo;
    uint mat_id;
    float ior;
};
void set_face_normal(inout HitRecord hr, Ray ray, vec3 outward_normal)
{
    // Sets the hit record normal vector.
    // NOTE: the parameter `outward_normal` is assumed to have unit length.

    hr.front_face = dot(ray.direction, outward_normal) < 0.0;
    hr.normal = hr.front_face ? outward_normal : -outward_normal;
}
vec3 rayAt(Ray ray, float t)
{
    return ray.origin + t*ray.direction;
}
struct Sphere
{
    vec3 center;
    float radius;
    vec3 color;
    float roughness;
    float transparency;
    float ior;
    uint mat_id;
};
const int NUM_SPHERES = 4;
Sphere spheres[NUM_SPHERES] = Sphere[](
    Sphere(vec3(0.0, 1.0, 0.0), 1.0, vec3(1.0, 0., 0.), 0.0, 1.0, 1.0, 0),  // lambertian
    Sphere(vec3(3.0, 1.0, 0.0), 1.0, vec3(1., 1.0, 0.1), 0.0, 0.0, 1.4, 1), // metal
    Sphere(vec3(6.0, 1.0, 0.0), 1.0, vec3(1., 1.0, 0.1), 0.0, 0.0, 1.4, 2), // Glass
    Sphere(vec3(0.0, -1000.0, 0.0), 1000.0, vec3(0.1, 1.0, 0.1), 0.0, 0.0, 1.0, 1)  // Floor
);

layout(std430, binding = 2) buffer PixelSSB
{
    uint pixels[];
};
layout(std430, binding = 5) buffer AccumPixelSSB
{
    uint accum[];
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

float linear_to_gamma(float linear_component)
{
    if (linear_component > 0)
        return sqrt(linear_component);

    return 0;
}
bool near_zero(vec3 v)
{
    float s = 1e-8;
    return abs(v.x) < s && abs(v.y) < s && abs(v.z) < s;
}

// float rand(vec2 co) {
//     return fract(sin(dot(co ,vec2(12.9898,78.233))) * 43758.5453);
// }
float rand(vec2 co) {
    float a = 12.9898;
    float b = 78.233;
    float c = 43758.5453;
    float dt = dot(co, vec2(a,b));
    float sn = mod(dt, 3.14159);
    return fract(sin(sn) * c);
}

vec3 random(float min, float max) {
    float range = max - min;
    vec2 base = vec2(gl_GlobalInvocationID.xy);
    return vec3(
        min + rand(base + vec2(0.0, 0.0)) * range,
        min + rand(base + vec2(1.0, 1.0)) * range,
        min + rand(base + vec2(2.0, 2.0)) * range
    );
}
float random_float() {
    return fract(sin(dot(gl_GlobalInvocationID.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

vec3 random_unit_vector()
{
    while (true)
    {
        vec3 p = random(-1,1);
        float lensq = dot(p, p);
        if (1e-5 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
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
float reflectance(float cosine, float refraction_index)
{
    // Use Schlick's approximation for reflectance.
    float r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}


bool scatter(Ray ray_in, HitRecord hr, inout vec3 attenuation, inout Ray scattered)
{
    if (hr.mat_id == 0)
    {
        vec3 scatter_direction = hr.normal + random_unit_vector();

        // Catch degenerate scatter direction
        if (near_zero(scatter_direction))
            scatter_direction = hr.normal;

        scattered = Ray(hr.p, scatter_direction);
        attenuation = hr.albedo;
        return true;
    }
    else if (hr.mat_id == 1)
    {
        vec3 reflected = reflect(ray_in.direction, hr.normal);
        scattered = Ray(hr.p, reflected);
        attenuation = hr.albedo;
        return true;
    }
    else if (hr.mat_id == 2)
    {
        attenuation = vec3(1.0, 1.0, 1.0);
        float ri = hr.front_face ? (1.0/hr.ior) : hr.ior;

        vec3 unit_direction = normalize(ray_in.direction);
        float cos_theta = min(dot(-unit_direction, hr.normal), 1.0);
        float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_float())
            direction = reflect(unit_direction, hr.normal);
        else
            direction = refract(unit_direction, hr.normal, ri);

        scattered = Ray(hr.p, direction);
        return true;
    }
}

bool hitSphere(inout HitRecord hr, Sphere sphere, Ray ray, float ray_tmin, float ray_tmax)
{
    vec3 oc = sphere.center - ray.origin;
    float a = dot(ray.direction, ray.direction);
    float h = dot(ray.direction, oc);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = h*h - a*c;

    if (discriminant < 0)
        return false;

    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (h - sqrtd) / a;
    if (root <= ray_tmin || ray_tmax <= root)
    {
        root = (h + sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root)
            return false;
    }

    hr.t = root;
    hr.p = rayAt(ray, hr.t);
    hr.albedo = sphere.color;
    hr.mat_id = sphere.mat_id;
    hr.ior = sphere.ior;
    vec3 outward_normal = (hr.p - sphere.center) / sphere.radius;
    set_face_normal(hr, ray, outward_normal);

    return true;
}


vec3 traceRay(Ray ray) {
    vec3 accumulated_color = vec3(1.0);
    const int MAX_BOUNCES = 10;

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++)
    {
        bool hit_anything = false;
        HitRecord hr;
        float closest_t = pos_inf;
        // int hit_index = -1;

        // Find closest hit
        for (uint i = 0; i < NUM_SPHERES; i++)
        {
            HitRecord temp_rec;
            if (hitSphere(temp_rec, spheres[i], ray, 0.00, closest_t))
            {
                closest_t = temp_rec.t;
                hr = temp_rec;
                hit_anything = true;
                // hit_index = int(i);
            }
        }

        if (hit_anything)
        {
            Ray scattered;
            vec3 attenuation;
            if (scatter(ray, hr, attenuation, scattered))
            {
                ray = scattered;
                accumulated_color *= attenuation; // attenuation}
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

    return vec3(0.0); // no light return if max bounces hit
}





// packing function (RGBA order)
uint packUint8To32(uint R, uint G, uint B, uint A)
{
    return (A << 24) | (B << 16) | (G << 8) | R;
}


// Hash by Dave Hoskins, simplified
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

void main()
{
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    if (x >= Nx || y >= Ny) return; // Bounds check

    uint idx = Nx * y + x;

    int num_samples = 100;  // Number of samples per pixel
    vec3 color = vec3(0.0);

    for (int i = 0; i < num_samples; i++)
    {
        float rx = hash(vec2(x + i, y));
        float ry = hash(vec2(x, y + i));

        float u = ((float(x) + 0.5 + rx) / float(Nx)) * 2.0 - 1.0;
        float v = ((float(y) + 0.5 + ry) / float(Ny)) * 2.0 - 1.0;

        Ray ray;
        ray.origin = cam_position;
        ray.direction = normalize(cam_direction + u * cam_right + v * cam_up);

        color += traceRay(ray);
    }

    // Average the color
    color /= float(num_samples);

    color = vec3(linear_to_gamma(color.r), linear_to_gamma(color.g), linear_to_gamma(color.b));

    // Clamp and pack the final color
    ivec3 i_color = ivec3(clamp(color * 255.0, 0.0, 255.0));
    pixels[idx] = packUint8To32(i_color.r, i_color.g, i_color.b, 255);
}
