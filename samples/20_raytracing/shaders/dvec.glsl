
#version 460
#extension GL_ARB_gpu_shader_fp64 : enable

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

double pos_inf = 1.0 / 0.0;
struct Ray
{
    dvec3 origin;
    dvec3 direction;
};
struct HitRecord
{
    dvec3 p;
    dvec3 normal;
    double t;
    bool front_face;
    dvec3 albedo;
    uint mat_id;
    double ior;
};
void set_face_normal(inout HitRecord hr, Ray ray, dvec3 outward_normal)
{
    // Sets the hit record normal vector.
    // NOTE: the parameter `outward_normal` is assumed to have unit length.

    hr.front_face = dot(ray.direction, outward_normal) < 0.0;
    hr.normal = hr.front_face ? outward_normal : -outward_normal;
}
dvec3 rayAt(Ray ray, double t)
{
    return ray.origin + t*ray.direction;
}
struct Sphere
{
    dvec3 center;
    double radius;
    dvec3 color;
    double roughness;
    double transparency;
    double ior;
    uint mat_id;
};
const int NUM_SPHERES = 4;
Sphere spheres[NUM_SPHERES] = Sphere[](
    Sphere(dvec3(0.0, 1.0, 0.0), 1.0, dvec3(1.0, 0.3, 0.2), 0.0, 1.0, 1.0, 1),  // Metal
    Sphere(dvec3(4.0, 1.0, 0.0), 1.0, dvec3(1., 1.0, 0.1), 0.0, 0.0, 1.4, 2), // Glass
    Sphere(dvec3(4.0, 1.0, 0.0), 1.0, dvec3(1., 1.0, 0.1), 0.0, 0.0, 1., 2), // Glass
    Sphere(dvec3(0.0, -1000.0, 0.0), 1000.0, dvec3(0.1, 1.0, 0.1), 0.0, 0.0, 1.0, 0)  // Floor
);

layout(std430, binding = 2) buffer PixelSSB
{
    uint pixels[];
};

layout(binding = 3) uniform UBO
{
    uint Nx;    // Image width
    uint Ny;    // Image height

    dvec3 cam_position;
    dvec3 cam_direction;
    dvec3 cam_up;
    dvec3 cam_right;
};

layout(std430, binding = 4) buffer RaySSB
{
    Ray rays[];
};

double linear_to_gamma(double linear_component)
{
    if (linear_component > 0)
        return sqrt(linear_component);

    return 0;
}
bool near_zero(dvec3 v)
{
    double s = 1e-8;
    return abs(v.x) < s && abs(v.y) < s && abs(v.z) < s;
}

// double rand(vec2 co) {
//     return fract(sin(dot(co ,vec2(12.9898,78.233))) * 43758.5453);
// }
double rand(vec2 co) {
    double a = 12.9898;
    double b = 78.233;
    double c = 43758.5453;
    double dt = dot(co, vec2(a,b));
    double sn = mod(dt, 3.14159);
    return fract(sin(float(sn)) * c);
}

dvec3 random(double min, double max) {
    double range = max - min;
    vec2 base = vec2(gl_GlobalInvocationID.xy);
    return dvec3(
        min + rand(base + vec2(0.0, 0.0)) * range,
        min + rand(base + vec2(1.0, 1.0)) * range,
        min + rand(base + vec2(2.0, 2.0)) * range
    );
}
double random_double() {
    return fract(sin(dot(gl_GlobalInvocationID.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

dvec3 random_unit_vector()
{
    while (true)
    {
        dvec3 p = random(-1,1);
        double lensq = dot(p, p);
        if (1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

dvec3 randomHemisphereDirection(dvec3 normal) 
{
    // Generate random angles for hemisphere sampling
    double r1 = fract(sin(gl_GlobalInvocationID.x * 12.9898 + gl_GlobalInvocationID.y * 78.233) * 43758.5453);
    double r2 = fract(sin(gl_GlobalInvocationID.y * 43.1212 + gl_GlobalInvocationID.x * 93.2121) * 43758.5453);
    
    double phi = 2.0 * 3.14159265359 * r1;
    double cosTheta = sqrt(1.0 - r2);
    double sinTheta = sqrt(r2);

    dvec3 tangent = normalize(cross(normal, dvec3(0.0, 1.0, 0.0)));
    dvec3 bitangent = cross(normal, tangent);
    
    return normalize(sinTheta * cos(float(phi)) * tangent + sinTheta * sin(float(phi)) * bitangent + cosTheta * normal);
}
double reflectance(double cosine, double refraction_index)
{
    // Use Schlick's approximation for reflectance.
    double r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0*r0;
    return r0 + (1-r0)*pow(float(1 - cosine),5);
}


bool scatter(Ray ray_in, HitRecord hr, inout dvec3 attenuation, inout Ray scattered)
{
    if (hr.mat_id == 0)
    {
        dvec3 scatter_direction = hr.normal + random_unit_vector();

        // Catch degenerate scatter direction
        if (near_zero(scatter_direction))
            scatter_direction = hr.normal;

        scattered = Ray(hr.p, scatter_direction);
        attenuation = hr.albedo;
        return true;
    }
    else if (hr.mat_id == 1)
    {
        dvec3 reflected = reflect(ray_in.direction, hr.normal);
        scattered = Ray(hr.p, reflected);
        attenuation = hr.albedo;
        return true;
    }
    else if (hr.mat_id == 2)
    {
        attenuation = dvec3(1.0, 1.0, 1.0);
        double ri = hr.front_face ? (1.0/hr.ior) : hr.ior;

        dvec3 unit_direction = normalize(ray_in.direction);
        double cos_theta = min(dot(-unit_direction, hr.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        dvec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_double())
            direction = reflect(unit_direction, hr.normal);
        else
            direction = refract(unit_direction, hr.normal, ri);

        scattered = Ray(hr.p, direction);
        return true;
    }
}

bool hitSphere(inout HitRecord hr, Sphere sphere, Ray ray, double ray_tmin, double ray_tmax)
{
    dvec3 oc = sphere.center - ray.origin;
    double a = dot(ray.direction, ray.direction);
    double h = dot(ray.direction, oc);
    double c = dot(oc, oc) - sphere.radius * sphere.radius;
    double discriminant = h*h - a*c;

    if (discriminant < 0)
        return false;

    double sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    double root = (h - sqrtd) / a;
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
    dvec3 outward_normal = (hr.p - sphere.center) / sphere.radius;
    set_face_normal(hr, ray, outward_normal);

    return true;
}


dvec3 traceRay(Ray ray) {
    dvec3 accumulated_color = dvec3(1.0);
    const int MAX_BOUNCES = 10;

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++)
    {
        bool hit_anything = false;
        HitRecord hr;
        double closest_t = pos_inf;
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
            dvec3 attenuation;
            if (scatter(ray, hr, attenuation, scattered))
            {
                ray = scattered;
                accumulated_color *= attenuation; // attenuation}
            }
        }
        else
        {
            // Background
            dvec3 unit_direction = normalize(ray.direction);
            double a = 0.5 * (unit_direction.y + 1.0);
            dvec3 background = mix(dvec3(1.0), dvec3(0.5, 0.7, 1.0), a);
            return accumulated_color * background;
        }
    }

    return dvec3(0.0); // no light return if max bounces hit
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
    dvec3 color = dvec3(0.0);

    for (int i = 0; i < num_samples; i++)
    {
        // Random jittering in the range [-0.5, 0.5]
        double jitterX = (fract(sin(float(x + y * 17 + i * 23)) * 43758.5453) - 0.5) / double(Nx);
        double jitterY = (fract(sin(float(y + x * 13 + i * 31)) * 43758.5453) - 0.5) / double(Ny);

        double u = ((double(x) + 0.5 + jitterX) / double(Nx)) * 2.0 - 1.0;
        double v = ((double(y) + 0.5 + jitterY) / double(Ny)) * 2.0 - 1.0;

        Ray ray;
        ray.origin = cam_position;
        ray.direction = normalize(cam_direction + u * cam_right + v * cam_up);

        color += traceRay(ray);
    }

    // Average the color
    color /= double(num_samples);

    color = dvec3(linear_to_gamma(color.r), linear_to_gamma(color.g), linear_to_gamma(color.b));

    // Clamp and pack the final color
    ivec3 i_color = ivec3(clamp(color * 255.0, 0.0, 255.0));
    pixels[idx] = packUint8To32(i_color.r, i_color.g, i_color.b, 255);
}
