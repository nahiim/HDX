#version 450

layout (binding = 1) uniform sampler2D samplerColor;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout(binding = 0) uniform Light
{
    vec4 position;
    vec4 color;
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 view_pos;
    float zNear;
    float zFar;
};

float LinearizeDepth(float depth)
{
  float n = zNear;
  float f = zFar;
  float z = depth;
  return (2.0 * n) / (f + n - z * (f - n));	
}

void main() 
{
	float depth = texture(samplerColor, inUV).r;
	outFragColor = vec4(vec3(depth), 1.0);
}
