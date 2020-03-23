#version 450

layout (set = 0, binding = 0) uniform texture2D diffuseColor;
layout (set = 0, binding = 1) uniform sampler textureSampler;

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main() 
{
    outFragColor = texture(sampler2D(diffuseColor, textureSampler), inUV, 0.f);
}
