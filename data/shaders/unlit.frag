#version 450

layout (set = 1, binding = 0) uniform texture2D diffuseColor;
layout (set = 1, binding = 1) uniform sampler textureSampler;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

void main() 
{
    outColor = texture(sampler2D(diffuseColor, textureSampler), inUV, 0.f);
}
