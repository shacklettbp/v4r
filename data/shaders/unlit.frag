#version 450

layout (set = 1, binding = 0) uniform texture2D diffuse_tex;
layout (set = 1, binding = 1) uniform sampler texture_sampler;

layout (location = 0) in vec2 in_uv;
layout (location = 1) in float in_linear_depth;

layout (location = 0) out vec4 out_color;
layout (location = 1) out float out_linear_depth;

void main() 
{
    out_color = texture(sampler2D(diffuse_tex, texture_sampler), in_uv, 0.f);

    out_linear_depth = in_linear_depth;
}
