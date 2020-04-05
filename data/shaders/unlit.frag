#version 450

layout(set = 0, binding = 0) uniform PerViewUBO
{
    mat4 vp;
    vec2 nearFar;
} per_view;

layout (set = 1, binding = 0) uniform texture2D diffuse_tex;
layout (set = 1, binding = 1) uniform sampler texture_sampler;

layout (location = 0) in vec2 in_uv;

layout (location = 0) out vec4 out_color;
layout (location = 1) out float out_depth;

void main() 
{
    out_color = texture(sampler2D(diffuse_tex, texture_sampler), in_uv, 0.f);

    out_depth = gl_FragCoord.z == 1.0 ? 0.0 : 
        per_view.nearFar[1] / (gl_FragCoord.z + per_view.nearFar[0]);
}
