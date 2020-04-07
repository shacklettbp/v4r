#version 450

layout(set = 0, binding = 0) uniform PerViewUBO
{
    mat4 vp;
} per_view;

layout(push_constant) uniform PushConstants {
	mat4 modelTransform;
} push_consts;

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec2 in_uv;

layout (location = 0) out vec2 out_uv;
layout (location = 1) out float out_linear_depth;

void main() 
{
    mat4 mvp = per_view.vp * push_consts.modelTransform;

    gl_Position = mvp * vec4(in_pos.xyz, 1.0);
    out_linear_depth = gl_Position.w;

    out_uv = in_uv;
}
