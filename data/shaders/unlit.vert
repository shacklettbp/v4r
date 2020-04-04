#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 uv;

layout (location = 0) out vec2 outUV;

layout(set = 0, binding = 0) uniform PerViewUBO
{
    mat4 vp;
} per_view;

layout(push_constant) uniform PushConstants {
	mat4 modelTransform;
} push_consts;

void main() 
{
    mat4 mvp = per_view.vp * push_consts.modelTransform;

    gl_Position = mvp * vec4(inPos.xyz, 1.0);

    outUV = uv;
}
