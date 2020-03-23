#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec3 inColor;

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

layout(push_constant) uniform PushConstants {
	mat4 mvp;
} push_consts;

void main() 
{
	gl_Position = push_consts.mvp * vec4(inPos.xyz, 1.0);

	outColor = inColor;
    outUV = uv;
}
