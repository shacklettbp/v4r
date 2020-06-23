#version 450

layout(set = 0, binding = 0) readonly buffer Transforms
{
    mat4 mvp[];
};

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_color;

layout (location = 0) out vec3 out_color;
layout (location = 1) out float out_linear_depth;

void main() 
{
    gl_Position = mvp[gl_InstanceIndex] * vec4(in_pos.xyz, 1.0);
    out_linear_depth = gl_Position.w;

    out_color = in_color;
}
