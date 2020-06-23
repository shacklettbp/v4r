#version 450

layout(set = 0, binding = 0) readonly buffer Transforms
{
    mat4 mvp[];
};

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec2 in_uv;

layout (location = 0) out vec2 out_uv;
layout (location = 1) out float out_linear_depth;
layout (location = 2) out uint instance_id;

void main() 
{
    gl_Position = mvp[gl_InstanceIndex] * vec4(in_pos.xyz, 1.0);
    out_linear_depth = gl_Position.w;

    out_uv = in_uv;

    instance_id = gl_InstanceIndex;
}
