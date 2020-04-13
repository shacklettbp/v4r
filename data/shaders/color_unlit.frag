#version 450

layout (location = 0) in vec3 in_vertex_color;
layout (location = 1) in float in_linear_depth;

layout (location = 0) out vec4 out_color;
layout (location = 1) out float out_linear_depth;

void main() 
{
    out_color = vec4(in_vertex_color, 1.f);

    out_linear_depth = in_linear_depth;
}
