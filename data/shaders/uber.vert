#version 450

layout(set = 0, binding = 0) readonly buffer Transforms
{
    mat4 mvp[];
};

layout (location = 0) in vec3 in_pos;

#if defined(TEXTURE_COLOR)
layout (location = UV_IN_LOC ) in vec2 in_uv;
layout (location = UV_LOC ) out vec2 out_uv;
#elif defined(VERTEX_COLOR)
layout (location = COLOR_IN_LOC) in vec3 in_color;
layout (location = COLOR_LOC) out vec3 out_color;
#endif

#ifdef OUTPUT_DEPTH
layout (location = DEPTH_LOC) out float out_linear_depth;
#endif

#ifdef FRAG_NEED_MATERIAL
layout (location = INSTANCE_LOC) out uint instance_id;
#endif

void main() 
{
    gl_Position = mvp[gl_InstanceIndex] * vec4(in_pos.xyz, 1.0);

#if defined(TEXTURE_COLOR)
    out_uv = in_uv;
#elif defined(VERTEX_COLOR)
    out_color = in_color;
#endif

#ifdef OUTPUT_DEPTH
    out_linear_depth = gl_Position.w;
#endif

#ifdef FRAG_NEED_MATERIAL
    instance_id = gl_InstanceIndex;
#endif
}
