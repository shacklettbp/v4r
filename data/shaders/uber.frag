#version 450
#extension GL_EXT_nonuniform_qualifier : require

#ifdef FRAG_NEED_MATERIAL
layout (set = 0, binding = 1) readonly buffer MaterialParams
{
    uint textureIdx[];
};

layout (location = INSTANCE_LOC) flat in uint instance_id;
#endif

#if defined(TEXTURE_COLOR)
layout (set = 1, binding = 0) uniform texture2D textures[];
layout (set = 1, binding = 1) uniform sampler texture_sampler;

layout (location = UV_LOC) in vec2 in_uv;
#elif defined(VERTEX_COLOR)
layout (location = COLOR_LOC) in vec3 in_vertex_color;
#endif

#ifdef OUTPUT_DEPTH
layout (location = DEPTH_LOC) in float in_linear_depth;
layout (location = DEPTH_OUT_LOC) out float out_linear_depth;
#endif

#ifdef OUTPUT_COLOR
layout (location = COLOR_OUT_LOC) out vec4 out_color;
#endif

void main() 
{
#if defined(TEXTURE_COLOR)
    uint tex_idx = textureIdx[instance_id];
    out_color = texture(sampler2D(textures[tex_idx],
                                  texture_sampler), in_uv, 0.f);
#elif defined(VERTEX_COLOR)
    out_color = vec4(in_vertex_color, 1.f);
#endif

#ifdef OUTPUT_DEPTH
    out_linear_depth = in_linear_depth;
#endif
}
