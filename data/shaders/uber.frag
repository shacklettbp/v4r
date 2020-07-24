#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

#include "shader_common.h"

#ifdef FRAG_NEED_MATERIAL
#ifdef TEXTURE_COLOR // FIXME

struct Material {
#ifdef TEXTURE_COLOR
    uint textureIdx;
#endif
};

layout (set = 0, binding = MATERIAL_BIND, scalar) readonly buffer
MaterialParams {
    Material materials[];
};

layout (location = INSTANCE_LOC) flat in uint instance_id;

#endif
#endif

#ifdef LIT_PIPELINE

layout (set = 0, binding = 0) readonly buffer ViewInfos {
    ViewInfo view_info[];
};

layout (push_constant, scalar) uniform PushConstant {
    RenderPushConstant render_const;
};

#include "brdf.glsl"

// FIXME
#define MAX_LIGHTS 500

layout (set = 0, binding = LIGHT_BIND, scalar) uniform LightingInfo {
    LightProperties lights[MAX_LIGHTS];
    uint numLights;
} lighting_info;

layout (location = NORMAL_LOC) in vec3 in_normal;
layout (location = CAMERA_POS_LOC) in vec3 in_camera_pos;

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
    uint tex_idx = materials[instance_id].textureIdx;
    vec4 albedo = texture(sampler2D(textures[tex_idx], texture_sampler),
                          in_uv, 0.f);
#elif defined(VERTEX_COLOR)
    vec4 albedo = vec4(in_vertex_color, 1.f);

#elif defined(OUTPUT_COLOR)
    // No vertex color, no textures, but still want color...
    // Make it white

    vec4 albedo = vec4(1.f);
#endif

#ifdef OUTPUT_COLOR

#if defined(LIT_PIPELINE)

    vec3 Lo = vec3(0.0);
    for (int light_idx = 0; light_idx < lighting_info.numLights; light_idx++) {
        vec3 world_light_position =
            lighting_info.lights[light_idx].position.xyz;
        vec3 light_position =
                (view_info[render_const.batchIdx].view *
                    vec4(world_light_position, 1.f)).xyz;
        vec3 light_color = lighting_info.lights[light_idx].color.xyz;
        BRDFParams params = makeBRDFParams(light_position, in_camera_pos,
                                           in_normal, light_color,
                                           albedo.xyz);

        Lo += blinnPhong(params);
    }

    out_color = vec4(Lo, 1.f);

#else

    out_color = albedo;

#endif

#endif

#ifdef OUTPUT_DEPTH
    out_linear_depth = in_linear_depth;
#endif
}
