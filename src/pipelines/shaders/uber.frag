#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require

#include "shader_common.h"

#ifdef LIT_PIPELINE

#include "brdf.glsl"

layout (set = 0, binding = 0) readonly buffer ViewInfos {
    ViewInfo view_info[];
};

layout (push_constant, scalar) uniform PushConstant {
    RenderPushConstant render_const;
};

layout (set = 0, binding = 1, scalar) uniform LightingInfo {
    LightProperties lights[MAX_LIGHTS];
    uint numLights;
} lighting_info;

layout (location = NORMAL_LOC) in vec3 in_normal;
layout (location = CAMERA_POS_LOC) in vec3 in_camera_pos;

#endif

#ifdef HAS_MATERIALS

layout (location = MATERIAL_LOC) flat in uint material_idx;

#endif

#ifdef MATERIAL_PARAMS

struct MaterialParams {
#ifdef ALBEDO_COLOR_UNIFORM
    vec4 albedoColor;
#endif

#ifdef DIFFUSE_COLOR_UNIFORM
    vec4 diffuseColor;
#endif 

#ifdef SPECULAR_COLOR_UNIFORM
    vec4 specularColor;
#endif

#ifdef SHININESS_UNIFORM
    float shininess;
    vec3 pad;
#endif

};

layout (set = 1, binding = PARAM_BIND, scalar) uniform Params {
    MaterialParams material_params[MAX_MATERIALS];
};

#endif

#ifdef HAS_TEXTURES
layout (location = UV_LOC) in vec2 in_uv;

layout (set = 1, binding = 0) uniform sampler texture_sampler;
#endif

#ifdef ALBEDO_COLOR_TEXTURE
layout (set = 1, binding = ALBEDO_COLOR_TEXTURE_BIND)
    uniform texture2D albedo_textures[];
#endif

#ifdef DIFFUSE_COLOR_TEXTURE
layout (set = 1, binding = DIFFUSE_COLOR_TEXTURE_BIND)
    uniform texture2D diffuse_textures[];
#endif

#ifdef SPECULAR_COLOR_TEXTURE
layout (set = 1, binding = SPECULAR_COLOR_TEXTURE_BIND)
    uniform texture2D specular_textures[];
#endif

#ifdef VERTEX_COLOR
layout (location = COLOR_LOC) in vec3 in_vertex_color;
#endif

#ifdef OUTPUT_DEPTH
layout (location = DEPTH_LOC) in float in_linear_depth;
layout (location = DEPTH_OUT_LOC) out float out_linear_depth;
#endif

#ifdef OUTPUT_COLOR
layout (location = COLOR_OUT_LOC) out vec4 out_color;
#endif

#ifdef OUTPUT_COLOR

#ifdef LIT_PIPELINE

vec4 compute_color()
{
#ifdef MATERIAL_PARAMS
    MaterialParams params = material_params[material_idx];
#endif

#if defined(DIFFUSE_COLOR_TEXTURE)
    vec3 diffuse = texture(sampler2D(diffuse_textures[material_idx],
                                     texture_sampler), in_uv, 0.f).xyz;
#elif defined(DIFFUSE_COLOR_UNIFORM)
    vec3 diffuse = params.diffuseColor.xyz;
#endif

#if defined(SPECULAR_COLOR_TEXTURE)
    vec3 specular = texture(sampler2D(specular_textures[material_idx],
                                      texture_sampler), in_uv, 0.f).xyz;
#elif defined(SPECULAR_COLOR_UNIFORM)
    vec3 specular = params.specularColor.xyz;
#endif

#if defined(SHININESS_UNIFORM)
    float shininess = params.shininess;
#endif

    vec3 Lo = vec3(0.0);
    for (int light_idx = 0; light_idx < lighting_info.numLights; light_idx++) {
        vec3 world_light_position =
            lighting_info.lights[light_idx].position.xyz;
        vec3 light_position =
                (view_info[render_const.batchIdx].view *
                    vec4(world_light_position, 1.f)).xyz;
        vec3 light_color = lighting_info.lights[light_idx].color.xyz;
        BRDFParams brdf_params = makeBRDFParams(light_position, in_camera_pos,
                                                in_normal, light_color);

#ifdef BLINN_PHONG
        Lo += blinnPhong(brdf_params, shininess, diffuse, specular);
#endif
    }

    return vec4(Lo, 1.f);
}

#else

vec4 compute_color()
{
#ifdef ALBEDO_COLOR_TEXTURE
    vec4 albedo = texture(sampler2D(albedo_textures[material_idx],
                                    texture_sampler), in_uv, 0.f);
#endif

#ifdef ALBEDO_COLOR_UNIFORM
    vec4 albedo = material_params[material_idx].albedoColor;
#endif

#ifdef ALBEDO_COLOR_VERTEX
    vec4 albedo = vec4(in_vertex_color, 1.f);
#endif

    return albedo;
}

#endif

#endif

void main() 
{
#ifdef OUTPUT_COLOR
    out_color = compute_color();
#endif

#ifdef OUTPUT_DEPTH
    out_linear_depth = in_linear_depth;
#endif
}
