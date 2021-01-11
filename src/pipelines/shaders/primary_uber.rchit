#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_8bit_storage : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_ray_query : require
#extension GL_GOOGLE_include_directive : require

#include "shader_common.h"
#include "rt_common.h"
#include "brdf.glsl"

struct Vertex
{
RCHIT_VERTEX_ATTRS
};

layout (push_constant, scalar) uniform PushConstant {
    RenderPushConstant render_const;
};

layout (set = 0, binding = 0) readonly buffer ViewInfos {
    ViewInfo view_info[];
};

#ifdef HAS_MATERIALS

layout (set = 0, binding = RCHIT_MAT_IDX_POS) readonly buffer MatIdxs {
    uint material_indices[];
};

#endif

#ifdef LIT_PIPELINE
layout (set = 0, binding = RCHIT_LIGHT_POS, scalar) uniform LightingInfo {
    LightProperties lights[MAX_LIGHTS];
    uint numLights;
} lighting_info;

layout (set = 1, binding = 0) uniform accelerationStructureEXT tlas;
#endif

layout (set = 2, binding = 0, scalar) buffer Vertices {
    Vertex vertices[];
};

layout (set = 2, binding = 1) buffer Indices {
    uint indices[];
};

#ifdef HAS_TEXTURES
layout (set = 2, binding = 2) uniform sampler texture_sampler;
#endif

#ifdef ALBEDO_COLOR_TEXTURE
layout (set = 2, binding = RCHIT_ALBEDO_COLOR_TEXTURE_BIND)
    uniform texture2D albedo_textures[];
#endif

#ifdef DIFFUSE_COLOR_TEXTURE
layout (set = 2, binding = RCHIT_DIFFUSE_COLOR_TEXTURE_BIND)
    uniform texture2D diffuse_textures[];
#endif

#ifdef SPECULAR_COLOR_TEXTURE
layout (set = 2, binding = RCHIT_SPECULAR_COLOR_TEXTURE_BIND)
    uniform texture2D specular_textures[];
#endif

#ifdef MATERIAL_PARAMS
struct MaterialParams {
    vec4 data[NUM_PARAM_VECS];
};

layout (set = 2, binding = RCHIT_PARAM_BIND) uniform Params {
    MaterialParams material_params[MAX_MATERIALS];
};
#endif

layout (location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec3 hitAttrs;

#ifdef LIT_PIPELINE
vec3 readNormal(vec3 barycentrics, vec3 idxs)
{
    float n00 = vertices[idxs.x].nx;
    float n01 = vertices[idxs.x].ny;
    float n02 = vertices[idxs.x].nz;
    float n10 = vertices[idxs.y].nx;
    float n11 = vertices[idxs.y].ny;
    float n12 = vertices[idxs.y].nz;
    float n20 = vertices[idxs.z].nx;
    float n21 = vertices[idxs.z].ny;
    float n22 = vertices[idxs.z].nz;

    return vec3(
        barycentrics.x * n00 + barycentrics.y * n10 + barycentrics.z * n20,
        barycentrics.x * n01 + barycentrics.y * n11 + barycentrics.z * n21,
        barycentrics.x * n02 + barycentrics.y * n12 + barycentrics.z * n22);
}
#endif

#ifdef HAS_TEXTURES
vec2 readUV(vec3 barycentrics, vec3 idxs)
{
    float u00 = vertices[idxs.x].ux;
    float u01 = vertices[idxs.x].uy;
    float u10 = vertices[idxs.y].ux;
    float u11 = vertices[idxs.y].uy;
    float u20 = vertices[idxs.z].ux;
    float u21 = vertices[idxs.z].uy;

    return vec2(
        barycentrics.x * u00 + barycentrics.y * u10 + barycentrics.z * u20,
        barycentrics.x * u01 + barycentrics.y * u11 + barycentrics.z * u21);
}
#endif

#ifdef HAS_VERTEX_COLOR
vec3 readVertexColor(vec3 barycentrics, vec3 idxs)
{
    float c00 = vertices[idxs.x].cx;
    float c01 = vertices[idxs.x].cy;
    float c02 = vertices[idxs.x].cz;
    float c10 = vertices[idxs.y].cx;
    float c11 = vertices[idxs.y].cy;
    float c12 = vertices[idxs.y].cz;
    float c20 = vertices[idxs.z].cx;
    float c21 = vertices[idxs.z].cy;
    float c22 = vertices[idxs.z].cz;

    return vec3(
        barycentrics.x * c00 + barycentrics.y * c10 + barycentrics.z * c20,
        barycentrics.x * c01 + barycentrics.y * c11 + barycentrics.z * c21,
        barycentrics.x * c02 + barycentrics.y * c12 + barycentrics.z * c22);
}
#endif

#ifdef LIT_PIPELINE

void computeColor()
{
    uint base_primitive = gl_InstanceCustomIndexEXT + 3 * gl_PrimitiveID;

    const vec3 barycentrics = vec3(1.f - hitAttrs.x - hitAttrs.y,
        hitAttrs.x, hitAttrs.y);

    const uvec3 idxs = uvec3(indices[nonuniformEXT(base_primitive)],
                             indices[nonuniformEXT(base_primitive + 1)],
                             indices[nonuniformEXT(base_primitive + 2)]);

    vec3 normal = readNormal(barycentrics, idxs);

    mat4 mv = view_info[render_const.batchIdx].view * mat4(gl_ObjectToWorldEXT);

    mat3 normal_mat = mat3(mv);
    vec3 normal_scale = vec3(1.f / dot(normal_mat[0], normal_mat[0]),
                             1.f / dot(normal_mat[1], normal_mat[1]),
                             1.f / dot(normal_mat[2], normal_mat[2]));
    normal = normal_mat * normal * normal_scale;
    
    vec3 Lo = vec3(0.f);
    for (int light_idx = 0; light_idx < lighting_info.numLights; light_idx++) {
        vec3 world_light_position =
            lighting_info.lights[light_idx].position.xyz;
        vec3 light_position =
                (view_info[render_const.batchIdx].view *
                    vec4(world_light_position, 1.f)).xyz;
        vec3 light_color = lighting_info.lights[light_idx].color.xyz;

        vec3 world_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT *
            gl_HitTEXT;

        vec3 camera_pos = (view_info[render_const.batchIdx].view *
            vec4(world_pos, 1.f)).xyz;

        BRDFParams brdf_params = makeBRDFParams(light_position, camera_pos,
                                                normal, light_color);

        rayQueryEXT shadow_query;
        rayQueryInitializeEXT(shadow_query, tlas,
                              gl_RayFlagsTerminateOnFirstHitEXT,
                              0xFF, world_pos, 0.01f,
                              world_light_position - world_pos, 10000.f);
        
        while(rayQueryProceedEXT(shadow_query)) {}
        
        if (rayQueryGetIntersectionTypeEXT(shadow_query, true) ==
            gl_RayQueryCommittedIntersectionNoneEXT)
        {
            Lo += blinnPhong(brdf_params, 32.f, vec3(1.f), vec3(1.f));
        }
    }

    payload = Lo;
}

#endif

#ifdef OUTPUT_DEPTH
float computeDepth()
{
}
#endif

void main()
{
#ifdef OUTPUT_COLOR
    payload.color = computeColor();
#endif

#ifdef OUTPUT_DEPTH
    payload.depth = computeDepth();
#endif
}
