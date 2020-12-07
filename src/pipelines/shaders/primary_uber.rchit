#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_8bit_storage : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_ray_query : require
#extension GL_GOOGLE_include_directive : require

#include "shader_common.h"
#include "brdf.glsl"

struct Vertex
{
    float px;
    float py;
    float pz;
    float nx;
    float ny;
    float nz;
};

layout (push_constant, scalar) uniform PushConstant {
    RenderPushConstant render_const;
};

layout (set = 0, binding = 1) readonly buffer ViewInfos {
    ViewInfo view_info[];
};

layout (set = 0, binding = 3, scalar) uniform LightingInfo {
    LightProperties lights[MAX_LIGHTS];
    uint numLights;
} lighting_info;

layout (set = 1, binding = 0) uniform accelerationStructureEXT tlas;

layout(set = 2, binding = 0, scalar) buffer Vertices {
    Vertex vertices[];
};

layout(set = 2, binding = 1) buffer Indices {
    uint indices[];
};

layout (location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec3 hitAttrs;

void main()
{
    uint base_primitive = gl_InstanceCustomIndexEXT + 3 * gl_PrimitiveID;

    const vec3 barycentrics = vec3(1.f - hitAttrs.x - hitAttrs.y,
        hitAttrs.x, hitAttrs.y);

    float n00 = vertices[indices[nonuniformEXT(base_primitive)]].nx;
    float n01 = vertices[indices[nonuniformEXT(base_primitive)]].ny;
    float n02 = vertices[indices[nonuniformEXT(base_primitive)]].nz;
    float n10 = vertices[indices[nonuniformEXT(base_primitive + 1)]].nx;
    float n11 = vertices[indices[nonuniformEXT(base_primitive + 1)]].ny;
    float n12 = vertices[indices[nonuniformEXT(base_primitive + 1)]].nz;
    float n20 = vertices[indices[nonuniformEXT(base_primitive + 2)]].nx;
    float n21 = vertices[indices[nonuniformEXT(base_primitive + 2)]].ny;
    float n22 = vertices[indices[nonuniformEXT(base_primitive + 2)]].nz;

    vec3 normal = vec3(
        barycentrics.x * n00 + barycentrics.y * n10 + barycentrics.z * n20,
        barycentrics.x * n01 + barycentrics.y * n11 + barycentrics.z * n21,
        barycentrics.x * n02 + barycentrics.y * n12 + barycentrics.z * n22);

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
