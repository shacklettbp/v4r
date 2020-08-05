#version 450
#extension GL_EXT_scalar_block_layout : require

#include "shader_common.h"

// FIXME switch back to a max sized uniform
layout (set = 0, binding = 0) readonly buffer ViewInfos {
    ViewInfo view_info[];
};

layout (push_constant, scalar) uniform PushConstant {
    RenderPushConstant render_const;
};

layout (location = 0) in vec3 in_pos;

layout (location = TXFM1_LOC) in vec4 txfm1;
layout (location = TXFM2_LOC) in vec4 txfm2;
layout (location = TXFM3_LOC) in vec4 txfm3;

#ifdef LIT_PIPELINE
layout (location = NORMAL_IN_LOC) in vec3 in_normal;
layout (location = NORMAL_LOC) out vec3 out_normal;
layout (location = CAMERA_POS_LOC) out vec3 out_camera_pos;

#ifdef USE_NORMAL_MATRIX
layout (location = NORMAL_TXFM1_LOC) in vec3 normal_txfm1;
layout (location = NORMAL_TXFM2_LOC) in vec3 normal_txfm2;
layout (location = NORMAL_TXFM3_LOC) in vec3 normal_txfm3;
#endif

#endif

#ifdef HAS_TEXTURES
layout (location = UV_IN_LOC) in vec2 in_uv;
layout (location = UV_LOC) out vec2 out_uv;
#endif

#ifdef VERTEX_COLOR
layout (location = COLOR_IN_LOC) in vec3 in_color;
layout (location = COLOR_LOC) out vec3 out_color;
#endif

#ifdef OUTPUT_DEPTH
layout (location = DEPTH_LOC) out float out_linear_depth;
#endif

#ifdef HAS_MATERIALS
layout (location = MATERIAL_IN_LOC) in uint in_material_idx;
layout (location = MATERIAL_LOC) out uint material_idx;
#endif

void main() 
{
    mat4 model = mat4(vec4(txfm1.xyz,                 0.f),
                      vec4(txfm1.w, txfm2.xy,         0.f),
                      vec4(txfm2.zw, txfm3.x,         0.f),
                      vec4(txfm3.yzw,                 1.f));

    mat4 mv = view_info[render_const.batchIdx].view * model;

    vec4 camera_space = mv * vec4(in_pos.xyz, 1.f);

    gl_Position = view_info[render_const.batchIdx].projection * camera_space;

#ifdef LIT_PIPELINE

#ifdef USE_NORMAL_MATRIX
    mat3 normal_mat = mat3(normal_txfm1, normal_txfm2, normal_txfm3);
    out_normal = normal_mat * in_normal;
#else
    mat3 normal_mat = mat3(mv);
    vec3 normal_scale = vec3(1.f / dot(normal_mat[0], normal_mat[0]),
                             1.f / dot(normal_mat[1], normal_mat[1]),
                             1.f / dot(normal_mat[2], normal_mat[2]));
    out_normal = normal_mat * in_normal * normal_scale;
#endif

    out_camera_pos = camera_space.xyz;

#endif

#ifdef HAS_TEXTURES
    out_uv = in_uv;
#endif

#ifdef VERTEX_COLOR
    out_color = in_color;
#endif

#ifdef OUTPUT_DEPTH
    out_linear_depth = gl_Position.w;
#endif

#ifdef MATERIAL_PARAMS
    material_idx = in_material_idx;
#endif
}
