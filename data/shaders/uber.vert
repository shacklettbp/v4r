#version 450
#extension GL_EXT_scalar_block_layout : require

#include "shader_common.h"

layout (set = 0, binding = 0) readonly buffer Transforms {
    mat4 modelTransforms[];
};

// FIXME switch back to a max sized uniform
layout (set = 0, binding = 1) readonly buffer ViewInfos {
    ViewInfo view_info[];
};

layout (push_constant, scalar) uniform PushConstant {
    RenderPushConstant render_const;
};

layout (location = 0) in vec3 in_pos;

#ifdef LIT_PIPELINE
layout (location = NORMAL_IN_LOC) in vec3 in_normal;
layout (location = NORMAL_LOC) out vec3 out_normal;
layout (location = CAMERA_POS_LOC) out vec3 out_camera_pos;
#endif

#if defined(TEXTURE_COLOR)
layout (location = UV_IN_LOC) in vec2 in_uv;
layout (location = UV_LOC) out vec2 out_uv;
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
    mat4 model = modelTransforms[gl_InstanceIndex];
    mat4 mv = view_info[render_const.batchIdx].view * model;

    vec4 camera_space = mv * vec4(in_pos.xyz, 1.f);

    gl_Position = view_info[render_const.batchIdx].projection * camera_space;

#ifdef LIT_PIPELINE
    out_normal = mat3(mv) * in_normal;
    out_camera_pos = camera_space.xyz;
#endif

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
