#ifndef SHADER_COMMON_H_INCLUDED
#define SHADER_COMMON_H_INCLUDED

struct ViewInfo {
    mat4 projection;
    mat4 view;
};

struct RenderPushConstant {
    uint batchIdx;
};

struct LightProperties {
    vec4 position;
    vec4 color;
};

#define MAX_MATERIALS (1000)
#define MAX_LIGHTS (2000)
#define WORKGROUP_SIZE (32)
#define MAX_BATCH_SIZE (1024)

#endif
