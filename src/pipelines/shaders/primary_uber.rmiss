#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "rt_common.h"

layout (location = 0) rayPayloadInEXT RTPayload payload; 

void main()
{
#ifdef OUTPUT_COLOR
    payload.color = vec3(0.f, 0.f, 0.f);
#endif

#ifdef OUTPUT_DEPTH
    payload.depth = 0.f;
#endif
}
