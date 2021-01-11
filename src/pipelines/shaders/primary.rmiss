#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "rt_common.h"

layout (location = 0) rayPayloadInEXT RTPayload payload; 

void main()
{
    payload.color = vec3(0.f, 0.f, 0.f);
}
