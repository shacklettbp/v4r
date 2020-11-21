#version 460
#extension GL_EXT_ray_tracing : require

layout (location = 0) rayPayloadEXT vec3 payload; 

void main()
{
    payload = vec3(0.f);
}
