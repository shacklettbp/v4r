#version 460
#extension GL_EXT_ray_tracing : require

layout (location = 0) rayPayloadInEXT vec3 payload; 

void main()
{
    payload = vec3(0.f, 0.f, 0.f);
}
