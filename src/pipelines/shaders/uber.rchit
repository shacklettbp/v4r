#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_8bit_storage : enable

struct Vertex
{
    vec3 pos;
    u8vec4 color;
};

layout(location = 0) rayPayloadInEXT vec3 hitPayload;

layout(set = 0, binding = 1, scalar) buffer Vertices {
    Vertex vertices[];
};

layout(set = 0, binding = 2) buffer Indices {
    uint indices[];
};

hitAttributeEXT vec3 hitAttrs;

void main()
{

    vec3 color0 = uvec3(vertices[indices[3 * gl_PrimitiveID]].color).xyz / 255.f;
    vec3 color1 = uvec3(vertices[indices[3 * gl_PrimitiveID + 1]].color).xyz / 255.f;
    vec3 color2 = uvec3(vertices[indices[3 * gl_PrimitiveID + 2]].color).xyz / 255.f;

    const vec3 barycentrics = vec3(1.f - hitAttrs.x - hitAttrs.y,
        hitAttrs.x, hitAttrs.y);
    
    hitPayload = color0 * barycentrics.x +
        color1 * barycentrics.y +
        color2 * barycentrics.z;
}
