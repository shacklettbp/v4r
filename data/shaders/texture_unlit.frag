#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout (set = 0, binding = 1) readonly buffer MaterialParams
{
    uint textureIdx[];
};

layout (set = 1, binding = 0) uniform texture2D textures[];
layout (set = 1, binding = 1) uniform sampler texture_sampler;

layout (location = 0) in vec2 in_uv;
layout (location = 1) in float in_linear_depth;
layout (location = 2) flat in uint instance_id;

layout (location = 0) out vec4 out_color;
layout (location = 1) out float out_linear_depth;

void main() 
{
    uint tex_idx = textureIdx[instance_id];
    out_color = texture(sampler2D(textures[tex_idx],
                                  texture_sampler), in_uv, 0.f);

    out_linear_depth = in_linear_depth;
}
