#ifndef SHADER_HPP_INCLUDED
#define SHADER_HPP_INCLUDED

#include <glm/glm.hpp>

namespace v4r {

namespace Shader {
using namespace glm;
using uint = uint32_t;

#include "pipelines/shaders/shader_common.h"
#include "pipelines/shaders/mesh_common.h"
};

using Shader::ViewInfo;
using Shader::RenderPushConstant;
using Shader::CullPushConstant;
using Shader::DrawInput;
using Shader::Meshlet;
using Shader::MeshChunk;
using Shader::MeshInfo;
using Shader::FrustumBounds;

namespace VulkanConfig {

constexpr uint32_t max_materials = MAX_MATERIALS;
constexpr uint32_t max_lights = MAX_LIGHTS;
constexpr uint32_t max_instances = 10000000;
constexpr uint32_t compute_workgroup_size = WORKGROUP_SIZE;

}

}

#endif
