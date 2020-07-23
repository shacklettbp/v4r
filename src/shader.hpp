#ifndef SHADER_HPP_INCLUDED
#define SHADER_HPP_INCLUDED

#include <glm/glm.hpp>

namespace v4r {

namespace Shader {
using namespace glm;
using uint = uint32_t;

#include "../data/shaders/shader_common.h"

};

using Shader::ViewInfo;
using Shader::RenderPushConstant;

}

#endif
