#ifndef V4R_CONFIG_HPP_INCLUDED
#define V4R_CONFIG_HPP_INCLUDED

#include <glm/glm.hpp>

namespace v4r {

struct RenderConfig {
    int gpuID;
    uint32_t numLoaders;
    uint32_t numStreams;
    uint32_t batchSize;
    uint32_t imgWidth;
    uint32_t imgHeight;
    glm::mat4 coordinateTransform;
};

}

#endif
