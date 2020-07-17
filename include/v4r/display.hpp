#ifndef V4R_DISPLAY_HPP_INCLUDED
#define V4R_DISPLAY_HPP_INCLUDED

#include <v4r.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace v4r {

struct PresentationState;
class BatchPresentRenderer;

class PresentCommandStream : public CommandStream {

private:
    PresentCommandStream(CommandStream &&base, GLFWwindow *window);
    Handle<PresentationState> presentation_state_;
friend class BatchPresentRenderer;
};

class BatchPresentRenderer : private BatchRenderer {
public:
    BatchPresentRenderer(const RenderConfig &cfg);

    using BatchRenderer::makeLoader;
    PresentCommandStream makeCommandStream(GLFWwindow *window);

    glm::u32vec2 getFrameDimensions() const;
};

}

#endif
