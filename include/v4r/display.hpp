#ifndef V4R_DISPLAY_HPP_INCLUDED
#define V4R_DISPLAY_HPP_INCLUDED

#include <v4r.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace v4r {

struct PresentationState;
class BatchPresentRenderer;

class PresentCommandStream : public CommandStream {
public:

    RenderSync render(const std::vector<Environment> &elems);

private:
    PresentCommandStream(CommandStream &&base, GLFWwindow *window,
                         bool benchmark_mode);

    Handle<PresentationState> presentation_state_;
    bool benchmark_mode_;
friend class BatchPresentRenderer;
};

class BatchPresentRenderer : private BatchRenderer {
public:
    template <typename PipelineType>
    BatchPresentRenderer(const RenderConfig &cfg,
                         const RenderFeatures<PipelineType> &features,
                         bool benchmark_mode);

    using BatchRenderer::makeLoader;
    PresentCommandStream makeCommandStream(GLFWwindow *window);

    glm::u32vec2 getFrameDimensions() const;

private:
    bool benchmark_mode_;
};

}

#endif
