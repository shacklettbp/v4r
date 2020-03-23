#include <iostream>
#include <thread>
#include <vector>

#include <v4r.hpp>

#include "dispatch.hpp"
#include "vulkan_config.hpp" // FIXME shouldn't need this
#include "vulkan_state.hpp"
#include "scene.hpp"

using namespace std;

namespace v4r {

template <typename T, typename... Args>
static inline RenderContext::Handle<T> make_handle(Args&&... args)
{
    return RenderContext::Handle<T>(new T(forward<Args>(args)...));
};

template <typename T>
void RenderContext::HandleDeleter<T>::operator()(T *ptr) const
{
    delete ptr;
}
template struct RenderContext::HandleDeleter<SceneID>;
template struct RenderContext::HandleDeleter<CommandStreamState>;

using CommandStream = RenderContext::CommandStream;

CommandStream::CommandStream(CommandStream::StreamStateHandle &&state,
                             RenderContext &global)
    : state_(move(state)),
      global_(global)
{}

RenderResult CommandStream::renderCamera(const SceneHandle &scene)
{
    VkBuffer buf = state_->render(scene->getState());
    return RenderResult {
        nullptr
    };
}

RenderContext::SceneHandle CommandStream::loadScene(const string &file)
{
    SceneID id = global_.scene_mgr_->loadScene(file, *state_);
    return make_handle<SceneID>(id);
}

void CommandStream::dropScene(RenderContext::SceneHandle &&handle)
{
    global_.scene_mgr_->dropScene(move(*handle));
}


RenderContext::RenderContext(const RenderConfig &cfg)
    : state_(make_unique<VulkanState>(cfg)),
      scene_mgr_(make_unique<SceneManager>())
{
}

RenderContext::~RenderContext() = default;

RenderContext::CommandStream RenderContext::makeCommandStream()
{
    // FIXME this should all be handled in VulkanState
    uint32_t stream_idx = state_->numStreams++;
    if (stream_idx == VulkanConfig::num_images_per_fb) {
        cerr << "Maxed out current framebuffer and no switching support" <<
            endl;
        fatalExit();
    }
    auto hdl = make_handle<CommandStreamState>(state_->inst,
                                               state_->dev,
                                               state_->descCfg,
                                               state_->pipeline,
                                               state_->fb,
                                               state_->alloc,
                                               state_->cfg.imgWidth,
                                               state_->cfg.imgHeight,
                                               stream_idx);

    return CommandStream(move(hdl), *this);
}

}
