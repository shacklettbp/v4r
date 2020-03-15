#include <iostream>
#include <thread>
#include <vector>

#include <v4r.hpp>

#include "dispatch.hpp"
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

CommandStream::CommandStream(CommandStream::StreamStateHandle &&state)
    : state_(move(state))
{}

RenderResult CommandStream::renderCamera(const SceneHandle &scene)
{
    return RenderResult {
        nullptr
    };
}

RenderContext::RenderContext(const RenderConfig &cfg)
    : state_(make_unique<VulkanState>(cfg)),
      scene_mgr_(make_unique<SceneManager>(*state_))
{
}

RenderContext::~RenderContext() = default;


RenderContext::SceneHandle RenderContext::loadScene(const string &file)
{
    SceneID id = scene_mgr_->loadScene(file);
    return make_handle<SceneID>(id);
}

void RenderContext::dropScene(RenderContext::SceneHandle &&handle)
{
    scene_mgr_->dropScene(move(*handle));
}

RenderContext::CommandStream RenderContext::makeCommandStream() const
{
    auto hdl = make_handle<CommandStreamState>(state_->dev, state_->fbCfg,
                                               state_->pipeline);

    return CommandStream(move(hdl));
}

}
