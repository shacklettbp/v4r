#include <iostream>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>

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

RenderResult CommandStream::renderCamera()
{
    return RenderResult();
}

RenderContext::RenderContext(int gpu_id)
    : state_(make_unique<VulkanState>(gpu_id))
{
}

RenderContext::~RenderContext() = default;


RenderContext::Handle<SceneID> RenderContext::loadScene(const std::string &file)
{
    return make_handle<SceneID>();
}

void RenderContext::dropScene(RenderContext::Handle<SceneID> &&handle)
{
}

RenderContext::CommandStream RenderContext::makeCommandStream() const
{
    auto hdl = make_handle<CommandStreamState>(state_->dev);

    return CommandStream(move(hdl));
}

}
