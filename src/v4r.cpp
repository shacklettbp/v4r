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
static inline RenderContext::Handle<T> make_handle(Args&&... args) {
    return RenderContext::Handle<T>(new T(forward<Args>(args)...));
};

template <typename T>
void RenderContext::HandleDeleter<T>::operator()(T *ptr) const {
    delete ptr;
}
template struct RenderContext::HandleDeleter<SceneID>;

RenderContext::RenderContext(int gpu_id)
    : state_(make_unique<VulkanState>(gpu_id))
{
}

RenderContext::~RenderContext() = default;


RenderContext::Handle<SceneID> RenderContext::loadScene(const std::string &file) {
    return make_handle<SceneID>();
}

void RenderContext::dropScene(RenderContext::Handle<SceneID> &&handle) {
}

}
