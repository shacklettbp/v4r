#include <iostream>
#include <thread>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

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

Camera::Camera(float fov, float aspect, float near, float far,
               const glm::vec3 &eye_pos, const glm::vec3 &look_pos,
               const glm::vec3 &up)
    : state_(new CameraState {
        glm::perspective(fov, aspect, near, far),
        glm::lookAt(eye_pos, look_pos, up)
    })
{}

Camera::Camera(Camera &&o)
    : state_(move(o.state_))
{}

Camera::~Camera() = default;

void Camera::rotate(float angle, const glm::vec3 &axis)
{
    glm::rotate(state_->view, angle, axis);
}

void Camera::translate(const glm::vec3 &v)
{
    glm::translate(state_->view, v);
}

const CameraState & Camera::getState() const
{
    return *state_;
}

using CommandStream = RenderContext::CommandStream;

CommandStream::CommandStream(CommandStream::StreamStateHandle &&state,
                             RenderContext &global)
    : state_(move(state)),
      global_(global)
{}

RenderResult CommandStream::render(const SceneHandle &scene, const Camera &camera)
{
    VkBuffer buf = state_->render(scene->getStreamState(), camera.getState());
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
    global_.scene_mgr_->dropScene(move(*handle), *state_);
}


RenderContext::RenderContext(const RenderConfig &cfg)
    : state_(make_unique<VulkanState>(cfg)),
      scene_mgr_(make_unique<SceneManager>())
{
}

RenderContext::~RenderContext() = default;

RenderContext::CommandStream RenderContext::makeCommandStream()
{
    return CommandStream(make_handle<CommandStreamState>(
            state_->makeStreamState()), *this);
}

}
