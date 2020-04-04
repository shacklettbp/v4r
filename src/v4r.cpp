#include <iostream>
#include <thread>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include <v4r.hpp>

#include "cuda_state.hpp"
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

glm::mat4 makePerspectiveMatrix(float hfov, uint32_t width, uint32_t height,
                                float near, float far)
{
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    float half_tan = tan(glm::radians(hfov) / 2.f);
    return glm::mat4(1.f / half_tan, 0.f, 0.f, 0.f,
                     0.f, -aspect / half_tan, 0.f, 0.f,
                     0.f, 0.f, far / (near - far), -1.f,
                     0.f, 0.f, far * near / (near - far), 0.f);
}

Camera::Camera(float hfov, uint32_t width, uint32_t height, float near,
               float far, const glm::mat4 &view)
    : state_(new CameraState {
        makePerspectiveMatrix(hfov, width, height, near, far),
        view
    })
{}

Camera::Camera(float hfov, uint32_t width, uint32_t height, float near,
               float far, const glm::vec3 &eye_pos, const glm::vec3 &look_pos,
               const glm::vec3 &up)
    : Camera(hfov, width, height, near, far,
             glm::lookAt(eye_pos, look_pos, up))
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

void Camera::setView(const glm::mat4 &m)
{
    state_->view = m;
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
    auto [color_off, depth_off] = state_->render(scene->getStreamState(),
                                                 camera.getState());

    return RenderResult {
        global_.cuda_->getPointer(color_off),
        global_.cuda_->getPointer(depth_off)
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
      scene_mgr_(make_unique<SceneManager>(cfg.coordinateTransform)),
      cuda_(make_unique<CudaState>(state_->getFramebufferFD(),
                                   state_->getFramebufferBytes()))
{}

RenderContext::~RenderContext() = default;

Camera RenderContext::makeCamera(float hfov, float near, float far,
                                 const glm::vec3 &eye_pos,
                                 const glm::vec3 &look_pos,
                                 const glm::vec3 &up) const
{
    return Camera(hfov, state_->cfg.imgWidth, state_->cfg.imgHeight,
                  near, far, eye_pos, look_pos, up);
}

Camera RenderContext::makeCamera(float hfov, float near, float far,
                                 const glm::mat4 &view) const
{
    return Camera(hfov, state_->cfg.imgWidth, state_->cfg.imgHeight,
                  near, far, view);
}

RenderContext::CommandStream RenderContext::makeCommandStream()
{
    return CommandStream(make_handle<CommandStreamState>(
            state_->makeStreamState()), *this);
}

}
