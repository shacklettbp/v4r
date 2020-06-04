#include <iostream>
#include <thread>
#include <vector>

#include <v4r.hpp>

#include "cuda_state.hpp"
#include "dispatch.hpp"
#include "vulkan_state.hpp"
#include "scene.hpp"

using namespace std;

namespace v4r {

template <typename T, typename... Args>
static inline Handle<T> make_handle(Args&&... args)
{
    return Handle<T>(new T(forward<Args>(args)...));
};

template <typename T>
void HandleDeleter<T>::operator()(T *ptr) const
{
    delete ptr;
}
template struct HandleDeleter<SceneID>;
template struct HandleDeleter<LoaderState>;
template struct HandleDeleter<CommandStreamState>;

static glm::mat4 makePerspectiveMatrix(float hfov, uint32_t width,
                                       uint32_t height, float near, float far)
{
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    float half_tan = tan(glm::radians(hfov) / 2.f);

    return glm::mat4(1.f / half_tan, 0.f, 0.f, 0.f,
                     0.f, -aspect / half_tan, 0.f, 0.f,
                     0.f, 0.f, far / (near - far), -1.f,
                     0.f, 0.f, far * near / (near - far), 0.f);
}

SceneLoader::SceneLoader(Handle<LoaderState> &&state,
                         SceneManager &mgr)
    : state_(move(state)),
      mgr_(mgr)
{}

SceneHandle SceneLoader::loadScene(const string &file)
{
    SceneID id = mgr_.loadScene(file, *state_);
    return make_handle<SceneID>(id);
}

void SceneLoader::dropScene(SceneHandle &&handle)
{
    mgr_.dropScene(move(*handle));
}

CommandStream::CommandStream(Handle<CommandStreamState> &&state,
                             const CudaState &cuda,
                             uint32_t render_width,
                             uint32_t render_height,
                             uint32_t batch_size)
    : state_(move(state)),
      cuda_(cuda),
      render_width_(render_width),
      render_height_(render_height),
      cur_inputs_(batch_size)
{}

FrameBatch CommandStream::getResultsPointer() const
{
    return FrameBatch {
        (uint8_t *)cuda_.getPointer(state_->getColorOffset()),
        (float *)cuda_.getPointer(state_->getDepthOffset())
    };
}

RenderFuture CommandStream::render()
{
    state_->render();

    return RenderFuture {};
}

void CommandStream::initState(uint32_t batch_idx,
                              const SceneHandle &scene,
                              float hfov, float near, float far)
{
    glm::mat4 perspective =
        makePerspectiveMatrix(hfov, render_width_, render_height_,
                              near, far);

    const SceneState &scene_state = scene->getState();

    auto input_ptrs = state_->setSceneRenderState(batch_idx, perspective,
                                                  scene_state);

    RenderInput &cur_input = cur_inputs_[batch_idx];

    cur_input.view = input_ptrs.view;
    cur_input.instanceTransforms = input_ptrs.instances;
    cur_input.numInstances = scene_state.instances.size();
    cur_input.dirty = false;
}

Renderer::Renderer(const RenderConfig &cfg)
    : state_(make_unique<VulkanState>(cfg)),
      scene_mgr_(make_unique<SceneManager>(cfg.coordinateTransform)),
      cuda_(make_unique<CudaState>(state_->getFramebufferFD(),
                                   state_->getFramebufferBytes()))
{}

Renderer::~Renderer() = default;

SceneLoader Renderer::makeLoader()
{
    return SceneLoader(make_handle<LoaderState>(
            state_->makeLoader()),
            *scene_mgr_);
}

CommandStream Renderer::makeCommandStream()
{
    return CommandStream(make_handle<CommandStreamState>(
            state_->makeStream()), *cuda_, state_->cfg.imgWidth,
            state_->cfg.imgHeight, state_->cfg.batchSize);
}

}
