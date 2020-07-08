#include <v4r.hpp>

#include "cuda_state.hpp"
#include "dispatch.hpp"
#include "vulkan_state.hpp"
#include "scene.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

namespace v4r {

struct SyncState {
    const DeviceState &dev;
    const VkFence fence;
    cudaExternalSemaphore_t cudaSemaphore;
};

template <typename T, typename... Args>
static inline Handle<T> make_handle(Args&&... args)
{
    return Handle<T>(new T(forward<Args>(args)...));
};

template <typename T>
void HandleDeleter<T>::operator()(remove_extent_t<T> *ptr) const
{
    if constexpr (is_array_v<T>) {
        delete[] ptr;
    } else {
        delete ptr;
    }
}

template struct HandleDeleter<LoaderState>;
template struct HandleDeleter<CommandStreamState>;
template struct HandleDeleter<CudaStreamState[]>;
template struct HandleDeleter<SyncState[]>;
template struct HandleDeleter<VulkanState>;
template struct HandleDeleter<CudaState>;
template struct HandleDeleter<EnvironmentState>;

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

AssetLoader::AssetLoader(Handle<LoaderState> &&state)
    : state_(move(state))
{}

shared_ptr<Mesh>
AssetLoader::loadMesh(const string &geometry_path)
{
    return state_->assetHelper.loadMesh(geometry_path);
}

template <typename VertexType>
shared_ptr<Mesh> AssetLoader::loadMesh(
        vector<VertexType> vertices, vector<uint32_t> indices)
{
    return state_->makeMesh(move(vertices), move(indices));
}

template shared_ptr<Mesh> AssetLoader::loadMesh(
        vector<UnlitRendererInputs::NoColorVertex>, vector<uint32_t>);
template shared_ptr<Mesh> AssetLoader::loadMesh(
        vector<UnlitRendererInputs::ColoredVertex>, vector<uint32_t>);
template shared_ptr<Mesh> AssetLoader::loadMesh(
        vector<UnlitRendererInputs::TexturedVertex>, vector<uint32_t>);
template shared_ptr<Mesh> AssetLoader::loadMesh(
        vector<LitRendererInputs::NoColorVertex>, vector<uint32_t>);
template shared_ptr<Mesh> AssetLoader::loadMesh(
        vector<LitRendererInputs::TexturedVertex>, vector<uint32_t>);

shared_ptr<Texture> AssetLoader::loadTexture(
        const string &texture_path)
{
    ifstream texture_file(texture_path, ios::in | ios::binary);
    if (!texture_file) {
        cerr << "Failed to read texture at " << texture_path << endl;
        fatalExit();
    }

    texture_file.seekg(0, ios::end);

    vector<uint8_t> raw(texture_file.tellg());

    texture_file.seekg(0, ios::beg);
    texture_file.read(reinterpret_cast<char *>(raw.data()), raw.size());
    texture_file.close();

    return state_->loadTexture(raw);
}

template <typename MaterialDescType>
shared_ptr<Material> AssetLoader::makeMaterial(
        MaterialDescType params)
{
    return state_->makeMaterial(params);
}

template shared_ptr<Material> AssetLoader::makeMaterial(
        UnlitRendererInputs::MaterialDescription);
template shared_ptr<Material> AssetLoader::makeMaterial(
        LitRendererInputs::MaterialDescription);

shared_ptr<Scene> AssetLoader::makeScene(
        const SceneDescription &desc)
{
    return state_->loadScene(desc);
}

shared_ptr<Scene> AssetLoader::loadScene(
        const string &scene_path)
{
    SceneDescription desc =
        state_->assetHelper.parseScene(scene_path,
                                       state_->coordinateTransform);

    return makeScene(desc);
}

RenderSync::RenderSync(const SyncState *state)
    : state_(state)
{}

void RenderSync::gpuWait(cudaStream_t strm)
{
    cudaGPUWait(state_->cudaSemaphore, strm);
}

void RenderSync::cpuWait()
{
    assert(state_->fence != VK_NULL_HANDLE);
    waitForFenceInfinitely(state_->dev, state_->fence);
    resetFence(state_->dev, state_->fence);
}

static CudaStreamState * makeCudaStreamStates(
        const CommandStreamState &cmd_stream,
        const CudaState &cuda_global,
        bool double_buffered)
{
    cuda_global.setActiveDevice();

    if (double_buffered) {
        return new CudaStreamState[2] {
            CudaStreamState {
                (uint8_t *)cuda_global.getPointer(
                        cmd_stream.getColorOffset(0)),
                (float *)cuda_global.getPointer(
                        cmd_stream.getDepthOffset(0)),
                cmd_stream.getSemaphoreFD(0)
            },
            CudaStreamState {
                (uint8_t *)cuda_global.getPointer(
                        cmd_stream.getColorOffset(1)),
                (float *)cuda_global.getPointer(
                        cmd_stream.getDepthOffset(1)),
                cmd_stream.getSemaphoreFD(1)
            }
        };
    } else {
        return new CudaStreamState[1] {
            CudaStreamState {
                (uint8_t *)cuda_global.getPointer(
                        cmd_stream.getColorOffset(0)),
                (float *)cuda_global.getPointer(
                        cmd_stream.getDepthOffset(0)),
                cmd_stream.getSemaphoreFD(0)
            }
        };
    }
}

static SyncState * makeSyncs(const CommandStreamState &cmd_stream,
                             const Handle<CudaStreamState[]> &cuda_states,
                             bool double_buffered)
{
    if (double_buffered) {
        return new SyncState[2] {
            SyncState {
                cmd_stream.dev,
                cmd_stream.getFence(0),
                cuda_states[0].getSemaphore()
            }, 
            SyncState {
                cmd_stream.dev,
                cmd_stream.getFence(1),
                cuda_states[1].getSemaphore()
            }
        };
    } else {
        return new SyncState[1] {
            SyncState {
                cmd_stream.dev,
                cmd_stream.getFence(0),
                cuda_states[0].getSemaphore()
            }
        };
    }
}

CommandStream::CommandStream(Handle<CommandStreamState> &&state,
                             const CudaState &cuda_global,
                             uint32_t render_width,
                             uint32_t render_height,
                             bool double_buffered)
    : state_(move(state)),
      cuda_(makeCudaStreamStates(*state_, cuda_global, double_buffered)),
      sync_(makeSyncs(*state_, cuda_, double_buffered)),
      render_width_(render_width),
      render_height_(render_height)
{}

Environment CommandStream::makeEnvironment(const shared_ptr<Scene> &scene,
                                           float hfov, float near, float far)
{
    glm::mat4 perspective =
        makePerspectiveMatrix(hfov, render_width_, render_height_,
                              near, far);

    return Environment(make_handle<EnvironmentState>(scene, perspective));
}

uint8_t * CommandStream::getColorDevPtr(bool alternate_buffer) const
{
    return cuda_[alternate_buffer].getColor();
}

float * CommandStream::getDepthDevPtr(bool alternate_buffer) const
{
    return cuda_[alternate_buffer].getDepth();
}

RenderSync CommandStream::render(const std::vector<Environment> &elems)
{
    uint32_t frame_idx = state_->render(elems);

    return RenderSync(&sync_[frame_idx]);
}

BatchRenderer::BatchRenderer(const RenderConfig &cfg)
    : state_(make_handle<VulkanState>(cfg, getUUIDFromCudaID(cfg.gpuID))),
      cuda_(make_handle<CudaState>(cfg.gpuID, state_->getFramebufferFD(),
                                   state_->getFramebufferBytes()))
{}

AssetLoader BatchRenderer::makeLoader()
{
    return AssetLoader(make_handle<LoaderState>(
            state_->makeLoader()));
}

CommandStream BatchRenderer::makeCommandStream()
{
    auto stream_state = make_handle<CommandStreamState>(state_->makeStream());

    return CommandStream(move(stream_state), *cuda_,
                         state_->cfg.imgWidth, state_->cfg.imgHeight,
                         state_->cfg.features.options &
                             RenderFeatures::Options::DoubleBuffered);
}

Environment::Environment(Handle<EnvironmentState> &&state)
    : state_(move(state)),
      view_(),
      index_map_(state_->scene->envDefaults.indexMap),
      transforms_(state_->scene->envDefaults.transforms),
      materials_(state_->scene->envDefaults.materials)
{}

uint32_t Environment::addInstance(uint32_t model_idx, uint32_t material_idx,
                                  const glm::mat4 &model_matrix)
{
    transforms_[model_idx].emplace_back(model_matrix);
    materials_[model_idx].emplace_back(material_idx);
    uint32_t instance_idx = transforms_[model_idx].size() - 1;

    uint32_t outer_id;
    if (state_->freeIDs.size() > 0) {
        uint32_t free_id = state_->freeIDs.back();
        state_->freeIDs.pop_back();
        index_map_[free_id].first = model_idx;
        index_map_[free_id].second = instance_idx;

        outer_id = free_id;
    } else {
        index_map_.emplace_back(model_idx, instance_idx);
        outer_id = index_map_.size() - 1;
    }

    state_->reverseIDMap[model_idx].emplace_back(outer_id);

    return outer_id;
}

void Environment::deleteInstance(uint32_t inst_id)
{
    auto [model_idx, instance_idx] = index_map_[inst_id];
    auto &transforms = transforms_[model_idx];
    auto &materials = materials_[model_idx];
    auto &reverse_ids = state_->reverseIDMap[model_idx];

    if (transforms.size() == 1) {
        transforms.clear();
        materials.clear();
        reverse_ids.clear();
    } else {
        // Keep contiguous
        transforms[instance_idx] = transforms.back();
        materials[instance_idx] = materials.back();
        reverse_ids[instance_idx] = reverse_ids.back();
        index_map_[reverse_ids[instance_idx]] = { model_idx, instance_idx };

        transforms.pop_back();
        materials.pop_back();
        reverse_ids.pop_back();
    }

    state_->freeIDs.push_back(inst_id);
}

uint32_t Environment::addLight(const glm::vec3 &position,
                               const glm::vec3 &color)
{
    state_->lights.push_back({
        glm::vec4(position, 1.f),
        glm::vec4(color, 1.f)
    });

    uint32_t light_idx = state_->lights.size() - 1;

    uint32_t light_id;
    if (state_->freeLightIDs.size() > 0) {
        uint32_t free_id = state_->freeLightIDs.back();
        state_->freeLightIDs.pop_back();
        state_->lightIDs[free_id] = light_idx;

        light_id = free_id;
    } else {
        state_->lightIDs.push_back(light_idx);
        light_id = state_->lightIDs.size() - 1;
    }

    state_->lightReverseIDs[light_id] = light_idx;
    return light_id;
}

void Environment::deleteLight(uint32_t light_id)
{
    uint32_t light_idx = state_->lightIDs[light_id];
    if (state_->lights.size() == 1) {
        state_->lights.clear();
        state_->lightIDs.clear();
        state_->lightReverseIDs.clear();
    } else {
        state_->lights[light_idx] = state_->lights.back();
        state_->lightReverseIDs[light_idx] = state_->lightReverseIDs.back();
        state_->lightIDs[state_->lightReverseIDs[light_idx]] = light_idx;

        state_->lights.pop_back();
        state_->lightReverseIDs.pop_back();
    }

    state_->freeLightIDs.push_back(light_id);
}

}
