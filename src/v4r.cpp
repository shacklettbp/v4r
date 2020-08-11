#include <v4r.hpp>

#include "dispatch.hpp"
#include "vulkan_state.hpp"
#include "scene.hpp"
#include "cuda_state.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

namespace v4r {

template struct HandleDeleter<LoaderState>;
template struct HandleDeleter<CommandStreamState>;
template struct HandleDeleter<VulkanState>;
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
    return state_->loadMesh(geometry_path);
}

template <typename VertexType>
shared_ptr<Mesh> AssetLoader::loadMesh(
        vector<VertexType> vertices, vector<uint32_t> indices)
{
    return state_->makeMesh(move(vertices), move(indices));
}

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

template <typename MaterialParamsType>
shared_ptr<Material> AssetLoader::makeMaterial(
        MaterialParamsType params)
{
    return state_->makeMaterial(params);
}

shared_ptr<Scene> AssetLoader::makeScene(
        const SceneDescription &desc)
{
    return state_->makeScene(desc);
}

shared_ptr<Scene> AssetLoader::loadScene(
        const string &scene_path)
{
    return state_->loadScene(scene_path);
}

void CommandStream::waitForFrame(uint32_t frame_id)
{
    VkFence fence = state_->getFence(frame_id);
    assert(fence != VK_NULL_HANDLE);
    waitForFenceInfinitely(state_->dev, fence);
    resetFence(state_->dev, fence);
}

CommandStream::CommandStream(Handle<CommandStreamState> &&state,
                             uint32_t render_width,
                             uint32_t render_height)
    : state_(move(state)),
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

uint32_t CommandStream::render(const std::vector<Environment> &elems)
{
    return  state_->render(elems);
}

template <typename PipelineType>
BatchRenderer::BatchRenderer(const RenderConfig &cfg,
                             const RenderFeatures<PipelineType> &features)
    : BatchRenderer(
            make_handle<VulkanState>(cfg, features, 
                                     getUUIDFromCudaID(cfg.gpuID)))
{}

BatchRenderer::BatchRenderer(Handle<VulkanState> &&vk_state)
    : state_(move(vk_state))
{}

AssetLoader BatchRenderer::makeLoader()
{
    return AssetLoader(make_handle<LoaderState>(
            state_->makeLoader()));
}

CommandStream BatchRenderer::makeCommandStream()
{
    auto stream_state = make_handle<CommandStreamState>(state_->makeStream());

    glm::u32vec2 img_dim = state_->getImageDimensions();

    return CommandStream(move(stream_state),
                         img_dim.x, img_dim.y);
}

Environment::Environment(Handle<EnvironmentState> &&state)
    : state_(move(state)),
      view_(),
      index_map_(state_->scene->envDefaults.indexMap),
      transforms_(state_->scene->envDefaults.transforms),
      materials_(state_->scene->envDefaults.materials)
{}

uint32_t Environment::addInstance(uint32_t model_idx, uint32_t material_idx,
                                  const glm::mat4x3 &model_matrix)
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

    state_->lightReverseIDs.push_back(light_idx);
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

#include "v4r_instantiations.inl"
