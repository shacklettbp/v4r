#include <v4r.hpp>

#include "asset_load.hpp"
#include "cuda_state.hpp"
#include "dispatch.hpp"
#include "vulkan_state.hpp"
#include "scene.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

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
template struct HandleDeleter<LoaderState>;
template struct HandleDeleter<CommandStreamState>;
template struct HandleDeleter<CudaStreamState>;
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

template <typename PipelineType>
AssetLoader<PipelineType>::AssetLoader(Handle<LoaderState> &&state)
    : state_(move(state))
{}

template <typename PipelineType>
shared_ptr<Mesh<typename PipelineType::VertexType>>
AssetLoader<PipelineType>::loadMesh(const string &geometry_path)
{
    Assimp::Importer importer;
    int flags = aiProcess_PreTransformVertices | aiProcess_Triangulate;
    const aiScene *raw_scene = importer.ReadFile(geometry_path.c_str(), flags);
    if (!raw_scene) {
        cerr << "Failed to load geometry file " << geometry_path << ": " <<
            importer.GetErrorString() << endl;
        fatalExit();
    }

    if (raw_scene->mNumMeshes == 0) {
        cerr << "No meshes in file " << geometry_path << endl;
        fatalExit();
    }

    // FIXME probably should just union all meshes into one mesh here
    aiMesh *raw_mesh = raw_scene->mMeshes[0];

    auto [vertices, indices] = assimpParseMesh<VertexType>(raw_mesh);

    return loadMesh(move(vertices), move(indices));
}

template <typename PipelineType>
shared_ptr<Mesh<typename PipelineType::VertexType>>
AssetLoader<PipelineType>::loadMesh(vector<VertexType> vertices,
                                    vector<uint32_t> indices)
{
    return make_shared<MeshType>(move(vertices), move(indices));
}

template <typename PipelineType>
shared_ptr<Texture> AssetLoader<PipelineType>::loadTexture(
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

    return readSDRTexture(raw.data(), raw.size());
}

template <typename PipelineType>
shared_ptr<Material<typename PipelineType::MaterialDescType>>
AssetLoader<PipelineType>::makeMaterial(
        typename PipelineType::MaterialDescType params)
{
    return make_shared<MaterialType>(MaterialType {
        move(params)
    });
}

template <typename PipelineType>
shared_ptr<Scene> AssetLoader<PipelineType>::makeScene(
        const SceneDescription<PipelineType> &desc)
{
    return state_->loadScene(desc);
}

template <typename PipelineType>
shared_ptr<Scene> AssetLoader<PipelineType>::loadScene(
        const string &scene_path)
{
    Assimp::Importer importer;
    int flags = aiProcess_PreTransformVertices | aiProcess_Triangulate;
    const aiScene *raw_scene = importer.ReadFile(scene_path.c_str(), flags);
    if (!raw_scene) {
        cerr << "Failed to load scene " << scene_path << ": " <<
            importer.GetErrorString() << endl;
        fatalExit();
    }

    vector<shared_ptr<MaterialType>> materials;
    vector<shared_ptr<MeshType>> geometry;
    vector<uint32_t> mesh_materials;

    auto material_params = assimpParseMaterials<MaterialDescType>(raw_scene);
    materials.reserve(material_params.size());
    for (auto &&params : material_params) {
        materials.emplace_back(makeMaterial(move(params)));
    }

    geometry.reserve(raw_scene->mNumMeshes);
    mesh_materials.reserve(raw_scene->mNumMeshes);
    for (uint32_t mesh_idx = 0; mesh_idx < raw_scene->mNumMeshes; mesh_idx++) {
        aiMesh *raw_mesh = raw_scene->mMeshes[mesh_idx];
        mesh_materials.push_back(raw_mesh->mMaterialIndex);

        auto [vertices, indices] = assimpParseMesh<VertexType>(raw_mesh);
        geometry.emplace_back(loadMesh(move(vertices), move(indices)));
    }

    SceneDescription<PipelineType> scene_desc(move(geometry), move(materials));

    assimpParseInstances<PipelineType>(scene_desc, raw_scene, mesh_materials,
                                       state_->coordinateTransform);

    return makeScene(scene_desc);
}

RenderSync::RenderSync(cudaExternalSemaphore_t sem)
    : ext_sem_(sem)
{}

void RenderSync::gpuWait(cudaStream_t strm)
{
    cudaGPUWait(ext_sem_, strm);
}

void RenderSync::cpuWait()
{
    cudaCPUWait(ext_sem_);
}

template <typename PipelineType>
CommandStream<PipelineType>::CommandStream(Handle<CommandStreamState> &&state,
                                           uint8_t *color_ptr,
                                           float *depth_ptr,
                                           uint32_t render_width,
                                           uint32_t render_height)
    : state_(move(state)),
      cuda_(make_handle<CudaStreamState>(color_ptr, depth_ptr,
                                         state_->getSemaphoreFD())),
      render_width_(render_width),
      render_height_(render_height)
{}

template <typename PipelineType>
Environment CommandStream<PipelineType>::makeEnvironment(
        const shared_ptr<Scene> &scene,
        float hfov, float near, float far)
{
    glm::mat4 perspective =
        makePerspectiveMatrix(hfov, render_width_, render_height_,
                              near, far);

    return Environment(make_handle<EnvironmentState>(scene, perspective));
}

template <typename PipelineType>
uint8_t * CommandStream<PipelineType>::getColorDevPtr() const
{
    return cuda_->getColor();
}

template <typename PipelineType>
float * CommandStream<PipelineType>::getDepthDevPtr() const
{
    return cuda_->getDepth();
}

template <typename PipelineType>
RenderSync CommandStream<PipelineType>::render(
        const std::vector<Environment> &elems)
{
    state_->render(elems);

    return RenderSync(cuda_->getSemaphore());
}

template <typename PipelineType>
BatchRenderer<PipelineType>::BatchRenderer(const RenderConfig &cfg)
    : state_(make_handle<VulkanState>(cfg, getUUIDFromCudaID(cfg.gpuID))),
      cuda_(make_handle<CudaState>(cfg.gpuID, state_->getFramebufferFD(),
                                   state_->getFramebufferBytes()))
{}

template <typename PipelineType>
AssetLoader<PipelineType> BatchRenderer<PipelineType>::makeLoader()
{
    return AssetLoader<PipelineType>(make_handle<LoaderState>(
            state_->makeLoader()));
}

template <typename PipelineType>
CommandStream<PipelineType> BatchRenderer<PipelineType>::makeCommandStream()
{
    auto stream_state = make_handle<CommandStreamState>(state_->makeStream());

    cuda_->setActiveDevice();
    auto color_ptr =
        (uint8_t *)cuda_->getPointer(stream_state->getColorOffset());
    auto depth_ptr =
        (float *)cuda_->getPointer(stream_state->getDepthOffset());

    return CommandStream<PipelineType>(move(stream_state),
            color_ptr, depth_ptr,
            state_->cfg.imgWidth, state_->cfg.imgHeight);
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

template class AssetLoader<UnlitPipeline>;
template class CommandStream<UnlitPipeline>;
template class BatchRenderer<UnlitPipeline>;

}
