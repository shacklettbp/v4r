#ifndef V4R_HPP_INCLUDED
#define V4R_HPP_INCLUDED

#include <v4r/assets.hpp>
#include <v4r/config.hpp>
#include <v4r/environment.hpp>
#include <v4r/fwd.hpp>
#include <v4r/pipelines.hpp>
#include <v4r/utils.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace v4r {

template <typename PipelineType>
class AssetLoader {
public:
    using VertexType = typename PipelineType::VertexType;
    using MeshType = Mesh<VertexType>;
    using MaterialDescType = typename PipelineType::MaterialDescType;
    using MaterialType = Material<MaterialDescType>;

    std::shared_ptr<MeshType> loadMesh(
            const std::string &geometry_path);

    std::shared_ptr<MeshType> loadMesh(
            std::vector<VertexType> vertices,
            std::vector<uint32_t> indices);

    std::shared_ptr<Texture> loadTexture(const std::string &texture_path);

    std::shared_ptr<MaterialType> makeMaterial(
            MaterialDescType params);

    std::shared_ptr<Scene> makeScene(
            const SceneDescription<PipelineType> &desc);

    // Shortcut for Gibson style scene files
    std::shared_ptr<Scene> loadScene(const std::string &scene_path);

private:
    AssetLoader(Handle<LoaderState> &&state);

    Handle<LoaderState> state_;

friend class BatchRenderer<PipelineType>;
};

class RenderSync {
public:
    void gpuWait(cudaStream_t strm);
    void cpuWait();

private:
    RenderSync(cudaExternalSemaphore_t sem);

    cudaExternalSemaphore_t ext_sem_;

template <typename PipelineType>
friend class CommandStream;
};

template <typename PipelineType>
class CommandStream {
public:
    Environment makeEnvironment(const std::shared_ptr<Scene> &scene,
                                float hfov, float near = 0.001f,
                                float far = 10000.f);
    // Render batch
    RenderSync render(const std::vector<Environment> &elems);

    // Fixed CUDA device pointers to result buffers
    uint8_t * getColorDevPtr() const;
    float * getDepthDevPtr() const;

private:
    CommandStream(Handle<CommandStreamState> &&state,
                  uint8_t *color_ptr,
                  float *depth_ptr,
                  uint32_t render_width,
                  uint32_t render_height);

    Handle<CommandStreamState> state_;
    Handle<CudaStreamState> cuda_;

    uint32_t render_width_;
    uint32_t render_height_;

friend class BatchRenderer<PipelineType>;
};

template <typename PipelineType>
class BatchRenderer {
public:
    BatchRenderer(const RenderConfig &cfg);

    using MeshType = Mesh<typename PipelineType::VertexType>;
    using MaterialType = Material<typename PipelineType::MaterialDescType>;
    using LoaderType = AssetLoader<PipelineType>;
    using CommandStreamType = CommandStream<PipelineType>;
    using SceneDescriptionType = SceneDescription<PipelineType>;

    LoaderType makeLoader();
    CommandStreamType makeCommandStream();

private:
    Handle<VulkanState> state_;
    Handle<CudaState> cuda_;
};

using UnlitBatchRenderer = BatchRenderer<UnlitPipeline>;

}

#endif
