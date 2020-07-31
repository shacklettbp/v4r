#ifndef V4R_HPP_INCLUDED
#define V4R_HPP_INCLUDED

#include <v4r/assets.hpp>
#include <v4r/config.hpp>
#include <v4r/environment.hpp>
#include <v4r/fwd.hpp>
#include <v4r/utils.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace v4r {

class AssetLoader {
public:
    std::shared_ptr<Mesh> loadMesh(
            const std::string &geometry_path);

    template <typename VertexType>
    std::shared_ptr<Mesh> loadMesh(
            std::vector<VertexType> vertices,
            std::vector<uint32_t> indices);

    std::shared_ptr<Texture> loadTexture(const std::string &texture_path);

    template <typename MaterialParamsType>
    std::shared_ptr<Material> makeMaterial(
            MaterialParamsType params);

    std::shared_ptr<Scene> makeScene(
            const SceneDescription &desc);

    // Shortcut for Gibson style scene files
    std::shared_ptr<Scene> loadScene(const std::string &scene_path);

private:
    AssetLoader(Handle<LoaderState> &&state);

    Handle<LoaderState> state_;

friend class BatchRenderer;
};

class RenderSync {
public:
    RenderSync(const SyncState *state);
    RenderSync(RenderSync &&) = default;
    RenderSync & operator=(RenderSync &&) = default;

    void gpuWait(cudaStream_t strm);
    void cpuWait();

private:
    const SyncState *state_;
};

class CommandStream {
public:
    Environment makeEnvironment(const std::shared_ptr<Scene> &scene,
                                float hfov, float near = 0.001f,
                                float far = 10000.f);
    // Render batch
    RenderSync render(const std::vector<Environment> &elems);

    // Fixed CUDA device pointers to result buffers
    uint8_t * getColorDevPtr(bool alternate_buffer = false) const;
    float * getDepthDevPtr(bool alternate_buffer = false) const;

protected:
    CommandStream(Handle<CommandStreamState> &&state,
                  const CudaState &cuda_global,
                  uint32_t render_width,
                  uint32_t render_height,
                  bool double_buffered);

    Handle<CommandStreamState> state_;
    Handle<CudaStreamState[]> cuda_;
    Handle<SyncState[]> sync_;

private:
    uint32_t render_width_;
    uint32_t render_height_;

friend class BatchRenderer;
};

class BatchRenderer {
public:
    template <typename PipelineType>
    BatchRenderer(const RenderConfig &cfg,
                  const RenderFeatures<PipelineType> &features);

    AssetLoader makeLoader();
    CommandStream makeCommandStream();

protected:
    BatchRenderer(Handle<VulkanState> &&vk_state, int gpu_id);
    Handle<VulkanState> state_;

private:
    Handle<CudaState> cuda_;
};

}

#endif
