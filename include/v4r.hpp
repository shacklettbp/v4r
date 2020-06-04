#ifndef V4R_HPP_INCLUDED
#define V4R_HPP_INCLUDED

#include <v4r/config.hpp>
#include <v4r/fwd.hpp>
#include <v4r/utils.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace v4r {

using SceneHandle = Handle<SceneID>;

class SceneLoader {
public:
    SceneHandle loadScene(const std::string &file);
    void dropScene(SceneHandle &&scene);

private:
    SceneLoader(Handle<LoaderState> &&state,
                SceneManager &mgr);

    Handle<LoaderState> state_;
    SceneManager &mgr_;

friend class BatchRenderer;
};

class RenderSync {
public:
    void gpuWait(cudaStream_t strm);
    void cpuWait();

private:
    RenderSync(cudaExternalSemaphore_t sem);

    cudaExternalSemaphore_t ext_sem_;
friend class CommandStream;
};

class CommandStream {
public:
    // Initialize state to render 'scene' into the frame at 'batch_idx'.
    // Can be called repeatedly with the same 'batch_idx' to switch scenes
    void initState(uint32_t batch_idx, const SceneHandle &scene, float hfov,
                   float near = 0.001f, float far = 10000.f);

    // Object Instance transformations
    inline const glm::mat4 & getInstanceTransform(uint32_t batch_idx,
                                                  uint32_t inst_idx) const;

    inline void updateInstanceTransform(uint32_t batch_idx,
                                        uint32_t inst_idx,
                                        const glm::mat4 &mat);

    inline uint32_t numInstanceTransforms(uint32_t batch_idx) const;

    // Camera transformations
    inline void setCameraView(uint32_t batch_idx, const glm::vec3 &eye,
                              const glm::vec3 &look, const glm::vec3 &up);

    inline void setCameraView(uint32_t batch_idx, const glm::mat4 &mat);

    inline void rotateCamera(uint32_t batch_idx, float angle,
                             const glm::vec3 &axis);

    inline void translateCamera(uint32_t batch_idx, const glm::vec3 &v);

    // Fixed CUDA device pointers to result buffers
    uint8_t * getColorDevPtr() const;
    float * getDepthDevPtr() const;

    // Render batch based on current state
    RenderSync render();

private:
    CommandStream(Handle<CommandStreamState> &&state,
                  const CudaState &renderer_cuda,
                  uint32_t render_width,
                  uint32_t render_height,
                  uint32_t batch_size);

    Handle<CommandStreamState> state_;
    Handle<CudaStreamState> cuda_;

    struct RenderInput {
        glm::mat4 *view;
        glm::mat4 *instanceTransforms;
        uint32_t numInstances;
        bool dirty;
    };

    uint32_t render_width_;
    uint32_t render_height_;
    std::vector<RenderInput> cur_inputs_;

friend class BatchRenderer;
};

class BatchRenderer {
public:
    BatchRenderer(const RenderConfig &cfg);

    SceneLoader makeLoader();
    CommandStream makeCommandStream();

private:
    Handle<VulkanState> state_;
    Handle<SceneManager> scene_mgr_;
    Handle<CudaState> cuda_;
};

}

#include <v4r/impl.inl>

#endif
