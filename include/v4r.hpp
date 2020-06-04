#ifndef V4R_HPP_INCLUDED
#define V4R_HPP_INCLUDED

#include <v4r/config.hpp>
#include <v4r/utils.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace v4r {

struct VulkanState;
struct CommandStreamState;
struct LoaderState;
struct CameraState;
class SceneID;
class SceneManager;
class CudaState;
class Renderer;
class SceneLoader;

struct FrameBatch {
public:
    uint8_t *colorPtr;
    float *depthPtr;
};

class RenderFuture {
public:
private:
friend class CommandStream;
};

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

friend class Renderer;
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

    // Fixed pointers to output buffers
    FrameBatch getResultsPointer() const;

    // Render batch based on current state
    RenderFuture render();

private:
    CommandStream(Handle<CommandStreamState> &&state,
                  const CudaState &cuda,
                  uint32_t render_width,
                  uint32_t render_height,
                  uint32_t batch_size);

    Handle<CommandStreamState> state_;
    const CudaState &cuda_;

    struct RenderInput {
        glm::mat4 *view;
        glm::mat4 *instanceTransforms;
        uint32_t numInstances;
        bool dirty;
    };

    uint32_t render_width_;
    uint32_t render_height_;
    std::vector<RenderInput> cur_inputs_;

friend class Renderer;
};

class Renderer {
public:
    Renderer(const RenderConfig &cfg);
    ~Renderer();

    SceneLoader makeLoader();
    CommandStream makeCommandStream();

private:
    std::unique_ptr<VulkanState> state_;
    std::unique_ptr<SceneManager> scene_mgr_;
    std::unique_ptr<CudaState> cuda_;
};

}

#include <v4r/impl.inl>

#endif
