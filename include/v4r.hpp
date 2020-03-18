#ifndef V4R_H_INCLUDED
#define V4R_H_INCLUDED

#include <memory>

#include <v4r/config.hpp>

namespace v4r {

struct VulkanState;
struct CommandStreamState;
class SceneID;
class SceneManager;

struct RenderResult {
public:
    void *cudaDevicePtr;
};

class RenderContext {
public:
    template <typename T>
    struct HandleDeleter {
        constexpr HandleDeleter() noexcept = default;
        void operator()(T *ptr) const;
    };

    template <typename T>
    using Handle = std::unique_ptr<T, HandleDeleter<T>>;
    using SceneHandle = Handle<SceneID>;

    class CommandStream {
    public:
        RenderResult renderCamera(const SceneHandle &scene);

        SceneHandle loadScene(const std::string &file);
        void dropScene(SceneHandle &&handle);

    private:
        using StreamStateHandle = Handle<CommandStreamState>;

        CommandStream(StreamStateHandle &&state, RenderContext &global);
        StreamStateHandle state_;
        RenderContext &global_;

        friend class RenderContext;
    };

    RenderContext(const RenderConfig &cfg);
    ~RenderContext();

    CommandStream makeCommandStream();

private:
    std::unique_ptr<VulkanState> state_;
    std::unique_ptr<SceneManager> scene_mgr_;
};

}

#endif
