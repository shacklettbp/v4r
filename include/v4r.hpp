#ifndef V4R_H_INCLUDED
#define V4R_H_INCLUDED

#include <memory>

namespace v4r {

struct VulkanState;
struct CommandStreamState;
class SceneID;

class RenderResult {
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
        RenderResult renderCamera();

    private:
        using StreamStateHandle = Handle<CommandStreamState>;

        CommandStream(StreamStateHandle &&state);
        StreamStateHandle state_;

        friend class RenderContext;
    };

    RenderContext(int gpu_id);
    ~RenderContext();

    SceneHandle loadScene(const std::string &file);
    void dropScene(SceneHandle &&handle);

    CommandStream makeCommandStream() const;

private:
    std::unique_ptr<VulkanState> state_;
};

}

#endif
