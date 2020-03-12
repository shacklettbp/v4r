#ifndef V4R_H_INCLUDED
#define V4R_H_INCLUDED

#include <memory>

namespace v4r {

struct VulkanState;
class SceneID;

class RenderContext {
public:
    template <typename T>
    struct HandleDeleter {
        constexpr HandleDeleter() noexcept = default;
        void operator()(T *ptr) const;
    };

    template <typename T>
    using Handle = std::unique_ptr<T, HandleDeleter<T>>;

    RenderContext(int gpu_id);
    ~RenderContext();

    Handle<SceneID> loadScene(const std::string &file);
    void dropScene(Handle<SceneID> &&handle);

private:
    std::unique_ptr<VulkanState> state_;
};

}

#endif
