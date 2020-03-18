#ifndef SCENE_HPP_INCLUDED
#define SCENE_HPP_INCLUDED

#include <list>
#include <mutex>
#include <unordered_map>

#include "vulkan_state.hpp"

namespace v4r {

class Scene {
public:
    Scene(const std::string &scene_path, CommandStreamState &renderer_state);

    const std::string & getPath() const { return path_; }
    void refIncrement() { ref_count_++; }
    bool refDecrement() { return !(--ref_count_); }

private:
    std::string path_;
    uint64_t ref_count_;

    SceneState state_;
};

class SceneManager;
class SceneID {
private:
    SceneID(const std::list<Scene>::iterator &iter)
        : iter_(iter)
    {}

    std::list<Scene>::iterator iter_;

    friend class SceneManager;
};

class SceneManager {
public:
    SceneManager();

    SceneID loadScene(const std::string &scene_path,
                      CommandStreamState &renderer_state);
    void dropScene(SceneID &&scene_id);

private:
    std::mutex load_mutex_;
    std::list<Scene> scenes_;
    std::unordered_map<std::string, SceneID> scene_lookup_;
};

}

#endif
