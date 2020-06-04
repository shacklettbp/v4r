#ifndef SCENE_HPP_INCLUDED
#define SCENE_HPP_INCLUDED

#include <list>
#include <mutex>
#include <unordered_map>

#include "vulkan_state.hpp"

namespace v4r {

class LoadedScene {
public:
    LoadedScene(const std::string &scene_path, LoaderState &loader_state,
          const glm::mat4 &coordinate_transform);

    const std::string & getPath() const { return path_; }
    void refIncrement() { ref_count_++; }
    bool refDecrement() { return !(--ref_count_); }

    const SceneState &getState() const { return state_; }

private:
    std::string path_;
    uint64_t ref_count_;

    SceneState state_;
};

class SceneManager;
class SceneID {
public:
    const SceneState &getState() const { return scene_->getState(); }

private:
    SceneID(std::list<LoadedScene>::iterator scene)
        : scene_(scene)
    {}

    std::list<LoadedScene>::iterator scene_;

    friend class SceneManager;
};

class SceneManager {
public:
    SceneManager(const glm::mat4 &coordinate_transform);

    SceneID loadScene(const std::string &scene_path,
                      LoaderState &loader_state);
    void dropScene(SceneID &&scene_id);

private:
    glm::mat4 coordinate_txfm_;
    std::mutex load_mutex_;
    std::list<LoadedScene> scenes_;
    std::unordered_map<std::string, std::list<LoadedScene>::iterator> scene_lookup_;
};

}

#endif
