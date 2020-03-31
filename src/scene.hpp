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
    const StreamSceneState &getStreamState() const { return stream_state_; }

private:
    SceneID(std::list<Scene>::iterator scene, StreamSceneState &&stream_state)
        : scene_(scene),
          stream_state_(std::move(stream_state))
    {}

    std::list<Scene>::iterator scene_;
    StreamSceneState stream_state_;

    friend class SceneManager;
};

class SceneManager {
public:
    SceneManager();

    SceneID loadScene(const std::string &scene_path,
                      CommandStreamState &renderer_state);
    void dropScene(SceneID &&scene_id,
                   CommandStreamState &renderer_state);

private:
    std::mutex load_mutex_;
    std::list<Scene> scenes_;
    std::unordered_map<std::string, std::list<Scene>::iterator> scene_lookup_;
};

}

#endif
