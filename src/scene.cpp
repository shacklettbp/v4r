#include "scene.hpp"

#include <cassert>

#include "utils.hpp"

using namespace std;

namespace v4r {

Scene::Scene(const std::string &scene_path)
    : path_(scene_path),
      ref_count_(1)
{
}

Scene::~Scene()
{
}

SceneManager::SceneManager()
    : load_mutex_(),
      scenes_(),
      scene_lookup_()
{}

SceneID SceneManager::loadScene(const std::string &scene_path)
{
    scoped_lock lock(load_mutex_);

    auto lookup_iter = scene_lookup_.find(scene_path);
    if (lookup_iter != scene_lookup_.end()) {
        SceneID id = lookup_iter->second;
        id.iter_->refIncrement();

        return id;
    } 

    scenes_.emplace_front(scene_path);
    auto scene_iter = scenes_.begin();

    SceneID id { scene_iter };

    scene_lookup_.emplace(scene_path, id);

    return id;
}

void SceneManager::dropScene(SceneID &&scene_id)
{
    scoped_lock lock(load_mutex_);
    
    bool should_free = scene_id.iter_->refDecrement();
    if (!should_free) return;

    [[maybe_unused]] size_t num_erased =
        scene_lookup_.erase(scene_id.iter_->getPath());

    assert(num_erased == 1);

    scenes_.erase(scene_id.iter_);
}

}
