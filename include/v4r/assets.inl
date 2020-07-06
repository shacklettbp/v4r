#ifndef V4R_ASSETS_INL_INCLUDED
#define V4R_ASSETS_INL_INCLUDED

#include <v4r/assets.hpp>

namespace v4r {

InstanceProperties::InstanceProperties(const glm::mat4 &model_txfm, uint32_t mat_idx)
    : modelTransform(model_txfm),
      materialIndex(mat_idx)
{}

SceneDescription::SceneDescription(
        std::vector<std::shared_ptr<Mesh>> meshes,
        std::vector<std::shared_ptr<Material>> materials)
    : meshes_(move(meshes)),
      materials_(move(materials)),
      default_instances_(meshes_.size())
{}

void SceneDescription::addInstance(uint32_t model_idx,
        uint32_t material_idx,
        const glm::mat4 &model_transform)
{
    default_instances_[model_idx].emplace_back(model_transform, material_idx);
}

const std::vector<std::shared_ptr<Mesh>> &
SceneDescription::getMeshes() const
{
    return meshes_;
}

const std::vector<std::shared_ptr<Material>> &
SceneDescription::getMaterials() const
{
    return materials_;
}

const std::vector<std::vector<InstanceProperties>> &
SceneDescription::getDefaultInstances() const
{
    return default_instances_;
}

}

#endif
