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
      default_instances_(),
      default_lights_()
{}

uint32_t SceneDescription::addInstance(
        uint32_t model_idx,
        uint32_t material_idx,
        const glm::mat4 &model_transform)
{
    default_instances_.emplace_back(model_idx,
                                    InstanceProperties {
                                        model_transform,
                                        material_idx });

    return default_instances_.size() - 1;
}

uint32_t SceneDescription::addLight(const glm::vec3 &position,
                                    const glm::vec3 &color)
{
    default_lights_.push_back({
        glm::vec4(position, 1.f),
        glm::vec4(color, 1.f)
    });

    return default_lights_.size() - 1;
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

const std::vector<std::pair<uint32_t, InstanceProperties>> &
SceneDescription::getDefaultInstances() const
{
    return default_instances_;
}

const std::vector<LightProperties> & SceneDescription::getDefaultLights() const
{
    return default_lights_;
}

}

#endif
