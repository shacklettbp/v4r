#ifndef V4R_ASSETS_HPP_INCLUDED
#define V4R_ASSETS_HPP_INCLUDED

#include <v4r/fwd.hpp>

#include <glm/glm.hpp>

#include <memory>
#include <vector>

namespace v4r {

struct InstanceProperties {
    uint32_t meshIndex;
    uint32_t materialIndex;
    glm::mat4x3 txfm;

    inline InstanceProperties(uint32_t mesh_idx, uint32_t mat_idx,
                              const glm::mat4x3 &txfm);
    inline InstanceProperties(uint32_t mesh_idx, uint32_t mat_idx,
                              const glm::mat4 &txfm);
};

struct LightProperties {
    glm::vec4 position;
    glm::vec4 color;
};

class SceneDescription {
public:
    inline SceneDescription(
            std::vector<std::shared_ptr<Mesh>> geometry,
            std::vector<std::shared_ptr<Material>> materials);

    inline uint32_t addInstance(uint32_t mesh_idx, uint32_t material_idx,
                                const glm::mat4x3 &txfm);

    inline uint32_t addInstance(uint32_t mesh_idx, uint32_t material_idx,
                                const glm::mat4 &txfm);

    inline uint32_t addLight(const glm::vec3 &position,
                             const glm::vec3 &color);

    inline const std::vector<std::shared_ptr<Mesh>> & getMeshes() const;
    inline const std::vector<std::shared_ptr<Material>> & getMaterials() const;
    inline const std::vector<InstanceProperties> & getDefaultInstances() const;
    inline const std::vector<LightProperties> & getDefaultLights() const;

private:
    std::vector<std::shared_ptr<Mesh>> meshes_;
    std::vector<std::shared_ptr<Material>> materials_;

    std::vector<InstanceProperties> default_instances_;

    std::vector<LightProperties> default_lights_;
};

}

// See comment in environment.hpp
#ifndef V4R_ASSETS_INL_INCLUDED
#include <v4r/assets.inl>
#endif

#endif
