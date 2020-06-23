#ifndef V4R_ASSETS_HPP_INCLUDED
#define V4R_ASSETS_HPP_INCLUDED

#include <v4r/fwd.hpp>

#include <glm/glm.hpp>

#include <memory>
#include <vector>

namespace v4r {

struct UnlitVertex {
    glm::vec3 position;
    glm::vec2 uv;
};

struct ColoredVertex {
    glm::vec3 position;
    glm::u8vec3 color;
};

struct UnlitMaterialDescription {
    std::shared_ptr<Texture> texture;
};

struct InstanceProperties {
    glm::mat4 modelTransform;
    uint32_t materialIndex;

    inline InstanceProperties(const glm::mat4 &model_txfm, uint32_t mat_idx);
};

template <typename PipelineType>
class SceneDescription {
public:
    using MeshType = Mesh<typename PipelineType::VertexType>;
    using MaterialType = Material<typename PipelineType::MaterialDescType>;

    SceneDescription(
            std::vector<std::shared_ptr<MeshType>> geometry,
            std::vector<std::shared_ptr<MaterialType>> materials);

    inline void addInstance(uint32_t model_idx, uint32_t material_idx,
                            const glm::mat4 &model_transform);

    const std::vector<std::shared_ptr<MeshType>> & getMeshes() const;
    const std::vector<std::shared_ptr<MaterialType>> & getMaterials() const;
    const std::vector<std::vector<InstanceProperties>> &
        getDefaultInstances() const;

private:
    std::vector<std::shared_ptr<MeshType>> meshes_;
    std::vector<std::shared_ptr<MaterialType>> materials_;

    std::vector<std::vector<InstanceProperties>>
        default_instances_;
};

}

#include <v4r/assets.inl>

#endif
