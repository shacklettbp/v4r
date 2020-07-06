#ifndef V4R_ASSETS_HPP_INCLUDED
#define V4R_ASSETS_HPP_INCLUDED

#include <v4r/fwd.hpp>

#include <glm/glm.hpp>

#include <memory>
#include <vector>

namespace v4r {

struct UnlitRendererInputs {
    struct NoColorVertex {
        glm::vec3 position;
    };

    struct ColoredVertex {
        glm::vec3 position;
        glm::u8vec3 color;
    };
    
    struct TexturedVertex {
        glm::vec3 position;
        glm::vec2 uv;
    };

    struct MaterialDescription {
        std::shared_ptr<Texture> texture;
    };
};

struct LitRendererInputs {
    struct NoColorVertex {
        glm::vec3 position;
        glm::vec3 normal;
    };
    
    struct TexturedVertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
    };

    struct MaterialDescription {
        std::shared_ptr<Texture> texture;
    };
};

struct InstanceProperties {
    glm::mat4 modelTransform;
    uint32_t materialIndex;

    inline InstanceProperties(const glm::mat4 &model_txfm, uint32_t mat_idx);
};

class SceneDescription {
public:
    inline SceneDescription(
            std::vector<std::shared_ptr<Mesh>> geometry,
            std::vector<std::shared_ptr<Material>> materials);

    inline void addInstance(uint32_t model_idx, uint32_t material_idx,
                            const glm::mat4 &model_transform);

    inline const std::vector<std::shared_ptr<Mesh>> & getMeshes() const;
    inline const std::vector<std::shared_ptr<Material>> & getMaterials() const;
    inline const std::vector<std::vector<InstanceProperties>> &
        getDefaultInstances() const;

private:
    std::vector<std::shared_ptr<Mesh>> meshes_;
    std::vector<std::shared_ptr<Material>> materials_;

    std::vector<std::vector<InstanceProperties>>
        default_instances_;
};

}

// See comment in environment.hpp
#ifndef V4R_ASSETS_INL_INCLUDED
#include <v4r/assets.inl>
#endif

#endif
