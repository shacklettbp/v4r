#ifndef V4R_ENVIRONMENT_HPP_INCLUDED
#define V4R_ENVIRONMENT_HPP_INCLUDED

#include <v4r/assets.hpp>
#include <v4r/fwd.hpp>
#include <v4r/utils.hpp>

#include <glm/glm.hpp>
#include <vector>

namespace v4r {

class Environment {
public:
    Environment(Environment &&) = default;

    // Instance transformations
    uint32_t addInstance(uint32_t model_idx, uint32_t material_idx,
                         const glm::mat4 &model_matrix);
    void deleteInstance(uint32_t inst_id);

    inline const glm::mat4 & getInstanceTransform(uint32_t inst_id) const;
    inline void updateInstanceTransform(uint32_t inst_id,
                                        const glm::mat4 &model_matrix);

    inline void setInstanceMaterial(uint32_t inst_id, uint32_t material_idx);

    // Camera transformations
    inline void setCameraView(const glm::vec3 &eye, const glm::vec3 &look,
                              const glm::vec3 &up);

    inline void setCameraView(const glm::mat4 &mat);

    inline void rotateCamera(float angle, const glm::vec3 &axis);

    inline void translateCamera(const glm::vec3 &v);

private:
    Environment(Handle<EnvironmentState> &&env);

    Handle<EnvironmentState> state_;
    glm::mat4 view_;
    std::vector<std::pair<uint32_t, uint32_t>> index_map_;
    std::vector<std::vector<glm::mat4>> transforms_;
    std::vector<std::vector<uint32_t>> materials_;

template <typename PipelineType>
friend class CommandStream;

friend class CommandStreamState;
};

}

#include <v4r/environment.inl>

#endif
