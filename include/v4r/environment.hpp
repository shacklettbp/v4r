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

    uint32_t addLight(const glm::vec3 &position, const glm::vec3 &color);
    void deleteLight(uint32_t light_id);

private:
    Environment(Handle<EnvironmentState> &&env);

    Handle<EnvironmentState> state_;
    glm::mat4 view_;
    std::vector<std::pair<uint32_t, uint32_t>> index_map_;
    std::vector<std::vector<glm::mat4>> transforms_;
    std::vector<std::vector<uint32_t>> materials_;

friend class CommandStream;
friend class CommandStreamState;
};

}

// This include guard isn't necessary for compilation, but clang static
// analysis tools will complain about recursive inclusion otherwise when
// processing the inl file on its own, because the inl file needs to also
// include this file for Environment's definition
#ifndef V4R_ENVIRONMENT_INL_INCLUDED
#include <v4r/environment.inl>
#endif

#endif
