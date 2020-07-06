#ifndef V4R_ENVIRONMENT_INL_INCLUDED
#define V4R_ENVIRONMENT_INL_INCLUDED

#include <v4r/environment.hpp>

#include <glm/gtc/matrix_transform.hpp>

namespace v4r {

const glm::mat4 & Environment::getInstanceTransform(uint32_t inst_id) const
{

    const auto &p = index_map_[inst_id];
    return transforms_[p.first][p.second];
}

void Environment::updateInstanceTransform(uint32_t inst_id,
                                          const glm::mat4 &mat)
{
    const auto &p = index_map_[inst_id];
    transforms_[p.first][p.second] = mat;
}

void Environment::setInstanceMaterial(uint32_t inst_id,
                                      uint32_t material_idx)
{
    const auto &p = index_map_[inst_id];
    materials_[p.first][p.second] = material_idx;
}

void Environment::setCameraView(const glm::vec3 &eye, const glm::vec3 &look,
                                const glm::vec3 &up)
{
    setCameraView(glm::lookAt(eye, look, up));
}

void Environment::setCameraView(const glm::mat4 &m)
{
    view_ = m;
}

void Environment::rotateCamera(float angle, const glm::vec3 &axis)
{
    view_ = glm::rotate(view_, angle, axis);
}

void Environment::translateCamera(const glm::vec3 &v)
{
    view_ = glm::translate(view_, v);
}

}

#endif
