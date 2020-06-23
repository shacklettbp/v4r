#ifndef ASSET_LOAD_HPP_INCLUDED
#define ASSET_LOAD_HPP_INCLUDED

#include "scene.hpp"

#include <v4r/assets.hpp>

#include <assimp/scene.h>

namespace v4r {

std::shared_ptr<Texture> readSDRTexture(const uint8_t *input,
                                        size_t num_bytes);

template <typename MaterialParamType>
std::vector<MaterialParamType> assimpParseMaterials(const aiScene *scene);

template <typename VertexType>
std::pair<std::vector<VertexType>, std::vector<uint32_t>> assimpParseMesh(
        const aiMesh *mesh);

template <typename PipelineType>
void assimpParseInstances(SceneDescription<PipelineType> &desc,
        const aiScene *scene,
        const std::vector<uint32_t> &mesh_materials,
        const glm::mat4 &coordinate_txfm);

}

#include "asset_load.inl"

#endif
