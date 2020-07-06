#ifndef ASSET_LOAD_INL_INCLUDED
#define ASSET_LOAD_INL_INCLUDED

#include "asset_load.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#undef STB_IMAGE_IMPLEMENTATION

#include <cassert>
#include <iostream>
#include <type_traits>
#include <vector>

namespace v4r {

std::shared_ptr<Texture> readSDRTexture(const uint8_t *raw_input,
                                        size_t num_bytes)
{
    if (stbi_is_hdr_from_memory(raw_input, num_bytes)) {
        std::cerr << "Trying to read HDR texture as SDR" << std::endl;
        fatalExit();
    }

    int width, height, num_channels;
    uint8_t *texture_data =
        stbi_load_from_memory(raw_input, num_bytes,
                              &width, &height, &num_channels, 4);

    if (texture_data == nullptr) {
        std::cerr << "Failed to load texture" << std::endl;
        fatalExit();
    }

    return std::make_shared<Texture>(Texture {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        4,
        ManagedArray<uint8_t>(texture_data, stbi_image_free)
    });
}

static const std::shared_ptr<Texture> assimpLoadTexture(
        const aiScene *raw_scene,
        const aiMaterial *raw_mat,
        aiTextureType type,
        std::unordered_map<std::string, std::shared_ptr<Texture>> &loaded)
{
    aiString tex_path;
    bool has_texture = raw_mat->Get(AI_MATKEY_TEXTURE(type, 0), tex_path) ==
        AI_SUCCESS;

    if (!has_texture) return nullptr;

    auto lookup = loaded.find(tex_path.C_Str());

    if (lookup != loaded.end()) return lookup->second;

    if (auto texture = raw_scene->GetEmbeddedTexture(tex_path.C_Str())) {
        if (texture->mHeight > 0) {
            std::cerr << "Uncompressed textures not supported" << std::endl;
            fatalExit();
        } else {
            const uint8_t *raw_input =
                reinterpret_cast<const uint8_t *>(texture->pcData);

            auto loaded_texture = readSDRTexture(raw_input, texture->mWidth);

            loaded.emplace(tex_path.C_Str(), loaded_texture);

            return loaded_texture;
        }
    } else {
        std::cerr << "External textures not supported yet" << std::endl;
        fatalExit();
    }
}

template <typename MaterialParamsType>
std::vector<MaterialParamsType> assimpParseMaterials(
        const aiScene *raw_scene)
{
    std::vector<MaterialParamsType> materials;
    
    std::unordered_map<std::string, std::shared_ptr<Texture>> loaded;

    for (uint32_t mat_idx = 0; mat_idx < raw_scene->mNumMaterials; mat_idx++) {
        const aiMaterial *raw_mat = raw_scene->mMaterials[mat_idx];

        auto ambient_tex = assimpLoadTexture(raw_scene, raw_mat,
                                             aiTextureType_AMBIENT,
                                             loaded);

        glm::vec4 ambient_color {};
        if (!ambient_tex) {
            aiColor4D color;
            raw_mat->Get(AI_MATKEY_COLOR_AMBIENT, color);
            ambient_color = glm::vec4(color.r, color.g, color.b, color.a);
        }

        auto diffuse_tex = assimpLoadTexture(raw_scene, raw_mat,
                                             aiTextureType_DIFFUSE,
                                             loaded);

        glm::vec4 diffuse_color {};
        if (!diffuse_tex) {
            aiColor4D color;
            raw_mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
            diffuse_color = glm::vec4(color.r, color.g, color.b, color.a);
        }

        auto specular_tex = assimpLoadTexture(raw_scene, raw_mat,
                                              aiTextureType_SPECULAR,
                                              loaded);
        glm::vec4 specular_color {};
        if (!specular_tex) {
            aiColor4D color;
            raw_mat->Get(AI_MATKEY_COLOR_SPECULAR, color);
            specular_color = glm::vec4(color.r, color.g, color.b, color.a);
        }

        float shininess = 0.f;
        raw_mat->Get(AI_MATKEY_SHININESS, shininess);

        if constexpr (std::is_same_v<MaterialParamsType, 
                UnlitRendererInputs::MaterialDescription>) {
            if (diffuse_tex) {
                materials.push_back({
                    diffuse_tex
                });
            }
        } else {
            std::cerr << "No assimp load support for pipeline type" << 
                std::endl;
            fatalExit();
        }
    }

    return materials;
}

template <typename VertexType>
static VertexType assimpParseVertex(const aiMesh *raw_mesh, uint32_t vert_idx);

static glm::vec3 readPosition(const aiMesh *raw_mesh, uint32_t vert_idx)
{
    return glm::make_vec3(&raw_mesh->mVertices[vert_idx].x);
}

static glm::vec3 readColor(const aiMesh *raw_mesh, uint32_t vert_idx)
{
    if (raw_mesh->HasVertexColors(0)) {
        const auto raw_color = &raw_mesh->mColors[0][vert_idx];
        return glm::u8vec3(raw_color->r * 255, raw_color->g * 255,
                           raw_color->b * 255);
    } else {
        return glm::u8vec3(255);
    }
}

static glm::vec2 readUV(const aiMesh *raw_mesh, uint32_t vert_idx)
{
    if (raw_mesh->HasTextureCoords(0)) {
        const auto &raw_uv = raw_mesh->mTextureCoords[0][vert_idx];
        return glm::vec2(raw_uv.x, 1.f - raw_uv.y);
    } else {
        return glm::vec2(0.f);
    }
}

static glm::vec3 readNormal(const aiMesh *raw_mesh, uint32_t vert_idx)
{
    if (raw_mesh->HasNormals()) {
        return glm::make_vec3(&raw_mesh->mNormals[vert_idx].x);
    } else {
        return glm::vec3(1.f, 0.f, 0.f);
    }
}

template <>
UnlitRendererInputs::NoColorVertex assimpParseVertex(
        const aiMesh *raw_mesh, uint32_t vert_idx)
{
    return {
        readPosition(raw_mesh, vert_idx)
    };
}

template <>
UnlitRendererInputs::ColoredVertex assimpParseVertex(
        const aiMesh *raw_mesh, uint32_t vert_idx)
{
    return {
        readPosition(raw_mesh, vert_idx),
        readColor(raw_mesh, vert_idx)
    };
}

template <>
UnlitRendererInputs::TexturedVertex assimpParseVertex(
        const aiMesh *raw_mesh, uint32_t vert_idx)
{
    return {
        readPosition(raw_mesh, vert_idx),
        readUV(raw_mesh, vert_idx)
    };
}

template <>
LitRendererInputs::NoColorVertex assimpParseVertex(
        const aiMesh *raw_mesh, uint32_t vert_idx)
{
    return {
        readPosition(raw_mesh, vert_idx),
        readNormal(raw_mesh, vert_idx)
    };
}

template <>
LitRendererInputs::TexturedVertex assimpParseVertex(
        const aiMesh *raw_mesh, uint32_t vert_idx)
{
    return {
        readPosition(raw_mesh, vert_idx),
        readNormal(raw_mesh, vert_idx),
        readUV(raw_mesh, vert_idx)
    };
}

template <typename VertexType>
std::pair<std::vector<VertexType>, std::vector<uint32_t>> assimpParseMesh(
        const aiMesh *raw_mesh)
{
    std::vector<VertexType> vertices;
    std::vector<uint32_t> indices;

    for (uint32_t vert_idx = 0; vert_idx < raw_mesh->mNumVertices;
            vert_idx++) {
        vertices.emplace_back(assimpParseVertex<VertexType>(
            raw_mesh, vert_idx));
    }

    for (uint32_t face_idx = 0; face_idx < raw_mesh->mNumFaces;
            face_idx++) {
        for (uint32_t tri_idx = 0; tri_idx < 3; tri_idx++) {
            indices.push_back(raw_mesh->mFaces[face_idx].mIndices[tri_idx]);
        }
    }

    return make_pair(vertices, indices);
}

void assimpParseInstances(SceneDescription &desc,
        const aiScene *raw_scene,
        const std::vector<uint32_t> &mesh_materials,
        const glm::mat4 &coordinate_txfm)
{
    std::vector<std::pair<aiNode *, glm::mat4>> node_stack {
        { raw_scene->mRootNode, coordinate_txfm }
    };

    while (!node_stack.empty()) {
        auto [cur_node, parent_txfm] = node_stack.back();
        node_stack.pop_back();
        auto raw_txfm = cur_node->mTransformation;
        glm::mat4 cur_txfm = parent_txfm *
            glm::transpose(
                glm::make_mat4(reinterpret_cast<const float *>(&raw_txfm.a1)));

        if (cur_node->mNumChildren == 0) {
            if (cur_node->mNumMeshes != 1) {
                std::cerr <<
"Assimp loading: only leaf nodes with a single mesh are supported" <<
                    std::endl;
                fatalExit();
            }

            uint32_t mesh_idx = cur_node->mMeshes[0];

            desc.addInstance(mesh_idx, mesh_materials[mesh_idx], cur_txfm);
        } else {
            for (unsigned child_idx = 0; child_idx < cur_node->mNumChildren;
                    child_idx++) {
                node_stack.emplace_back(cur_node->mChildren[child_idx],
                                        cur_txfm);
            }
        }
    }
}

}

#endif
