#ifndef ASSET_LOAD_INL_INCLUDED
#define ASSET_LOAD_INL_INCLUDED

#include "asset_load.hpp"
#include "scene.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <basisu_transcoder.h>

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
#include <fstream>

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

static void basisImageFree(void *ptr)
{
    delete[] reinterpret_cast<uint8_t *>(ptr);
}

std::shared_ptr<Texture> readBasisTexture(const uint8_t *raw_input,
                                          size_t num_bytes)
{
    using namespace basist;
    static basist::etc1_global_selector_codebook codebook;

    basisu_transcoder transcoder(&codebook);

    bool valid_basis = transcoder.validate_header(raw_input, num_bytes);
    if (!valid_basis) {
        std::cerr << "Invalid basis texture" << std::endl;
        fatalExit();
    }

    basisu_file_info file_info;
    bool file_info_success =
        transcoder.get_file_info(raw_input, num_bytes, file_info);

    if (!file_info_success) {
        std::cerr << "Invalid basis texture: file info failed" << std::endl;
        fatalExit();
    }

    if (file_info.m_total_images != 1) {
        std::cerr << "Invalid basis texture: only single images supported"
                  << std::endl;
        fatalExit();
    }


    uint32_t width, height, total_blocks;
    transcoder.get_image_level_desc(raw_input, num_bytes, 0, 0, 
                                    width, height, total_blocks);

    uint32_t num_pixels = width * height;
    uint8_t *transcoded = new uint8_t[num_pixels * 4];

    transcoder.start_transcoding(raw_input, num_bytes);

    bool transcode_success = transcoder.transcode_image_level(
            raw_input, num_bytes, 0, 0, transcoded, num_pixels,
            transcoder_texture_format::cTFRGBA32, 0, width,
            nullptr, height);

    if (!transcode_success) {
        std::cerr << "Basis transcode failed" << std::endl;
        fatalExit();
    }

    transcoder.stop_transcoding();

    // FIXME really should do this flip on the GPU or something
    if (file_info.m_y_flipped) {
        for (uint32_t row_idx = 0; row_idx < height / 2; row_idx++) {
            uint32_t orig_row_idx = height - row_idx - 1;
            for (uint32_t col_idx = 0; col_idx < width; col_idx++) {
                for (uint32_t comp_idx = 0; comp_idx < 4; comp_idx++) {
                    uint32_t flip_idx =
                        (row_idx * width + col_idx) * 4 + comp_idx;
                    uint32_t orig_idx =
                        (orig_row_idx * width + col_idx) * 4 + comp_idx;

                    std::swap(transcoded[flip_idx], transcoded[orig_idx]);
                }
            }
        }
    }

    return std::make_shared<Texture>(Texture {
        width,
        height,
        4,
        ManagedArray<uint8_t>(transcoded, basisImageFree)
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
std::vector<std::shared_ptr<Material>> assimpParseMaterials(
        const aiScene *raw_scene,
        const std::shared_ptr<Texture> &default_diffuse)
{
    std::vector<std::shared_ptr<Material>> materials;
    
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

        if (!diffuse_tex) {
            diffuse_tex = default_diffuse;
        }

        if (!specular_tex) {
            specular_tex = default_diffuse;
        }

        materials.emplace_back(MaterialImpl<MaterialParamsType>::make(
                MaterialParam::DiffuseColorTexture { move(diffuse_tex) },
                MaterialParam::DiffuseColorUniform { diffuse_color },
                MaterialParam::SpecularColorTexture { move(specular_tex) },
                MaterialParam::SpecularColorUniform { specular_color },
                MaterialParam::ShininessUniform { shininess }));
    }

    return materials;
}

static glm::vec3 readPosition(const aiMesh *raw_mesh, uint32_t vert_idx)
{
    return glm::make_vec3(&raw_mesh->mVertices[vert_idx].x);
}

static glm::u8vec3 readColor(const aiMesh *raw_mesh, uint32_t vert_idx)
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

template <typename VertexType>
static VertexType assimpParseVertex(const aiMesh *raw_mesh,
                                    uint32_t vert_idx) {

    auto getPosition = [&]() {
        if constexpr (VertexImpl<VertexType>::hasPosition) {
            return std::make_tuple(
                readPosition(raw_mesh, vert_idx)
            );
        } else {
            return std::tuple<>();
        }
    };

    auto getNormal = [&]() {
        if constexpr (VertexImpl<VertexType>::hasNormal) {
            return std::make_tuple(
                    readNormal(raw_mesh, vert_idx)
            );
        } else {
            return std::tuple<>();
        }
    };

    auto getColor = [&]() {
        if constexpr (VertexImpl<VertexType>::hasColor) {
            return std::make_tuple(
                readColor(raw_mesh, vert_idx)
            );
        } else {
            return std::tuple<>();
        }
    };

    auto getUV = [&]() {
        if constexpr (VertexImpl<VertexType>::hasUV) {
            return std::make_tuple(
                readUV(raw_mesh, vert_idx)
            );
        } else {
            return std::tuple<>();
        }
    };

    return std::apply([] (auto&& ...args) {
        return VertexType {
            args
            ...
        };
    }, std::tuple_cat(getPosition(),
                      getNormal(),
                      getColor(),
                      getUV()));
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
            for (uint32_t i = 0; i < cur_node->mNumMeshes; i++) {
                uint32_t mesh_idx = cur_node->mMeshes[i];

                uint32_t material_idx = mesh_materials[mesh_idx];
                if (material_idx > raw_scene->mNumMaterials) {
                    material_idx = 0;
                }

                desc.addInstance(mesh_idx,
                                 mesh_materials.size() > 0 ?
                                     material_idx : 0,
                                 cur_txfm);
            }
        } else {
            for (unsigned child_idx = 0; child_idx < cur_node->mNumChildren;
                    child_idx++) {
                node_stack.emplace_back(cur_node->mChildren[child_idx],
                                        cur_txfm);
            }
        }
    }
}

struct GLBHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t length;
};

struct ChunkHeader {
    uint32_t chunkLength;
    uint32_t chunkType;
};

inline GLTFScene gltfLoad(const std::string_view gltf_path) noexcept
{
    GLTFScene scene;

    auto suffix = gltf_path.substr(gltf_path.find('.') + 1);
    bool binary = suffix == "glb";
    if (binary) {
        std::ifstream binary_file(std::string(gltf_path),
                                  std::ios::in | std::ios::binary);

        GLBHeader glb_header;
        binary_file.read(reinterpret_cast<char *>(&glb_header),
                         sizeof(GLBHeader));

        uint32_t total_length = glb_header.length;

        ChunkHeader json_header;
        binary_file.read(reinterpret_cast<char *>(&json_header),
                         sizeof(ChunkHeader));

        std::vector<uint8_t> json_buffer(
                json_header.chunkLength + simdjson::SIMDJSON_PADDING);

        binary_file.read(reinterpret_cast<char *>(json_buffer.data()),
                         json_header.chunkLength);

        try {
            scene.root = scene.jsonParser.parse(json_buffer.data(),
                                                json_header.chunkLength,
                                                false);
        } catch (const simdjson::simdjson_error &e) {
            std::cerr << "GLTF loadng failed: " << e.what() << std::endl;
            fatalExit();
        }

        if (json_header.chunkLength < total_length) {
            ChunkHeader bin_header;
            binary_file.read(reinterpret_cast<char *>(&bin_header),
                             sizeof(ChunkHeader));

            assert(bin_header.chunkType == 0x004E4942);

            scene.internalData.resize(bin_header.chunkLength);
                    
            binary_file.read(
                reinterpret_cast<char *>(scene.internalData.data()),
                bin_header.chunkLength);
        }
    } else {
        scene.root = scene.jsonParser.load(std::string(gltf_path));
    }

    try {
        for (const auto &buffer : scene.root["buffers"]) {
            std::string_view uri {};
            const uint8_t *data_ptr = nullptr;

            auto uri_elem = buffer.at_key("uri");
            if (uri_elem.error() != simdjson::NO_SUCH_FIELD) {
                uri = uri_elem.get_string();
            } else {
                data_ptr = scene.internalData.data();
            }
            scene.buffers.push_back(GLTFBuffer {
                data_ptr,
                uri,
            });
        }

        for (const auto &view : scene.root["bufferViews"]) {
            uint64_t stride_res;
            auto stride_error = view["byteStride"].get(stride_res);
            if (stride_error) {
                stride_res = 0;
            }
            scene.bufferViews.push_back(GLTFBufferView {
                static_cast<uint32_t>(view["buffer"].get_uint64()),
                static_cast<uint32_t>(view["byteOffset"].get_uint64()),
                static_cast<uint32_t>(stride_res),
                static_cast<uint32_t>(view["byteLength"].get_uint64()),
            });
        }

        for (const auto &accessor : scene.root["accessors"]) {
            GLTFComponentType type;
            uint64_t component_type = accessor["componentType"];
            if (component_type == 5126) {
                type = GLTFComponentType::FLOAT;
            } else if (component_type == 5125) {
                type = GLTFComponentType::UINT32;
            } else if (component_type == 5123) {
                type = GLTFComponentType::UINT16;
            } else {
                std::cerr << "GLTF loading failed: unknown component type"
                          << std::endl;
                fatalExit();
            }

            scene.accessors.push_back(GLTFAccessor {
                static_cast<uint32_t>(accessor["bufferView"].get_uint64()),
                static_cast<uint32_t>(accessor["byteOffset"].get_uint64()),
                static_cast<uint32_t>(accessor["count"].get_uint64()),
                type,
            });
        }

        for (const auto &json_image : scene.root["images"]) {
            GLTFImage img {};
            std::string_view uri {};
            auto uri_err = json_image["uri"].get(uri);
            if (!uri_err) {
                img.type = GLTFImageType::EXTERNAL;
                img.filePath = uri;
            } else {
                uint64_t view_idx = json_image["bufferView"];
                std::string_view mime = json_image["mimeType"];
                if (mime == "image/jpeg") {
                    img.type = GLTFImageType::JPEG;
                } else if (mime == "image/png") {
                    img.type = GLTFImageType::PNG;
                } else if (mime == "image/x-basis") {
                    img.type = GLTFImageType::BASIS;
                }

                img.viewIdx = view_idx;
            }

            scene.images.push_back(img);
        }

        for (const auto &texture : scene.root["textures"]) {
            uint64_t source_idx;
            auto src_err = texture["source"].get(source_idx);
            if (src_err) {
                auto ext_err = texture["extensions"]["GOOGLE_texture_basis"][
                    "source"].get(source_idx);
                if (ext_err) {
                    std::cerr << "GLTF loading failed: texture without source"
                              << std::endl;
                    fatalExit();
                }
            }

            scene.textures.push_back(GLTFTexture {
                static_cast<uint32_t>(source_idx),
                static_cast<uint32_t>(texture["sampler"].get_uint64()),
            });
        }

        for (const auto &material : scene.root["materials"]) {
            const auto &pbr = material["pbrMetallicRoughness"];
            simdjson::dom::element base_tex;
            // FIXME assumes tex coord 0
            uint64_t tex_idx;
            auto tex_err = pbr["baseColorTexture"]["index"].get(tex_idx);
            if (tex_err) {
                tex_idx = scene.textures.size();
            }

            glm::vec4 base_color(0.f);
            simdjson::dom::array base_color_json;
            auto color_err = pbr["baseColorFactor"].get(base_color_json);
            if (!color_err) {
                float *base_color_data = glm::value_ptr(base_color);
                for (double comp : base_color_json) {
                    *base_color_data = comp;
                    base_color_data++;
                }
            }

            double metallic;
            auto metallic_err = pbr["metallicFactor"].get(metallic);
            if (metallic_err) {
                metallic = 0;
            }

            double roughness;
            auto roughness_err = pbr["roughnessFactor"].get(roughness);
            if (roughness_err) {
                roughness = 1;
            }

            scene.materials.push_back(GLTFMaterial {
                static_cast<uint32_t>(tex_idx),
                base_color,
                static_cast<float>(metallic),
                static_cast<float>(roughness),
            });
        }

        for (const auto &mesh : scene.root["meshes"]) {
            simdjson::dom::array prims = mesh["primitives"];
            if (prims.size() != 1) {
                std::cerr << "GLTF loading failed: "
                          << "Only single primitive meshes supported"
                          << std::endl;
                fatalExit();
            }

            simdjson::dom::element prim = prims.at(0);

            simdjson::dom::element attrs = prim["attributes"];

            std::optional<uint32_t> position_idx;
            std::optional<uint32_t> normal_idx;
            std::optional<uint32_t> uv_idx;
            std::optional<uint32_t> color_idx;

            uint64_t position_res;
            auto position_error = attrs["POSITION"].get(position_res);
            if (!position_error) {
                position_idx = position_res;
            }

            uint64_t normal_res;
            auto normal_error = attrs["NORMAL"].get(normal_res);
            if (!normal_error) {
                normal_idx = normal_res;
            }

            uint64_t uv_res;
            auto uv_error = attrs["TEXCOORD_0"].get(uv_res);
            if (!uv_error) {
                uv_idx = uv_res;
            }

            uint64_t color_res;
            auto color_error = attrs["COLOR_0"].get(color_res);
            if (!color_error) {
                color_idx = color_res;
            }

            scene.meshes.push_back(GLTFMesh {
                position_idx,
                normal_idx,
                uv_idx,
                color_idx,
                static_cast<uint32_t>(prim["indices"].get_uint64()),
                static_cast<uint32_t>(prim["material"].get_uint64()),
            });
        }

        for (const auto &node : scene.root["nodes"]) {
            std::vector<uint32_t> children;
            simdjson::dom::array json_children;
            auto children_error = node["children"].get(json_children);

            if (!children_error) {
                for (uint64_t child : json_children) {
                    children.push_back(child);
                }
            }

            uint64_t mesh_idx;
            auto mesh_error = node["mesh"].get(mesh_idx);
            if (mesh_error) {
                mesh_idx = scene.meshes.size();
            }

            glm::mat4 txfm(1.f);

            simdjson::dom::array matrix;
            auto matrix_error = node["matrix"].get(matrix);
            if (!matrix_error) {
                float *txfm_data = glm::value_ptr(txfm);
                for (double mat_elem : matrix) {
                    *txfm_data = mat_elem;
                    txfm_data++;
                }
            }

            // FIXME TRS support

            scene.nodes.push_back(GLTFNode {
                move(children),
                static_cast<uint32_t>(mesh_idx),
                txfm
            });
        }

        simdjson::dom::array scenes = scene.root["scenes"];
        if (scenes.size() > 1) {
            std::cerr << "GLTF loading failed: Multiscene files not supported"
                      << std::endl;
            fatalExit();
        }

        for (uint64_t node_idx : scenes.at(0)["nodes"]) {
            scene.rootNodes.push_back(node_idx);
        }
        
    } catch (const simdjson::simdjson_error &e) {
        std::cerr << "GLTF loading failed: " << e.what() << std::endl;
        fatalExit();
    }

    return scene;
}

template <typename T>
static StridedSpan<T> getGLTFBufferView(const GLTFScene &scene,
                                        uint32_t view_idx,
                                        uint32_t start_offset = 0,
                                        uint32_t num_elems = 0)
{
    const GLTFBufferView &view = scene.bufferViews[view_idx];
    const GLTFBuffer &buffer = scene.buffers[view.bufferIdx];

    if (buffer.dataPtr == nullptr) {
        std::cerr << "GLTF loading failed: external references not supported"
                  << std::endl;
    }

    size_t total_offset = start_offset + view.offset;
    const uint8_t *start_ptr = buffer.dataPtr + total_offset;;

    uint32_t stride = view.stride;
    if (stride == 0) {
        stride = sizeof(T);
    }

    if (num_elems == 0) {
        num_elems = view.numBytes / stride;
    }

    return StridedSpan<T>(start_ptr, num_elems, stride);
}

template <typename T> 
static StridedSpan<T> getGLTFAccessorView(const GLTFScene &scene,
                                          uint32_t accessor_idx)
{
    const GLTFAccessor &accessor = scene.accessors[accessor_idx];

    return getGLTFBufferView<T>(scene, accessor.viewIdx,
                                accessor.offset,
                                accessor.numElems);
}

static std::shared_ptr<Texture> gltfLoadTexture(const GLTFScene &scene,
                                                uint32_t texture_idx)
{
    const GLTFImage &img = scene.images[scene.textures[texture_idx].sourceIdx];
    if (img.type == GLTFImageType::EXTERNAL) {
        std::cerr << "GLTF loading failed: External textures not supported"
                  << std::endl;
        fatalExit();
    }
    auto img_data = getGLTFBufferView<const uint8_t>(scene, img.viewIdx);
    if (!img_data.contiguous()) {
        std::cerr <<
            "GTLF loading failed: internal image needs to be contiguous"
                  << std::endl;
        fatalExit();
    }

    if (img.type == GLTFImageType::JPEG ||
        img.type == GLTFImageType::PNG) {
        return readSDRTexture(img_data.data(), img_data.size());
    } else if (img.type == GLTFImageType::BASIS) {
        return readBasisTexture(img_data.data(), img_data.size());
    } else {
        assert(false);
    }
}

template <typename MaterialParamsType>
std::vector<std::shared_ptr<Material>> gltfParseMaterials(
        const GLTFScene &scene,
        const std::shared_ptr<Texture> &default_diffuse)
{
    std::vector<std::shared_ptr<Material>> materials;
    std::vector<std::shared_ptr<Texture>> textures(scene.textures.size());

    for (const auto &gltf_mat : scene.materials) {
        std::shared_ptr<Texture> texture;
        if (gltf_mat.textureIdx < scene.textures.size()) {
            texture = textures[gltf_mat.textureIdx];
        } else {
            texture = default_diffuse;
        }

        if (texture == nullptr) {
            texture = gltfLoadTexture(scene, gltf_mat.textureIdx);
            textures[gltf_mat.textureIdx] = texture;
        }

        materials.emplace_back(MaterialImpl<MaterialParamsType>::make(
                MaterialParam::DiffuseColorTexture { move(texture) },
                MaterialParam::DiffuseColorUniform { 
                    glm::vec4(gltf_mat.baseColor, 1.f),
                },
                MaterialParam::SpecularColorTexture { nullptr },
                MaterialParam::SpecularColorUniform { glm::vec4() },
                MaterialParam::ShininessUniform { 1.f - gltf_mat.roughness }));
    }

    return materials;
}

template <typename VertexType>
std::pair<std::vector<VertexType>, std::vector<uint32_t>>
gltfParseMesh(const GLTFScene &scene, uint32_t mesh_idx)
{
    std::vector<VertexType> vertices;
    std::vector<uint32_t> indices;

    const GLTFMesh &mesh = scene.meshes[mesh_idx];
    std::optional<StridedSpan<const glm::vec3>> position_accessor;
    std::optional<StridedSpan<const glm::vec3>> normal_accessor;
    std::optional<StridedSpan<const glm::vec2>> uv_accessor;
    std::optional<StridedSpan<const glm::u8vec3>> color_accessor;

    if constexpr (VertexImpl<VertexType>::hasPosition) {
        position_accessor = 
            getGLTFAccessorView<const glm::vec3>(scene,
                                                 mesh.positionIdx.value());
    }

    if constexpr (VertexImpl<VertexType>::hasNormal) {
        normal_accessor = 
            getGLTFAccessorView<const glm::vec3>(scene,
                                                 mesh.normalIdx.value());
    }

    if constexpr (VertexImpl<VertexType>::hasUV) {
        uv_accessor = 
            getGLTFAccessorView<const glm::vec2>(scene, mesh.uvIdx.value());
    }

    if constexpr (VertexImpl<VertexType>::hasColor) {
        color_accessor = 
            getGLTFAccessorView<const glm::u8vec3>(scene,
                                                   mesh.colorIdx.value());
    }

    uint32_t max_idx = 0;

    auto index_type = scene.accessors[mesh.indicesIdx].type;

    if (index_type == GLTFComponentType::UINT32) {
        auto idx_accessor =
            getGLTFAccessorView<const uint32_t>(scene, mesh.indicesIdx);
        indices.reserve(idx_accessor.size());

        for (uint32_t idx : idx_accessor) {
            if (idx > max_idx) {
                max_idx = idx;
            }

            indices.push_back(idx);
        }
    } else if (index_type == GLTFComponentType::UINT16) {
        auto idx_accessor =
            getGLTFAccessorView<const uint16_t>(scene, mesh.indicesIdx);
        indices.reserve(idx_accessor.size());

        for (uint16_t idx : idx_accessor) {
            if (idx > max_idx) {
                max_idx = idx;
            }

            indices.push_back(idx);
        }
    } else {
        std::cerr << "GLTF loading failed: unsupported index type"
                  << std::endl;
        fatalExit();
    }

    assert(max_idx < position_accessor->size());

    vertices.reserve(max_idx + 1);
    for (uint32_t vert_idx = 0; vert_idx <= max_idx; vert_idx++) {
        VertexType vert;

        if constexpr (VertexImpl<VertexType>::hasPosition) {
            vert.position = (*position_accessor)[vert_idx];
        }

        if constexpr (VertexImpl<VertexType>::hasNormal) {
            vert.normal = (*normal_accessor)[vert_idx];
        }

        if constexpr (VertexImpl<VertexType>::hasUV) {
            vert.uv = (*uv_accessor)[vert_idx];
        }

        if constexpr (VertexImpl<VertexType>::hasColor) {
            vert.color = (*color_accessor)[vert_idx];
        }

        vertices.push_back(vert);
    }

    return { move(vertices), move(indices) };
}

inline void gltfParseInstances(SceneDescription &desc,
                        const GLTFScene &scene,
                        const glm::mat4 &coordinate_txfm)
{
    std::vector<std::pair<uint32_t, glm::mat4>> node_stack;
    for (uint32_t root_node : scene.rootNodes) {
        node_stack.emplace_back(root_node, coordinate_txfm);
    }

    while (!node_stack.empty()) {
        auto [node_idx, parent_txfm] = node_stack.back();
        node_stack.pop_back();

        const GLTFNode &cur_node = scene.nodes[node_idx];
        glm::mat4 cur_txfm = parent_txfm * cur_node.transform;

        for (const uint32_t child_idx : cur_node.children) {
            node_stack.emplace_back(child_idx, cur_txfm);
        }

        if (cur_node.meshIdx < scene.meshes.size()) {
            desc.addInstance(cur_node.meshIdx,
                             scene.meshes[cur_node.meshIdx].materialIdx,
                             cur_txfm);
        }
    }
}

}

#endif
