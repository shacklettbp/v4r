#ifndef ASSET_LOAD_INL_INCLUDED
#define ASSET_LOAD_INL_INCLUDED

#include "asset_load.hpp"
#include "scene.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <cassert>
#include <iostream>
#include <type_traits>
#include <vector>
#include <fstream>

namespace v4r {

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
    scene.sceneDirectory = std::filesystem::path(gltf_path).parent_path();

    auto suffix = gltf_path.substr(gltf_path.rfind('.') + 1);
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

        std::vector<uint8_t> json_buffer(json_header.chunkLength +
                                         simdjson::SIMDJSON_PADDING);

        binary_file.read(reinterpret_cast<char *>(json_buffer.data()),
                         json_header.chunkLength);

        try {
            scene.root = scene.jsonParser.parse(
                json_buffer.data(), json_header.chunkLength, false);
        } catch (const simdjson::simdjson_error &e) {
            std::cerr << "GLTF loading '" << gltf_path
                      << "' failed: " << e.what() << std::endl;
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

        std::cout << "Buffers" << std::endl;

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

        std::cout << "bufferViews" << std::endl;

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
                std::cerr << "GLTF loading '" << gltf_path
                          << "' failed: unknown component type" << std::endl;
                fatalExit();
            }

            uint64_t byte_offset;
            auto offset_error = accessor["byteOffset"].get(byte_offset);
            if (offset_error) {
                byte_offset = 0;
            }

            scene.accessors.push_back(GLTFAccessor {
                static_cast<uint32_t>(accessor["bufferView"].get_uint64()),
                static_cast<uint32_t>(byte_offset),
                static_cast<uint32_t>(accessor["count"].get_uint64()),
                type,
            });
        }

        std::cout << "accessors" << std::endl;

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

        std::cout << "images" << std::endl;

        for (const auto &texture : scene.root["textures"]) {
            uint64_t source_idx;
            auto src_err = texture["source"].get(source_idx);
            if (src_err) {
                auto ext_err =
                    texture["extensions"]["GOOGLE_texture_basis"]["source"]
                        .get(source_idx);
                if (ext_err) {
                    std::cerr << "GLTF loading '" << gltf_path
                              << "' failed: texture without source"
                              << std::endl;
                    fatalExit();
                }
            }

            uint64_t sampler_idx;
            auto sampler_error = texture["sampler"].get(sampler_idx);
            if (sampler_error) {
                sampler_idx = 0;
            }

            scene.textures.push_back(GLTFTexture {
                static_cast<uint32_t>(source_idx),
                static_cast<uint32_t>(sampler_idx),
            });
        }

        std::cout << "textures" << std::endl;

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

        std::cout << "materials" << std::endl;

        for (const auto &mesh : scene.root["meshes"]) {
            simdjson::dom::array prims = mesh["primitives"];
            if (prims.size() != 1) {
                std::cerr << "GLTF loading '" << gltf_path << "' failed: "
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

        std::cout << "meshes" << std::endl;

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
            } else {
                glm::mat4 translation(1.f);
                simdjson::dom::array translate_raw;
                auto translate_error = node["translation"].get(translate_raw);
                if (!translate_error) {
                    glm::vec3 translate_vec;
                    float *translate_ptr = glm::value_ptr(translate_vec);
                    for (double vec_elem : translate_raw) {
                        *translate_ptr = vec_elem;
                        translate_ptr++;
                    }
                    translation = glm::translate(translate_vec);
                }

                glm::mat4 rotation(1.f);
                simdjson::dom::array quat_raw;
                auto quat_error = node["rotation"].get(quat_raw);
                if (!quat_error) {
                    glm::quat quat_vec;
                    float *quat_ptr = glm::value_ptr(quat_vec);
                    for (double vec_elem : quat_raw) {
                        *quat_ptr = vec_elem;
                        quat_ptr++;
                    }
                    rotation = glm::mat4_cast(quat_vec);
                }

                glm::mat4 scale(1.f);
                simdjson::dom::array scale_raw;
                auto scale_error = node["scale"].get(scale_raw);
                if (!scale_error) {
                    glm::vec3 scale_vec;
                    float *scale_ptr = glm::value_ptr(scale_vec);
                    for (double vec_elem : scale_raw) {
                        *scale_ptr = vec_elem;
                        scale_ptr++;
                    }
                    scale = glm::scale(scale_vec);
                }

                txfm = translation * rotation * scale;
            }

            scene.nodes.push_back(GLTFNode {
                move(children), static_cast<uint32_t>(mesh_idx), txfm});
        }

        std::cout << "nodes" << std::endl;

        simdjson::dom::array scenes = scene.root["scenes"];
        if (scenes.size() > 1) {
            std::cerr << "GLTF loading '" << gltf_path
                      << "' failed: Multiscene files not supported"
                      << std::endl;
            fatalExit();
        }

        for (uint64_t node_idx : scenes.at(0)["nodes"]) {
            scene.rootNodes.push_back(node_idx);
        }

    } catch (const simdjson::simdjson_error &e) {
        std::cerr << "GLTF loading '" << gltf_path << "' failed: " << e.what()
                  << std::endl;
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
    const uint8_t *start_ptr = buffer.dataPtr + total_offset;
    ;

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

    return getGLTFBufferView<T>(scene, accessor.viewIdx, accessor.offset,
                                accessor.numElems);
}

void ktxCheck(KTX_error_code res)
{
    static const char *ktx_errors[] = {
        "KTX_SUCCESS",
        "KTX_FILE_DATA_ERROR",
        "KTX_FILE_ISPIPE",
        "KTX_FILE_OPEN_FAILED",
        "KTX_FILE_OVERFLOW",
        "KTX_FILE_READ_ERROR",
        "KTX_FILE_SEEK_ERROR",
        "KTX_FILE_UNEXPECTED_EOF",
        "KTX_FILE_WRITE_ERROR",
        "KTX_GL_ERROR",
        "KTX_INVALID_OPERATION",
        "KTX_INVALID_VALUE",
        "KTX_NOT_FOUND",
        "KTX_OUT_OF_MEMORY",
        "KTX_TRANSCODE_FAILED",
        "KTX_UNKNOWN_FILE_FORMAT",
        "KTX_UNSUPPORTED_TEXTURE_TYPE",
        "KTX_UNSUPPORTED_FEATURE",
        "KTX_LIBRARY_NOT_LINKED",
    };

    if (res != KTX_SUCCESS) {
        const char *ktx_error;
        if (res <= KTX_LIBRARY_NOT_LINKED) {
            ktx_error = ktx_errors[res];
        } else {
            ktx_error = "unknown error";
        }
        std::cerr << "GLTF loading failed: failed to load ktx texture - "
                  << ktx_error << std::endl;
        fatalExit();
    }
}

std::shared_ptr<Texture> loadKTXFile(FILE *file)
{
    ktxTexture *ktx_texture;

    KTX_error_code result = ktxTexture_CreateFromStdioStream(
        file, KTX_TEXTURE_CREATE_NO_FLAGS, &ktx_texture);

    ktxCheck(result);

    if (ktx_texture->generateMipmaps) {
        return nullptr;
    }

    return std::shared_ptr<Texture>(new Texture {
        ktx_texture->baseWidth,
        ktx_texture->baseHeight,
        ktx_texture->numLevels,
        ktx_texture,
    });
}

static std::shared_ptr<Texture> gltfLoadTexture(const GLTFScene &scene,
                                                uint32_t texture_idx)
{
    const GLTFImage &img = scene.images[scene.textures[texture_idx].sourceIdx];
    if (img.type != GLTFImageType::EXTERNAL) {
        std::cerr
            << "GLTF loading failed: Only external KTX2 textures supported"
            << std::endl;
        fatalExit();
    }

    const auto &texture_path = scene.sceneDirectory / img.filePath;

    return nullptr;
}

template <typename MaterialParamsType>
std::vector<std::shared_ptr<Material>> gltfParseMaterials(
    const GLTFScene &scene)
{
    std::vector<std::shared_ptr<Material>> materials;
    std::vector<std::shared_ptr<Texture>> textures(scene.textures.size());

    for (const auto &gltf_mat : scene.materials) {
        std::shared_ptr<Texture> texture(nullptr);

        if (gltf_mat.textureIdx < scene.textures.size()) {
            texture = textures[gltf_mat.textureIdx];

            if (texture == nullptr) {
                texture = gltfLoadTexture(scene, gltf_mat.textureIdx);
                textures[gltf_mat.textureIdx] = texture;
            }
        }

        materials.emplace_back(MaterialImpl<MaterialParamsType>::make(
            MaterialParam::DiffuseColorTexture {move(texture)},
            MaterialParam::DiffuseColorUniform {
                glm::vec4(gltf_mat.baseColor, 1.f),
            },
            MaterialParam::SpecularColorTexture {nullptr},
            MaterialParam::SpecularColorUniform {glm::vec4()},
            MaterialParam::ShininessUniform {1.f - gltf_mat.roughness}));
    }

    return materials;
}

template <typename VertexType>
std::pair<std::vector<VertexType>, std::vector<uint32_t>> gltfParseMesh(
    const GLTFScene &scene,
    uint32_t mesh_idx)
{
    std::vector<VertexType> vertices;
    std::vector<uint32_t> indices;

    const GLTFMesh &mesh = scene.meshes[mesh_idx];
    std::optional<StridedSpan<const glm::vec3>> position_accessor;
    std::optional<StridedSpan<const glm::vec3>> normal_accessor;
    std::optional<StridedSpan<const glm::vec2>> uv_accessor;
    std::optional<StridedSpan<const glm::u8vec3>> color_accessor;

    if constexpr (VertexImpl<VertexType>::hasPosition) {
        position_accessor = getGLTFAccessorView<const glm::vec3>(
            scene, mesh.positionIdx.value());
    }

    if constexpr (VertexImpl<VertexType>::hasNormal) {
        if (mesh.normalIdx.has_value()) {
            normal_accessor = getGLTFAccessorView<const glm::vec3>(
                scene, mesh.normalIdx.value());
        }
    }

    if constexpr (VertexImpl<VertexType>::hasUV) {
        if (mesh.uvIdx.has_value()) {
            uv_accessor = getGLTFAccessorView<const glm::vec2>(scene,
                mesh.uvIdx.value());
        }
    }

    if constexpr (VertexImpl<VertexType>::hasColor) {
        if (mesh.colorIdx.has_value()) {
            color_accessor = getGLTFAccessorView<const glm::u8vec3>(
                scene, mesh.colorIdx.value());
        }
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
        VertexType vert {};

        if constexpr (VertexImpl<VertexType>::hasPosition) {
            vert.position = (*position_accessor)[vert_idx];
        }

        if constexpr (VertexImpl<VertexType>::hasNormal) {
            if (normal_accessor.has_value()) {
                vert.normal = (*normal_accessor)[vert_idx];
            }
        }

        if constexpr (VertexImpl<VertexType>::hasUV) {
            if (uv_accessor.has_value()) {
                vert.uv = (*uv_accessor)[vert_idx];
            }
        }

        if constexpr (VertexImpl<VertexType>::hasColor) {
            if (color_accessor.has_value()) {
                vert.color = glm::u8vec4((*color_accessor)[vert_idx], 255);
            }
        }

        vertices.push_back(vert);
    }

    return {move(vertices), move(indices)};
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
