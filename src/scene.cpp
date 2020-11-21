#include "scene.hpp"
#include <vulkan/vulkan_core.h>
#include "loader_definitions.inl"

#include "asset_load.hpp"
#include "shader.hpp"
#include "utils.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#undef TINYPLY_IMPLEMENTATION
#pragma GCC diagnostic pop

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_map>

using namespace std;

namespace v4r {

Texture::~Texture()
{
    ktxTexture_Destroy(data);
}

EnvironmentInit::EnvironmentInit(const vector<InstanceProperties> &instances,
                                 const vector<LightProperties> &l,
                                 uint32_t num_meshes)
    : transforms(num_meshes),
      materials(num_meshes),
      indexMap(),
      reverseIDMap(num_meshes),
      lights(l),
      lightIDs(),
      lightReverseIDs()
{
    indexMap.reserve(instances.size());

    for (uint32_t cur_id = 0; cur_id < instances.size(); cur_id++) {
        const auto &inst = instances[cur_id];
        uint32_t mesh_idx = inst.meshIndex;

        uint32_t inst_idx = transforms[mesh_idx].size();

        transforms[mesh_idx].push_back(inst.txfm);
        materials[mesh_idx].push_back(inst.materialIndex);
        reverseIDMap[mesh_idx].push_back(cur_id);
        indexMap.emplace_back(mesh_idx, inst_idx);
    }

    lightIDs.reserve(lights.size());
    lightReverseIDs.reserve(lights.size());
    for (uint32_t light_idx = 0; light_idx < lights.size(); light_idx++) {
        lightIDs.push_back(light_idx);
        lightReverseIDs.push_back(light_idx);
    }
}

static FrustumBounds computeFrustumBounds(const glm::mat4 &proj)
{
    glm::mat4 t = glm::transpose(proj);
    glm::vec4 xplane = t[3] + t[0];
    glm::vec4 yplane = t[3] + t[1];

    xplane /= glm::length(glm::vec3(xplane));
    yplane /= glm::length(glm::vec3(yplane));

    float znear = proj[3][2] / proj[2][2];
    float zfar = znear * proj[2][2] / (1.f + proj[2][2]);

    return {
        glm::vec4(xplane.x, xplane.z, yplane.y, yplane.z),
        glm::vec2(znear, zfar),
    };
}

EnvironmentState::EnvironmentState(const shared_ptr<Scene> &s,
                                   const glm::mat4 &proj)
    : scene(s),
      projection(proj),
      frustumBounds(computeFrustumBounds(proj)),
      reverseIDMap(scene->envDefaults.reverseIDMap),
      freeIDs(),
      lights(s->envDefaults.lights),
      freeLightIDs(),
      lightIDs(s->envDefaults.lightIDs),
      lightReverseIDs(s->envDefaults.lightReverseIDs)
{}

template <typename VertexType>
static StagedScene stageScene(const vector<shared_ptr<Mesh>> &meshes,
                              const vector<uint8_t> &param_bytes,
                              const DeviceState &dev,
                              MemoryAllocator &alloc)
{
    using MeshT = VertexMesh<VertexType>;

    VkDeviceSize total_vertex_bytes = 0;
    VkDeviceSize total_index_bytes = 0;

    for (const auto &generic_mesh : meshes) {
        auto mesh = static_cast<const MeshT *>(generic_mesh.get());
        total_vertex_bytes += sizeof(VertexType) * mesh->vertices.size();
        total_index_bytes += sizeof(uint32_t) * mesh->indices.size();
    }

    VkDeviceSize total_geometry_bytes = total_vertex_bytes + total_index_bytes;

    VkDeviceSize total_bytes = total_geometry_bytes;
    VkDeviceSize material_offset = 0;
    VkDeviceSize num_material_bytes = param_bytes.size();
    if (num_material_bytes > 0) {
        material_offset = alloc.alignUniformBufferOffset(total_bytes);

        total_bytes = material_offset + param_bytes.size();
    }

    VkDeviceSize mesh_info_offset =
        alloc.alignStorageBufferOffset(total_bytes);

    total_bytes = mesh_info_offset + (sizeof(MeshInfo) * meshes.size());

    HostBuffer staging = alloc.makeStagingBuffer(total_bytes);

    // Copy all vertices
    uint32_t vertex_offset = 0;
    uint8_t *staging_start = reinterpret_cast<uint8_t *>(staging.ptr);
    uint8_t *cur_ptr = staging_start;
    // MeshInfo *mesh_infos = reinterpret_cast<MeshInfo *>(
    //        staging_start + mesh_info_offset);

    for (uint32_t mesh_idx = 0; mesh_idx < meshes.size(); mesh_idx++) {
        const auto &generic_mesh = meshes[mesh_idx];
        auto mesh = static_cast<const MeshT *>(generic_mesh.get());
        VkDeviceSize vertex_bytes = sizeof(VertexType) * mesh->vertices.size();
        memcpy(cur_ptr, mesh->vertices.data(), vertex_bytes);

        // mesh_infos[mesh_idx].vertexOffset = vertex_offset;

        cur_ptr += vertex_bytes;
        vertex_offset += mesh->vertices.size();
    }

    // Copy all indices
    uint32_t cur_mesh_index = 0;
    for (uint32_t mesh_idx = 0; mesh_idx < meshes.size(); mesh_idx++) {
        auto mesh = static_cast<const MeshT *>(meshes[mesh_idx].get());

        VkDeviceSize index_bytes = sizeof(uint32_t) * mesh->indices.size();
        memcpy(cur_ptr, mesh->indices.data(), index_bytes);

        // mesh_infos[mesh_idx].indexOffset = cur_mesh_index;
        // mesh_infos[mesh_idx].indexCount =
        //    static_cast<uint32_t>(mesh->indices.size());

        cur_ptr += index_bytes;
        cur_mesh_index += mesh->indices.size();
    }

    // Optionally copy material params
    if (param_bytes.size() > 0) {
        memcpy(staging_start + material_offset, param_bytes.data(),
               param_bytes.size());
    }

    staging.flush(dev);

    // FIXME: totally broken post mesh chunks requirement
    return {move(staging),
            StagingHeader {
                total_vertex_bytes,
                0,
                0,
                0,
                0,
                0,
                0,
                material_offset,
                num_material_bytes,
                total_bytes,
                static_cast<uint32_t>(meshes.size()),
            },
            {}};
}

template <typename VertexType>
VertexMesh<VertexType>::VertexMesh(vector<VertexType> v, vector<uint32_t> i)
    : Mesh {move(i)},
      vertices(move(v))
{}

template <typename VertexType>
static shared_ptr<Mesh> makeSharedMesh(vector<VertexType> vertices,
                                       vector<uint32_t> indices)
{
    return shared_ptr<VertexMesh<VertexType>>(
        new VertexMesh<VertexType>(move(vertices), move(indices)));
}

template <typename VertexType>
static shared_ptr<Mesh> loadMeshGLTF(string_view geometry_path)
{
    auto scene = gltfLoad(geometry_path);

    if (scene.meshes.size() == 0) {
        cerr << "No meshes in file " << geometry_path << endl;
    }

    auto [vertices, indices] = gltfParseMesh<VertexType>(scene, 0);

    return makeSharedMesh(move(vertices), move(indices));
}

static bool isGLTF(string_view gltf_path)
{
    auto suffix = gltf_path.substr(gltf_path.rfind('.') + 1);
    return suffix == "glb" || suffix == "gltf";
}

template <typename VertexType>
static shared_ptr<Mesh> loadMesh(string_view geometry_path)
{
    if (isGLTF(geometry_path)) {
        return loadMeshGLTF<VertexType>(geometry_path);
    } else {
        cerr << "Only GLTF is supported" << endl;
        abort();
    }
}

template <typename VertexType, typename MaterialParamsType>
static SceneDescription parseGLTFScene(std::string_view scene_path,
                                       const glm::mat4 &coordinate_txfm)
{
    auto raw_scene = gltfLoad(scene_path);

    constexpr bool need_materials =
        !std::is_same_v<MaterialParamsType, NoMaterial>;

    std::vector<std::shared_ptr<Material>> materials;
    std::vector<std::shared_ptr<Mesh>> geometry;
    geometry.reserve(raw_scene.meshes.size());

    if constexpr (need_materials) {
        materials = gltfParseMaterials<MaterialParamsType>(raw_scene);
    }

    for (uint32_t mesh_idx = 0; mesh_idx < raw_scene.meshes.size();
         mesh_idx++) {
        auto [vertices, indices] =
            gltfParseMesh<VertexType>(raw_scene, mesh_idx);
        geometry.emplace_back(makeSharedMesh(move(vertices), move(indices)));
    }

    SceneDescription scene_desc(move(geometry), move(materials));

    gltfParseInstances(scene_desc, raw_scene, coordinate_txfm);

    return scene_desc;
}

template <typename VertexType>
static SceneDescription parsePLYScene(string_view scene_path,
                                      const glm::mat4 &coordinate_txfm)
{
    using namespace tinyply;

    ifstream file(filesystem::path(scene_path), ios::binary);

    if (file.fail()) {
        cerr << "Failed to open: " << scene_path << endl;
        fatalExit();
    }

    PlyFile ply_file;
    ply_file.parse_header(file);

    vector<shared_ptr<Mesh>> geometry;
    try {
        auto positions = ply_file.request_properties_from_element(
            "vertex", {"x", "y", "z"});

        auto colors = ply_file.request_properties_from_element(
            "vertex", {"red", "green", "blue"});

        auto faces = ply_file.request_properties_from_element(
            "face", {"vertex_indices"}, 4);

        ply_file.read(file);

        vector<VertexType> vertices(positions->count);
        glm::vec3 *pos_buffer =
            reinterpret_cast<glm::vec3 *>(positions->buffer.get());
        glm::u8vec3 *color_buffer =
            reinterpret_cast<glm::u8vec3 *>(colors->buffer.get());
        for (uint32_t i = 0; i < vertices.size(); i++) {
            auto &v = vertices[i];

            if constexpr (VertexImpl<VertexType>::hasPosition) {
                v.position = pos_buffer[i];
            }

            if constexpr (VertexImpl<VertexType>::hasColor) {
                v.color = glm::u8vec4(color_buffer[i], 255);
            }
        }

        uint32_t *quad_indices = reinterpret_cast<uint32_t *>(faces->buffer.get());
        vector<uint32_t> indices;
        indices.reserve(faces->count * 6);

        for (uint32_t i = 0; i < faces->count; i++) {
            uint32_t base_quad_idx = i * 4;
            indices.push_back(quad_indices[base_quad_idx]);
            indices.push_back(quad_indices[base_quad_idx + 1]);
            indices.push_back(quad_indices[base_quad_idx + 2]);

            indices.push_back(quad_indices[base_quad_idx + 2]);
            indices.push_back(quad_indices[base_quad_idx + 3]);
            indices.push_back(quad_indices[base_quad_idx]);
        }

        geometry.emplace_back(makeSharedMesh(move(vertices), move(indices)));
    } catch (exception &e) {
        cerr << "Failed to read ply file: " << e.what() << endl;
        fatalExit();
    }

    SceneDescription scene_desc(move(geometry), {});

    scene_desc.addInstance(0, 0, coordinate_txfm);
    
    return scene_desc;
}

static bool isPLY(string_view scene_path)
{
    auto suffix = scene_path.substr(scene_path.rfind('.') + 1);
    return suffix == "ply";
}

template <typename VertexType, typename MaterialParamsType>
static SceneDescription parseScene(string_view scene_path,
                                   const glm::mat4 &coordinate_txfm)
{
    if (isGLTF(scene_path)) {
        return parseGLTFScene<VertexType, MaterialParamsType>(scene_path,
                                                              coordinate_txfm);
    } else if (isPLY(scene_path)) {
        return parsePLYScene<VertexType>(scene_path, coordinate_txfm);
    } else {
        cerr << "Only GLTF is supported" << endl;
        abort();
    }
}

template <typename VertexType, typename MaterialParamsType>
static SceneLoadInfo loadPreprocessedScene(string_view scene_path_name,
                                           const glm::mat4 &coordinate_txfm,
                                           MemoryAllocator &alloc)
{
    filesystem::path scene_path(scene_path_name);
    filesystem::path scene_dir = scene_path.parent_path();

    ifstream scene_file(scene_path, ios::binary);

    auto read_uint = [&]() {
        uint32_t val;
        scene_file.read(reinterpret_cast<char *>(&val), sizeof(uint32_t));

        return val;
    };

    uint32_t magic = read_uint();
    if (magic != 0x55555555) {
        cerr << "Invalid preprocessed scene" << endl;
        fatalExit();
    }

    uint32_t depth_offset = read_uint();
    uint32_t rgb_offset = read_uint();

    // FIXME something less hacky to determine this:
    if constexpr (is_same_v<MaterialParamsType, NoMaterial> &&
                  !VertexImpl<VertexType>::hasColor) {
        scene_file.seekg(depth_offset * sizeof(uint32_t), ios::cur);
    } else {
        scene_file.seekg(rgb_offset * sizeof(uint32_t), ios::cur);
    }

    StagingHeader hdr;
    scene_file.read(reinterpret_cast<char *>(&hdr), sizeof(StagingHeader));

    auto cur_pos = scene_file.tellg();
    auto post_hdr_alignment = cur_pos % 256;
    if (post_hdr_alignment != 0) {
        scene_file.seekg(256 - post_hdr_alignment, ios::cur);
    }

    HostBuffer staging_buffer = alloc.makeStagingBuffer(hdr.totalBytes);
    scene_file.read(reinterpret_cast<char *>(staging_buffer.ptr),
                    hdr.totalBytes);

    vector<MeshInfo> mesh_infos(hdr.numMeshes);
    scene_file.read(reinterpret_cast<char *>(mesh_infos.data()),
                    sizeof(MeshInfo) * hdr.numMeshes);

    MaterialMetadata materials;
    uint32_t num_textures = read_uint();
    vector<char> name_buffer;
    for (uint32_t tex_idx = 0; tex_idx < num_textures; tex_idx++) {
        do {
            name_buffer.push_back(scene_file.get());
        } while (name_buffer.back() != 0);

        materials.textures.emplace_back(scene_dir / name_buffer.data());
        name_buffer.clear();
    }

    materials.numMaterials = read_uint();
    materials.texturesPerMaterial = read_uint();

    materials.textureIndices.resize(materials.numMaterials *
                                    materials.texturesPerMaterial);

    scene_file.read(reinterpret_cast<char *>(materials.textureIndices.data()),
                    sizeof(uint32_t) * materials.textureIndices.size());

    uint32_t num_instances = read_uint();

    // FIXME this should just be baked in...
    vector<InstanceProperties> instances;
    instances.reserve(num_instances);

    for (uint32_t inst_idx = 0; inst_idx < num_instances; inst_idx++) {
        uint32_t mesh_index = read_uint();
        uint32_t material_index = read_uint();
        glm::mat4x3 txfm;
        scene_file.read(reinterpret_cast<char *>(&txfm), sizeof(glm::mat4x3));
        instances.emplace_back(mesh_index, material_index,
                               glm::mat4x3(coordinate_txfm * glm::mat4(txfm)));
    }

    return SceneLoadInfo {StagedScene {
                              move(staging_buffer),
                              hdr,
                              move(mesh_infos),
                          },
                          move(materials),
                          EnvironmentInit(move(instances), {}, hdr.numMeshes),
                          sizeof(VertexType)};
}

template <typename VertexType, typename MaterialParamsType>
LoaderImpl LoaderImpl::create()
{
    return {
        v4r::stageScene<VertexType>,
        v4r::parseScene<VertexType, MaterialParamsType>,
        v4r::loadMesh<VertexType>,
        v4r::loadPreprocessedScene<VertexType, MaterialParamsType>,
    };
}

LoaderState::LoaderState(
    const DeviceState &d,
    const LoaderImpl &impl,
    const VkDescriptorSetLayout &scene_set_layout,
    DescriptorManager::MakePoolType make_scene_pool,
    const VkDescriptorSetLayout &mesh_cull_scene_set_layout,
    DescriptorManager::MakePoolType make_mesh_cull_scene_pool,
    MemoryAllocator &alc,
    QueueManager &queue_manager,
    const glm::mat4 &coordinate_transform)
    : dev(d),
      gfxPool(makeCmdPool(dev, dev.gfxQF)),
      gfxQueue(queue_manager.allocateGraphicsQueue()),
      gfxCopyCommand(makeCmdBuffer(dev, gfxPool)),
      transferPool(makeCmdPool(dev, dev.transferQF)),
      transferQueue(queue_manager.allocateTransferQueue()),
      transferStageCommand(makeCmdBuffer(dev, transferPool)),
      semaphore(makeBinarySemaphore(dev)),
      fence(makeFence(dev)),
      alloc(alc),
      descriptorManager(dev, scene_set_layout, make_scene_pool),
      cullDescriptorManager(dev,
                            mesh_cull_scene_set_layout,
                            make_mesh_cull_scene_pool),
      coordinateTransform(coordinate_transform),
      impl_(impl)
{}

// FIXME not static so preprocess can link
pair<vector<uint8_t>, MaterialMetadata> stageMaterials(
    const vector<shared_ptr<Material>> &materials)
{
    vector<filesystem::path> textures;
    unordered_map<string, size_t> texture_tracker;
    vector<size_t> param_offsets;
    param_offsets.reserve(materials.size());

    uint64_t num_material_bytes = 0;
    for (const auto &material : materials) {
        num_material_bytes += material->paramBytes.size();
    }

    vector<uint8_t> packed_params(num_material_bytes);
    uint8_t *cur_param_ptr = packed_params.data();

    uint32_t textures_per_material = 0;
    if (materials.size() > 0) {
        textures_per_material = materials[0]->textures.size();
    }

    vector<uint32_t> texture_indices;
    texture_indices.reserve(materials.size() * textures_per_material);

    (void)cur_param_ptr;

#if 0
    for (const auto &material : materials) {
        memcpy(cur_param_ptr, material->paramBytes.data(),
               material->paramBytes.size());

        for (const auto &texture : material->textures) {
            if (texture == nullptr) {
                cerr << "Texture missing when renderer set with a"
                        "pipeline that needs textures"
                     << endl;
            }
            auto [iter, inserted] =
                texture_tracker.emplace(texture.get(), textures.size());

            if (inserted) {
                textures.push_back(texture);
            }

            texture_indices.push_back(iter->second);
        }

        param_offsets.push_back(cur_param_ptr - packed_params.data());
        cur_param_ptr += material->paramBytes.size();
    }
#endif

    return {move(packed_params),
            {
                textures,
                uint32_t(materials.size()),
                textures_per_material,
                texture_indices,
            }};
}

shared_ptr<Scene> LoaderState::loadScene(string_view scene_path)
{
    if (scene_path.substr(scene_path.rfind('.') + 1) == "bps") {
        return makeScene(impl_.loadPreprocessedScene(
            scene_path, coordinateTransform, alloc));
    } else {
        SceneDescription desc =
            impl_.parseScene(scene_path, coordinateTransform);

        return makeScene(desc);
    }
}

shared_ptr<Scene> LoaderState::makeScene(const SceneDescription &desc)
{
#if 0
    auto [staged_params, material_metadata] =
        stageMaterials(desc.getMaterials());

    StagedScene staged_scene =
        impl_.stageScene(desc.getMeshes(), staged_params, dev, alloc);

    return makeScene(SceneLoadInfo {
        move(staged_scene),
        move(material_metadata),
        EnvironmentInit(desc.getDefaultInstances(), desc.getDefaultLights(),
                        staged_scene.hdr.numMeshes),
    });
#endif

    (void)desc;
    return nullptr;
}

static VkDeviceAddress getBufferDeviceAddress(const DeviceState &dev,
                                              VkBuffer hdl)
{
    VkBufferDeviceAddressInfo addr_info;
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.pNext = nullptr;
    addr_info.buffer = hdl;
    return dev.dt.getBufferDeviceAddressKHR(dev.hdl, &addr_info);
}

shared_ptr<Scene> LoaderState::makeScene(SceneLoadInfo load_info)
{
    auto &[staged, material_metadata, env_init, vertex_size] = load_info;

    vector<shared_ptr<Texture>> cpu_textures;
    cpu_textures.reserve(material_metadata.textures.size());
    for (const auto &texture_path : material_metadata.textures) {
        // Hack to keep file descriptor count down
        FILE *file = fopen(texture_path.c_str(), "rb");
        if (!file) {
            cerr << "Texture loading failed: Could not open "
                 << texture_path << endl;
            fatalExit();
        }

        auto texture = loadKTXFile(file);
        if (texture == nullptr) {
            cerr << "Texture loading failed: Error loading "
                 << texture_path << endl;
            fatalExit();
        }

        ktxTexture *ktx = texture->data;
        assert(ktx->classId == ktxTexture2_c);

        ktxTexture2 *ktx2 = reinterpret_cast<ktxTexture2 *>(ktx);
        KTX_error_code res =
            ktxTexture2_TranscodeBasis(ktx2, KTX_TTF_BC7_RGBA, 0);
        ktxCheck(res);
        fclose(file);

        cpu_textures.emplace_back(move(texture));
    }

    vector<LocalImage> gpu_textures;

    gpu_textures.reserve(cpu_textures.size());

    // FIXME - custom loader or hacked loader that makes doing this mip
    // level by mip level possible
    VkDeviceSize total_texture_bytes = 0;
    for (const shared_ptr<Texture> &texture : cpu_textures) {
        ktxTexture *ktx = texture->data;

        for (uint32_t level = 0; level < texture->numLevels; level++) {
            total_texture_bytes += ktxTexture_GetImageSize(ktx, level);
        }

        gpu_textures.emplace_back(alloc.makeTexture(
            texture->width, texture->height, texture->numLevels));
    }
    const uint32_t num_textures = cpu_textures.size();

    optional<HostBuffer> texture_staging;
    if (num_textures > 0) {
        texture_staging.emplace(alloc.makeStagingBuffer(total_texture_bytes));
    }

    // Copy all geometry into single buffer

    LocalBuffer data = alloc.makeLocalBuffer(staged.hdr.totalBytes);

    // FIXME more than one mesh
    VkAccelerationStructureCreateGeometryTypeInfoKHR blas_geo_info {};
    blas_geo_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_GEOMETRY_TYPE_INFO_KHR;
    blas_geo_info.pNext = nullptr;
    blas_geo_info.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    blas_geo_info.maxPrimitiveCount = staged.meshMetadata[0].numIndices / 3;
    blas_geo_info.indexType = VK_INDEX_TYPE_UINT32;
    blas_geo_info.maxVertexCount = staged.meshMetadata[0].numVertices;
    blas_geo_info.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    blas_geo_info.allowsTransforms = false;

    VkAccelerationStructureCreateInfoKHR blas_info;
    blas_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    blas_info.pNext = nullptr;
    blas_info.compactedSize = 0;
    blas_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    blas_info.flags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    blas_info.maxGeometryCount = 1;
    blas_info.pGeometryInfos = &blas_geo_info;
    blas_info.deviceAddress = 0;

    VkAccelerationStructureKHR blas;
    REQ_VK(dev.dt.createAccelerationStructureKHR(dev.hdl, &blas_info, nullptr,
                                                 &blas));

    VkDeviceMemory blas_memory = alloc.allocateAccelerationStructureMemory(blas);

    VkBindAccelerationStructureMemoryInfoKHR blas_bind_info {};
    blas_bind_info.sType =
        VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_KHR;
    blas_bind_info.pNext = nullptr;
    blas_bind_info.accelerationStructure = blas;
    blas_bind_info.memory = blas_memory;
    REQ_VK(dev.dt.bindAccelerationStructureMemoryKHR(dev.hdl, 1, &blas_bind_info));

    VkDeviceOrHostAddressConstKHR vertex_dev_addr;
    vertex_dev_addr.deviceAddress = getBufferDeviceAddress(dev, data.buffer);

    VkDeviceOrHostAddressConstKHR index_dev_addr;
    index_dev_addr.deviceAddress =
        vertex_dev_addr.deviceAddress + staged.meshMetadata[0].indexOffset;

    VkAccelerationStructureGeometryKHR blas_geo;
    blas_geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    blas_geo.pNext = nullptr;
    blas_geo.geometryType = blas_geo_info.geometryType;
    blas_geo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureGeometryTrianglesDataKHR &blas_tri_data =
        blas_geo.geometry.triangles;

    blas_tri_data.sType = 
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    blas_tri_data.pNext = nullptr;
    blas_tri_data.vertexFormat = blas_geo_info.vertexFormat;
    blas_tri_data.vertexData = vertex_dev_addr;
    blas_tri_data.vertexStride = vertex_size;
    blas_tri_data.indexType = blas_geo_info.indexType;
    blas_tri_data.indexData = index_dev_addr;
    blas_tri_data.transformData.hostAddress = nullptr;

    LocalBuffer blas_scratch = alloc.makeAccelerationStructureScratchBuffer(blas);
    VkDeviceAddress blas_scratch_dev_addr =
        getBufferDeviceAddress(dev, blas_scratch.buffer);

    VkAccelerationStructureGeometryKHR *blas_geo_ptr = &blas_geo;

    VkAccelerationStructureBuildGeometryInfoKHR blas_build_info;
    blas_build_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    blas_build_info.pNext = nullptr;
    blas_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    blas_build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    blas_build_info.update = VK_FALSE;
    blas_build_info.srcAccelerationStructure = VK_NULL_HANDLE;
    blas_build_info.dstAccelerationStructure = blas;
    blas_build_info.geometryArrayOfPointers = VK_FALSE;
    blas_build_info.geometryCount = 1;
    blas_build_info.ppGeometries = &blas_geo_ptr;
    blas_build_info.scratchData.deviceAddress = blas_scratch_dev_addr;

    VkAccelerationStructureBuildOffsetInfoKHR blas_offset;
    blas_offset.primitiveOffset = 0;
    blas_offset.primitiveCount = blas_geo_info.maxPrimitiveCount;
    blas_offset.firstVertex = 0;
    blas_offset.transformOffset = 0;

    auto blas_offset_ptr = &blas_offset;

    VkAccelerationStructureDeviceAddressInfoKHR blas_addr_info;
    blas_addr_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    blas_addr_info.pNext = nullptr;
    blas_addr_info.accelerationStructure = blas;


    AccelerationStructure blas_data {
        blas,
        dev.dt.getAccelerationStructureDeviceAddressKHR(dev.hdl, &blas_addr_info),
        blas_memory,
    };

    VkAccelerationStructureCreateGeometryTypeInfoKHR tlas_geo_info {};
    tlas_geo_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_GEOMETRY_TYPE_INFO_KHR;
    tlas_geo_info.pNext = nullptr;
    tlas_geo_info.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlas_geo_info.maxPrimitiveCount = 1;

    VkAccelerationStructureCreateInfoKHR tlas_info;
    tlas_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    tlas_info.pNext = nullptr;
    tlas_info.compactedSize = 0;
    tlas_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlas_info.flags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlas_info.maxGeometryCount = 1;
    tlas_info.pGeometryInfos = &tlas_geo_info;
    tlas_info.deviceAddress = 0;

    VkAccelerationStructureKHR tlas;
    REQ_VK(dev.dt.createAccelerationStructureKHR(dev.hdl, &tlas_info,
                                                 nullptr, &tlas));

    VkDeviceMemory tlas_memory = alloc.allocateAccelerationStructureMemory(tlas);

    VkBindAccelerationStructureMemoryInfoKHR tlas_bind_info {};
    tlas_bind_info.sType =
        VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_KHR;
    tlas_bind_info.accelerationStructure = tlas;
    tlas_bind_info.memory = tlas_memory;
    REQ_VK(dev.dt.bindAccelerationStructureMemoryKHR(dev.hdl, 1, &tlas_bind_info));

    VkTransformMatrixKHR tlas_txfm {{
        {1.f, 0.f, 0.f, 0.f},
        {0.f, 1.f, 0.f, 0.f},
        {0.f, 0.f, 1.f, 0.f},
    }};

    VkAccelerationStructureInstanceKHR tlas_instance;
    tlas_instance.transform = tlas_txfm;
    tlas_instance.instanceCustomIndex = 0;
    tlas_instance.mask = 0xFF;
    tlas_instance.instanceShaderBindingTableRecordOffset = 0;
    tlas_instance.flags = 0;
    tlas_instance.accelerationStructureReference = blas_data.devAddr;

    HostBuffer inst_buffer = alloc.makeHostBuffer(sizeof(tlas_instance));
    memcpy(inst_buffer.ptr, &tlas_instance, sizeof(tlas_instance));

    VkDeviceOrHostAddressConstKHR inst_buffer_addr;
    inst_buffer_addr.deviceAddress =
        getBufferDeviceAddress(dev, inst_buffer.buffer);

    VkAccelerationStructureGeometryKHR tlas_geo;
    tlas_geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlas_geo.pNext = nullptr;
    tlas_geo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlas_geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;

    VkAccelerationStructureGeometryInstancesDataKHR &tlas_geo_inst =
        tlas_geo.geometry.instances;

    tlas_geo_inst.sType = 
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlas_geo_inst.pNext = nullptr;
    tlas_geo_inst.arrayOfPointers = false;
    tlas_geo_inst.data = inst_buffer_addr;

    auto tlas_geo_ptr = &tlas_geo;

    LocalBuffer tlas_scratch = alloc.makeAccelerationStructureScratchBuffer(tlas);
    VkDeviceAddress tlas_scratch_dev_addr =
        getBufferDeviceAddress(dev, tlas_scratch.buffer);

    VkAccelerationStructureBuildGeometryInfoKHR tlas_build_info;
    tlas_build_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlas_build_info.pNext = nullptr;
    tlas_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlas_build_info.flags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlas_build_info.update = VK_FALSE;
    tlas_build_info.srcAccelerationStructure = VK_NULL_HANDLE;
    tlas_build_info.dstAccelerationStructure = tlas;
    tlas_build_info.geometryArrayOfPointers = VK_FALSE;
    tlas_build_info.geometryCount = 1;
    tlas_build_info.ppGeometries = &tlas_geo_ptr;
    tlas_build_info.scratchData.deviceAddress = tlas_scratch_dev_addr;

    VkAccelerationStructureDeviceAddressInfoKHR tlas_addr_info;
    blas_addr_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    blas_addr_info.pNext = nullptr;
    blas_addr_info.accelerationStructure = tlas;


    // Start recording for transfer queue
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(transferStageCommand, &begin_info));

    // Copy vertex/index buffer onto GPU
    VkBufferCopy copy_settings {};
    copy_settings.size = staged.hdr.totalBytes;
    dev.dt.cmdCopyBuffer(transferStageCommand, staged.buffer.buffer,
                         data.buffer, 1, &copy_settings);

    // Set initial texture layouts
    DynArray<VkImageMemoryBarrier> texture_barriers(num_textures);
    for (size_t i = 0; i < num_textures; i++) {
        const LocalImage &gpu_texture = gpu_textures[i];
        VkImageMemoryBarrier &barrier = texture_barriers[i];

        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = gpu_texture.image;
        barrier.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, gpu_texture.mipLevels, 0, 1,
        };
    }

    if (num_textures > 0) {
        dev.dt.cmdPipelineBarrier(
            transferStageCommand, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
            texture_barriers.size(), texture_barriers.data());

        // Copy all textures into staging buffer & record cpu -> gpu copies
        uint8_t *base_texture_staging =
            reinterpret_cast<uint8_t *>(texture_staging->ptr);
        VkDeviceSize cur_staging_offset = 0;

        vector<VkBufferImageCopy> copy_infos;
        for (size_t i = 0; i < num_textures; i++) {
            const shared_ptr<Texture> &cpu_texture = cpu_textures[i];
            const LocalImage &gpu_texture = gpu_textures[i];
            uint32_t base_width = cpu_texture->width;
            uint32_t base_height = cpu_texture->height;
            uint32_t num_levels = cpu_texture->numLevels;

            copy_infos.resize(num_levels);
            ktxTexture *ktx = cpu_texture->data;
            const uint8_t *ktx_data = ktxTexture_GetData(ktx);

            for (uint32_t level = 0; level < num_levels; level++) {
                // Copy to staging
                VkDeviceSize ktx_level_offset;
                KTX_error_code res = ktxTexture_GetImageOffset(
                    ktx, level, 0, 0, &ktx_level_offset);
                ktxCheck(res);

                VkDeviceSize num_level_bytes =
                    ktxTexture_GetImageSize(ktx, level);

                memcpy(base_texture_staging + cur_staging_offset,
                       ktx_data + ktx_level_offset, num_level_bytes);

                uint32_t level_div = 1 << level;
                uint32_t level_width = max(1U, base_width / level_div);
                uint32_t level_height = max(1U, base_height / level_div);

                // Set level copy
                VkBufferImageCopy copy_info {};
                copy_info.bufferOffset = cur_staging_offset;
                copy_info.imageSubresource.aspectMask =
                    VK_IMAGE_ASPECT_COLOR_BIT;
                copy_info.imageSubresource.mipLevel = level;
                copy_info.imageSubresource.baseArrayLayer = 0;
                copy_info.imageSubresource.layerCount = 1;
                copy_info.imageExtent = {
                    level_width,
                    level_height,
                    1,
                };

                copy_infos[level] = copy_info;

                cur_staging_offset += num_level_bytes;
            }

            // Note that number of copy commands is num_levels
            // not copy_infos.size(), because the vector is shared
            // between textures to avoid allocs
            dev.dt.cmdCopyBufferToImage(
                transferStageCommand, texture_staging->buffer,
                gpu_texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                num_levels, copy_infos.data());
        }

        // Flush staging buffer
        texture_staging->flush(dev);

        // Transfer queue relinquish texture barriers
        for (VkImageMemoryBarrier &barrier : texture_barriers) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            ;
            barrier.srcQueueFamilyIndex = dev.transferQF;
            barrier.dstQueueFamilyIndex = dev.gfxQF;
        }
    }

    // Transfer queue relinquish geometry
    VkBufferMemoryBarrier geometry_barrier;
    geometry_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    geometry_barrier.pNext = nullptr;
    geometry_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    geometry_barrier.dstAccessMask = 0;
    geometry_barrier.srcQueueFamilyIndex = dev.transferQF;
    geometry_barrier.dstQueueFamilyIndex = dev.gfxQF;
    geometry_barrier.buffer = data.buffer;
    geometry_barrier.offset = 0;
    geometry_barrier.size = staged.hdr.totalBytes;

    // Geometry & texture barrier execute.
    dev.dt.cmdPipelineBarrier(
        transferStageCommand, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &geometry_barrier,
        texture_barriers.size(), texture_barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(transferStageCommand));

    VkSubmitInfo copy_submit {};
    copy_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    copy_submit.commandBufferCount = 1;
    copy_submit.pCommandBuffers = &transferStageCommand;
    copy_submit.signalSemaphoreCount = 1;
    copy_submit.pSignalSemaphores = &semaphore;

    transferQueue.submit(dev, 1, &copy_submit, VK_NULL_HANDLE);

    // Start recording for graphics queue
    REQ_VK(dev.dt.beginCommandBuffer(gfxCopyCommand, &begin_info));

    // Finish moving geometry onto graphics queue family
    // geometry and textures need separate barriers due to different
    // dependent stages
    geometry_barrier.srcAccessMask = 0;
    geometry_barrier.dstAccessMask =
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    dev.dt.cmdPipelineBarrier(gfxCopyCommand,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 0,
        nullptr, 1, &geometry_barrier, 0, nullptr);

    if (num_textures > 0) {
        for (VkImageMemoryBarrier &barrier : texture_barriers) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = dev.transferQF;
            barrier.dstQueueFamilyIndex = dev.gfxQF;
        }

        // Finish acquiring mip level 0 on graphics queue and transition layout
        dev.dt.cmdPipelineBarrier(
            gfxCopyCommand, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
            texture_barriers.size(), texture_barriers.data());
    }

    dev.dt.cmdBuildAccelerationStructureKHR(gfxCopyCommand, 1,
                                            &blas_build_info,
                                            &blas_offset_ptr);

    VkMemoryBarrier as_barrier;
    as_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    as_barrier.pNext = nullptr;
    as_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    as_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    // Is this barrier between BLAS and TLAS actually necessary?
    dev.dt.cmdPipelineBarrier(gfxCopyCommand,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0,
        1, &as_barrier,
        0, nullptr,
        0, nullptr);

    dev.dt.cmdBuildAccelerationStructureKHR(gfxCopyCommand, 1,
                                            &tlas_build_info,
                                            &blas_offset_ptr);

    dev.dt.cmdPipelineBarrier(gfxCopyCommand,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0,
        1, &as_barrier,
        0, nullptr,
        0, nullptr);

    REQ_VK(dev.dt.endCommandBuffer(gfxCopyCommand));

    VkSubmitInfo gfx_submit {};
    gfx_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    gfx_submit.waitSemaphoreCount = 1;
    gfx_submit.pWaitSemaphores = &semaphore;
    VkPipelineStageFlags sema_wait_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    gfx_submit.pWaitDstStageMask = &sema_wait_mask;
    gfx_submit.commandBufferCount = 1;
    gfx_submit.pCommandBuffers = &gfxCopyCommand;

    gfxQueue.submit(dev, 1, &gfx_submit, fence);

    waitForFenceInfinitely(dev, fence);
    resetFence(dev, fence);

    vector<VkImageView> texture_views;
    texture_views.reserve(num_textures);
    for (const LocalImage &gpu_texture : gpu_textures) {
        VkImageViewCreateInfo view_info;
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.pNext = nullptr;
        view_info.flags = 0;
        view_info.image = gpu_texture.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = alloc.getFormats().sdrTexture;
        view_info.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                                VK_COMPONENT_SWIZZLE_B,
                                VK_COMPONENT_SWIZZLE_A};
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                      gpu_texture.mipLevels, 0, 1};

        VkImageView view;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &view));

        texture_views.push_back(view);
    }

    assert(material_metadata.numMaterials <= VulkanConfig::max_materials);

    DescriptorSet scene_set = descriptorManager.makeSet();

    vector<VkWriteDescriptorSet> desc_updates;
    desc_updates.reserve(3);

    VkWriteDescriptorSetAccelerationStructureKHR desc_update_as;
    desc_update_as.sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    desc_update_as.pNext = nullptr;
    desc_update_as.accelerationStructureCount = 1;
    desc_update_as.pAccelerationStructures = &tlas;

    VkWriteDescriptorSet desc_update;
    desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_update.pNext = &desc_update_as;
    desc_update.dstSet = scene_set.hdl;
    desc_update.dstBinding = 0;
    desc_update.dstArrayElement = 0;
    desc_update.descriptorCount = 1;
    desc_update.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    desc_update.pImageInfo = nullptr;
    desc_update.pBufferInfo = nullptr;
    desc_update.pTexelBufferView = nullptr;
    desc_updates.push_back(desc_update);

    VkDescriptorBufferInfo vertex_buffer_info;
    vertex_buffer_info.buffer = data.buffer;
    vertex_buffer_info.offset = 0;
    vertex_buffer_info.range = staged.meshMetadata[0].numVertices * vertex_size;

    desc_update.pNext = nullptr;
    desc_update.dstBinding = 1;
    desc_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    desc_update.pImageInfo = nullptr;
    desc_update.pBufferInfo = &vertex_buffer_info;
    desc_updates.push_back(desc_update);

    VkDescriptorBufferInfo index_buffer_info;
    vertex_buffer_info.buffer = data.buffer;
    vertex_buffer_info.offset = staged.hdr.indexOffset;
    vertex_buffer_info.range = staged.meshMetadata[0].numIndices * sizeof(uint32_t);
    desc_update.pBufferInfo = &index_buffer_info;
    desc_updates.push_back(desc_update);

    dev.dt.updateDescriptorSets(dev.hdl, desc_updates.size(),
                                desc_updates.data(), 0, nullptr);

    return make_shared<Scene>(Scene {
        dev,
        move(gpu_textures),
        move(texture_views),
        move(scene_set),
        move(data),
        staged.hdr.indexOffset,
        move(staged.meshMetadata),
        staged.hdr.numMeshes,
        AccelerationStructure {
            tlas,
            dev.dt.getAccelerationStructureDeviceAddressKHR(dev.hdl,
                                                            &tlas_addr_info),
            tlas_memory,
        },
        { blas_data },
        move(env_init),
    });
}

Scene::~Scene()
{
    for (VkImageView v : texture_views) {
        dev.dt.destroyImageView(dev.hdl, v, nullptr);
    }

    dev.dt.destroyAccelerationStructureKHR(dev.hdl, tlas.accelerationStructure,
                                           nullptr);

    dev.dt.freeMemory(dev.hdl, tlas.memory, nullptr);

    for (auto &blas : blases) {
        dev.dt.destroyAccelerationStructureKHR(dev.hdl,
                                               blas.accelerationStructure,
                                               nullptr);

        dev.dt.freeMemory(dev.hdl, blas.memory, nullptr);
    }
}

shared_ptr<Texture> LoaderState::loadTexture(const vector<uint8_t> &raw)
{
    // FIXME
    (void)raw;
    return nullptr;
}

template <typename MaterialParamsType>
shared_ptr<Material> LoaderState::makeMaterial(MaterialParamsType params)
{
    return MaterialImpl<MaterialParamsType>::make(move(params));
}

std::shared_ptr<Mesh> LoaderState::loadMesh(string_view geometry_path)
{
    return impl_.loadMesh(geometry_path);
}

template <typename VertexType>
shared_ptr<Mesh> LoaderState::makeMesh(vector<VertexType> vertices,
                                       vector<uint32_t> indices)
{
    return makeSharedMesh(move(vertices), move(indices));
}

}

#include "loader_instantiations.inl"
