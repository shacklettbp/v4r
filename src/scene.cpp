#include "scene.hpp"
#include <vulkan/vulkan_core.h>

#include "asset_load.hpp"
#include "shader.hpp"
#include "utils.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_map>

#include "loader_definitions.inl"

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

static RTEnvironmentState makeRTEnv(const DeviceState &dev,
    MemoryAllocator &alloc, const shared_ptr<Scene> &scene,
    DescriptorSet &&rt_desc_set, VkCommandBuffer build_cmd,
    const QueueState &build_queue)
{
    uint32_t num_instances = 0;
    for (const auto &transforms : scene->envDefaults.transforms) {
        num_instances += transforms.size();
    }

    HostBuffer instance_buf = alloc.makeAccelerationStructureInstanceBuffer(
        num_instances * sizeof(VkAccelerationStructureInstanceKHR));

    uint32_t cur_instance = 0;
    VkAccelerationStructureInstanceKHR *instances =
        reinterpret_cast<VkAccelerationStructureInstanceKHR  *>(
            instance_buf.ptr);
    for (uint32_t model_idx = 0;
         model_idx < scene->envDefaults.transforms.size();
         model_idx++) {
        const auto &transforms = scene->envDefaults.transforms[model_idx];
        for (const glm::mat4x3 &txfm : transforms) {
            VkAccelerationStructureInstanceKHR &cur_inst =
                instances[cur_instance];

            memcpy(&cur_inst.transform,
                   glm::value_ptr(glm::transpose(txfm)),
                   sizeof(VkTransformMatrixKHR));

            cur_inst.instanceCustomIndex = scene->meshMetadata[model_idx].indexOffset;
            cur_inst.mask = 0xff;
            cur_inst.instanceShaderBindingTableRecordOffset = 0;
            cur_inst.flags = 0;
            cur_inst.accelerationStructureReference =
                scene->blases->accelStructs[model_idx].devAddr;

            cur_instance++;
        }
    }
    assert(cur_instance == num_instances);

    instance_buf.flush(dev);

    VkBufferDeviceAddressInfo inst_addr_info {
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        nullptr,
        instance_buf.buffer,
    };
    VkDeviceAddress instance_data_addr = 
        dev.dt.getBufferDeviceAddressKHR(dev.hdl, &inst_addr_info);

    VkAccelerationStructureGeometryKHR tlas_geometry;
    tlas_geometry.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlas_geometry.pNext = nullptr;
    tlas_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlas_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto &tlas_instances = tlas_geometry.geometry.instances;
    tlas_instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlas_instances.pNext = nullptr;
    tlas_instances.arrayOfPointers = false;
    tlas_instances.data.deviceAddress = instance_data_addr;

    VkAccelerationStructureBuildGeometryInfoKHR build_info;
    build_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.pNext = nullptr;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build_info.flags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.srcAccelerationStructure = VK_NULL_HANDLE;
    build_info.dstAccelerationStructure = VK_NULL_HANDLE;
    build_info.geometryCount = 1;
    build_info.pGeometries = &tlas_geometry;
    build_info.ppGeometries = nullptr;
    build_info.scratchData.deviceAddress = VK_NULL_HANDLE;

    VkAccelerationStructureBuildSizesInfoKHR size_info;
    size_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    size_info.pNext = nullptr;

    dev.dt.getAccelerationStructureBuildSizesKHR(dev.hdl,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &num_instances, &size_info);

    optional<LocalBuffer> accel_mem_opt =
        alloc.makeAccelerationStructureBuffer(size_info.accelerationStructureSize);

    optional<LocalBuffer> scratch_mem_opt =
        alloc.makeAccelerationStructureScratchBuffer(size_info.buildScratchSize);

    LocalBuffer &accel_mem = accel_mem_opt.value();
    LocalBuffer &scratch_mem = scratch_mem_opt.value();

    VkBufferDeviceAddressInfoKHR scratch_addr_info;
    scratch_addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    scratch_addr_info.pNext = nullptr;
    scratch_addr_info.buffer = scratch_mem.buffer;
    VkDeviceAddress scratch_addr =
        dev.dt.getBufferDeviceAddressKHR(dev.hdl, &scratch_addr_info);

    VkAccelerationStructureCreateInfoKHR create_info;
    create_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    create_info.pNext = nullptr;
    create_info.createFlags = 0;
    create_info.buffer = accel_mem.buffer;
    create_info.offset = 0;
    create_info.size = size_info.accelerationStructureSize;
    create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    create_info.deviceAddress = VK_NULL_HANDLE;

    VkAccelerationStructureKHR tlas;
    REQ_VK(dev.dt.createAccelerationStructureKHR(dev.hdl, &create_info,
                                                 nullptr, &tlas));

    build_info.dstAccelerationStructure = tlas;
    build_info.scratchData.deviceAddress = scratch_addr;

    VkAccelerationStructureBuildRangeInfoKHR range_info;
    range_info.primitiveCount = num_instances;
    range_info.primitiveOffset = 0;
    range_info.firstVertex = 0;
    range_info.transformOffset = 0;
    const auto *range_info_ptr = &range_info;

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(build_cmd, &begin_info));
    dev.dt.cmdBuildAccelerationStructuresKHR(build_cmd, 1, &build_info,
                                             &range_info_ptr);
    REQ_VK(dev.dt.endCommandBuffer(build_cmd));

    VkSubmitInfo build_submit {};
    build_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    build_submit.commandBufferCount = 1;
    build_submit.pCommandBuffers = &build_cmd;

    VkFence build_fence = makeFence(dev);

    build_queue.submit(dev, 1, &build_submit, build_fence);

    waitForFenceInfinitely(dev, build_fence);
    dev.dt.destroyFence(dev.hdl, build_fence, nullptr);

    VkWriteDescriptorSetAccelerationStructureKHR accel_desc_info;
    accel_desc_info.sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    accel_desc_info.pNext = nullptr;
    accel_desc_info.accelerationStructureCount = 1;
    accel_desc_info.pAccelerationStructures = &tlas;

    VkWriteDescriptorSet tlas_set_update;
    tlas_set_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    tlas_set_update.pNext = &accel_desc_info;
    tlas_set_update.dstSet = rt_desc_set.hdl;
    tlas_set_update.dstBinding = 0;
    tlas_set_update.dstArrayElement = 0;
    tlas_set_update.descriptorCount = 1;
    tlas_set_update.descriptorType =
        VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    tlas_set_update.pImageInfo = nullptr;
    tlas_set_update.pBufferInfo = nullptr;
    tlas_set_update.pTexelBufferView = nullptr;
    dev.dt.updateDescriptorSets(dev.hdl, 1,
                                &tlas_set_update, 0, nullptr);

    return RTEnvironmentState {
        tlas,
        move(rt_desc_set),
        move(accel_mem),
    };
}

EnvironmentState::EnvironmentState(const DeviceState &d,
    MemoryAllocator &alloc, const shared_ptr<Scene> &s,
    const glm::mat4 &proj, bool enable_rt, DescriptorSet &&rt_desc_set,
    VkCommandBuffer build_cmd, const QueueState &build_queue)
    : scene(s),
      projection(proj),
      frustumBounds(computeFrustumBounds(proj)),
      reverseIDMap(scene->envDefaults.reverseIDMap),
      freeIDs(),
      lights(s->envDefaults.lights),
      freeLightIDs(),
      lightIDs(s->envDefaults.lightIDs),
      lightReverseIDs(s->envDefaults.lightReverseIDs),
      dev(d),
      rtState(enable_rt ? make_optional(
              makeRTEnv(dev, alloc, s, move(rt_desc_set),
                        build_cmd, build_queue)) :
              optional<RTEnvironmentState>())
{}

EnvironmentState::EnvironmentState(EnvironmentState &&o)
    : scene(move(o.scene)),
      projection(move(o.projection)),
      frustumBounds(move(o.frustumBounds)),
      reverseIDMap(move(o.reverseIDMap)),
      freeIDs(move(o.freeIDs)),
      lights(move(o.lights)),
      freeLightIDs(move(o.freeLightIDs)),
      lightIDs(move(o.lightIDs)),
      lightReverseIDs(move(o.lightReverseIDs)),
      dev(o.dev),
      rtState(move(o.rtState))
{
    o.rtState.reset();
}

EnvironmentState::~EnvironmentState()
{
    if (rtState.has_value()) {
        dev.dt.destroyAccelerationStructureKHR(dev.hdl, rtState->tlas,
                                               nullptr);
    }
}

TextureData::TextureData(const DeviceState &d,
                         MemoryAllocator &a)
    : textures(),
      views(),
      memory(MemoryChunk {
          VK_NULL_HANDLE,
          0,
          0,
      }),
      dev(d),
      alloc(a)
{}

TextureData::TextureData(TextureData &&o)
    : textures(move(o.textures)),
      views(move(o.views)),
      memory(move(o.memory)),
      dev(o.dev),
      alloc(o.alloc)
{
    o.memory.hdl = VK_NULL_HANDLE;
}

TextureData::~TextureData()
{
    for (VkImageView view : views) {
        dev.dt.destroyImageView(dev.hdl, view, nullptr);
    }

    for (LocalTexture &texture : textures) {
        alloc.destroyTexture(move(texture));
    }

    if (memory.hdl != VK_NULL_HANDLE) {
        alloc.free(memory);
    }
}

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

    total_bytes = mesh_info_offset + (sizeof(MeshCullInfo) * meshes.size());

    HostBuffer staging = alloc.makeStagingBuffer(total_bytes);

    // Copy all vertices
    uint32_t vertex_offset = 0;
    uint8_t *staging_start = reinterpret_cast<uint8_t *>(staging.ptr);
    uint8_t *cur_ptr = staging_start;
    // MeshCullInfo *mesh_infos = reinterpret_cast<MeshCullInfo *>(
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
                0,
                0,
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
            {},
    };
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

template <typename VertexType, typename MaterialParamsType>
static SceneDescription parseScene(string_view scene_path,
                                   const glm::mat4 &coordinate_txfm)
{
    if (isGLTF(scene_path)) {
        return parseGLTFScene<VertexType, MaterialParamsType>(scene_path,
                                                              coordinate_txfm);
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
    if constexpr (is_same_v<MaterialParamsType, NoMaterial>) {
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
                          EnvironmentInit(move(instances), {}, hdr.numMeshes)};
}

template <typename VertexType, typename MaterialParamsType>
LoaderImpl LoaderImpl::create()
{
    return {
        v4r::stageScene<VertexType>,
        v4r::parseScene<VertexType, MaterialParamsType>,
        v4r::loadMesh<VertexType>,
        v4r::loadPreprocessedScene<VertexType, MaterialParamsType>,
        sizeof(VertexType),
    };
}

LoaderState::LoaderState(
    const DeviceState &d,
    const LoaderImpl &impl,
    const VkDescriptorSetLayout &scene_set_layout,
    DescriptorManager::MakePoolType make_scene_pool,
    const VkDescriptorSetLayout &mesh_cull_scene_set_layout,
    DescriptorManager::MakePoolType make_mesh_cull_scene_pool,
    const VkDescriptorSetLayout &rt_scene_set_layout,
    DescriptorManager::MakePoolType make_rt_scene_pool,
    MemoryAllocator &alc,
    const QueueState &transfer_queue,
    const QueueState &gfx_queue,
    const glm::mat4 &coordinate_transform,
    const QueueState *compute_queue,
    bool need_acceleration_structure)
    : dev(d),
      transferQueue(transfer_queue),
      gfxQueue(gfx_queue),
      gfxPool(makeCmdPool(dev, dev.gfxQF)),
      gfxCopyCommand(makeCmdBuffer(dev, gfxPool)),
      transferPool(makeCmdPool(dev, dev.transferQF)),
      transferStageCommand(makeCmdBuffer(dev, transferPool)),
      ownershipSemaphore(makeBinarySemaphore(dev)),
      fence(makeFence(dev)),
      alloc(alc),
      descriptorManager(dev, scene_set_layout, make_scene_pool),
      cullDescriptorManager(dev,
                            mesh_cull_scene_set_layout,
                            make_mesh_cull_scene_pool),
      rtDescriptorManager(dev,
                          rt_scene_set_layout,
                          make_rt_scene_pool),
      coordinateTransform(coordinate_transform),
      computeQueue(compute_queue),
      computePool(need_acceleration_structure ?
                  makeCmdPool(dev, dev.computeQF) : VK_NULL_HANDLE),
      asBuildCommand(need_acceleration_structure ?
                     makeCmdBuffer(dev, computePool) : VK_NULL_HANDLE),
      accelBuildSemaphore(need_acceleration_structure ?
                          makeBinarySemaphore(dev) : VK_NULL_HANDLE),
      impl_(impl),
      need_acceleration_structure_(need_acceleration_structure)
{}

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

static optional<tuple<BLASData, LocalBuffer, VkDeviceSize>> makeBLASes(
    const DeviceState &dev, MemoryAllocator &alloc, uint32_t vertex_size,
    const vector<AccelBuildInfo> &geometry, VkCommandBuffer build_cmd)
{
    vector<VkAccelerationStructureGeometryKHR> geo_infos;
    vector<VkAccelerationStructureBuildGeometryInfoKHR> build_infos;
    vector<tuple<VkDeviceSize, VkDeviceSize, VkDeviceSize>> memory_locs;
    geo_infos.reserve(geometry.size());
    build_infos.reserve(geometry.size());
    memory_locs.reserve(geometry.size());

    VkDeviceSize total_scratch_bytes = 0;
    VkDeviceSize total_accel_bytes = 0;

    for (const AccelBuildInfo &src_info : geometry) {
        VkAccelerationStructureGeometryKHR geo_info;
        geo_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geo_info.pNext = nullptr;
        geo_info.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geo_info.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        auto &tri_info = geo_info.geometry.triangles;
        tri_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tri_info.pNext = nullptr;
        tri_info.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tri_info.vertexData.deviceAddress = src_info.vertexAddress;
        tri_info.vertexStride = vertex_size;
        tri_info.maxVertex = src_info.maxVertexIndex;
        tri_info.indexType = VK_INDEX_TYPE_UINT32;
        tri_info.indexData.deviceAddress = src_info.indexAddress;
        tri_info.transformData.deviceAddress = VK_NULL_HANDLE;

        geo_infos.push_back(geo_info);

        VkAccelerationStructureBuildGeometryInfoKHR build_info;
        build_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build_info.pNext = nullptr;
        build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build_info.flags =
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        build_info.mode =
            VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_info.srcAccelerationStructure = VK_NULL_HANDLE;
        build_info.dstAccelerationStructure = VK_NULL_HANDLE;
        build_info.geometryCount = 1;
        build_info.pGeometries = &geo_infos.back();
        build_info.ppGeometries = nullptr;
        // Set device address to 0 before space calculation 
        build_info.scratchData.deviceAddress = VK_NULL_HANDLE;
        build_infos.push_back(build_info);

        VkAccelerationStructureBuildSizesInfoKHR size_info;
        size_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        size_info.pNext = nullptr;

        dev.dt.getAccelerationStructureBuildSizesKHR(
            dev.hdl, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &build_infos.back(),
            &src_info.numTriangles,
            &size_info);

        // Must be aligned to 256 as per spec
        total_accel_bytes = alignOffset(total_accel_bytes, 256);

        memory_locs.emplace_back(total_scratch_bytes, total_accel_bytes,
                                 size_info.accelerationStructureSize);

        total_scratch_bytes += size_info.buildScratchSize;
        total_accel_bytes += size_info.accelerationStructureSize;
    }

    optional<LocalBuffer> scratch_mem_opt =
        alloc.makeAccelerationStructureScratchBuffer(
            total_scratch_bytes);

    optional<LocalBuffer> accel_mem_opt =
        alloc.makeAccelerationStructureBuffer(
            total_accel_bytes);

    if (!scratch_mem_opt.has_value() || !accel_mem_opt.has_value()) {
        return {};
    }

    LocalBuffer &scratch_mem = scratch_mem_opt.value();
    LocalBuffer &accel_mem = accel_mem_opt.value();

    VkBufferDeviceAddressInfoKHR scratch_addr_info;
    scratch_addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    scratch_addr_info.pNext = nullptr;
    scratch_addr_info.buffer = scratch_mem.buffer;
    VkDeviceAddress scratch_base_addr =
        dev.dt.getBufferDeviceAddressKHR(dev.hdl, &scratch_addr_info);

    vector<BLAS> accel_structs;
    vector<VkAccelerationStructureBuildRangeInfoKHR> range_infos;
    vector<VkAccelerationStructureBuildRangeInfoKHR *> range_info_ptrs;

    accel_structs.reserve(geometry.size());
    range_infos.reserve(build_infos.size());
    range_info_ptrs.reserve(build_infos.size());

    for (uint32_t blas_idx = 0; blas_idx < geometry.size(); blas_idx++) {
        VkAccelerationStructureCreateInfoKHR create_info;
        create_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.pNext = nullptr;
        create_info.createFlags = 0;
        create_info.buffer = accel_mem.buffer;
        create_info.offset = get<1>(memory_locs[blas_idx]);
        create_info.size = get<2>(memory_locs[blas_idx]);
        create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        create_info.deviceAddress = VK_NULL_HANDLE;

        VkAccelerationStructureKHR blas;
        REQ_VK(dev.dt.createAccelerationStructureKHR(dev.hdl, &create_info,
                                                     nullptr, &blas));

        auto &build_info = build_infos[blas_idx];
        build_info.dstAccelerationStructure = blas;
        build_info.scratchData.deviceAddress =
            scratch_base_addr + get<0>(memory_locs[blas_idx]);

        VkAccelerationStructureDeviceAddressInfoKHR addr_info;
        addr_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addr_info.pNext = nullptr;
        addr_info.accelerationStructure = blas;

        VkDeviceAddress dev_addr = 
            dev.dt.getAccelerationStructureDeviceAddressKHR(dev.hdl, &addr_info);

        accel_structs.emplace_back(BLAS {blas, dev_addr});

        VkAccelerationStructureBuildRangeInfoKHR range_info;
        range_info.primitiveCount = geometry[blas_idx].numTriangles;
        range_info.primitiveOffset = 0;
        range_info.firstVertex = 0;
        range_info.transformOffset = 0;
        range_infos.push_back(range_info);
        range_info_ptrs.push_back(&range_infos.back());
    }

    dev.dt.cmdBuildAccelerationStructuresKHR(build_cmd,
        build_infos.size(), build_infos.data(), range_info_ptrs.data());

    return make_tuple(
        BLASData {
            dev,
            move(accel_structs),
            move(accel_mem),
        },
        move(scratch_mem),
        total_accel_bytes);
}

BLASData::~BLASData()
{
    for (const auto &blas : accelStructs) {
        dev.dt.destroyAccelerationStructureKHR(dev.hdl, blas.accelStruct,
                                               nullptr);
    }
}

shared_ptr<Scene> LoaderState::makeScene(SceneLoadInfo load_info)
{
    auto &[staged, material_metadata, env_init] = load_info;

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

    const uint32_t num_textures = cpu_textures.size();

    TextureData texture_store(dev, alloc);

    vector<LocalTexture> &gpu_textures = texture_store.textures;
    vector<VkImageView> &texture_views = texture_store.views;
    vector<VkDeviceSize> texture_offsets;

    gpu_textures.reserve(num_textures);
    texture_views.reserve(num_textures);
    texture_offsets.reserve(num_textures);

    // FIXME - custom loader or hacked loader that makes doing this mip
    // level by mip level possible
    VkDeviceSize cpu_texture_bytes = 0;
    VkDeviceSize gpu_texture_bytes = 0;
    for (const shared_ptr<Texture> &texture : cpu_textures) {
        ktxTexture *ktx = texture->data;

        for (uint32_t level = 0; level < texture->numLevels; level++) {
            cpu_texture_bytes += ktxTexture_GetImageSize(ktx, level);
        }

        auto [gpu_texture, reqs] = alloc.makeTexture(
            texture->width, texture->height, texture->numLevels);

        gpu_texture_bytes = alignOffset(gpu_texture_bytes, reqs.alignment);

        texture_offsets.emplace_back(gpu_texture_bytes);
        gpu_textures.emplace_back(move(gpu_texture));

        gpu_texture_bytes += reqs.size;
    }

    optional<MemoryChunk> texture_memory; 
    optional<HostBuffer> texture_staging;

    if (num_textures > 0) {
        texture_memory = alloc.alloc(gpu_texture_bytes);
        
        if (!texture_memory.has_value()) {
            cerr << "Out of memory, failed to allocate texture storage"
                 << endl;
            fatalExit();
        }
        texture_store.memory = texture_memory.value();

        texture_staging.emplace(alloc.makeStagingBuffer(cpu_texture_bytes));
    }

    // Copy all geometry into single buffer

    optional<LocalBuffer> data_opt = alloc.makeLocalBuffer(staged.hdr.totalBytes);

    if (!data_opt.has_value()) {
        cerr << "Out of memory, failed to allocate geometry storage"
             << endl;
        fatalExit();
    }

    LocalBuffer data = move(data_opt.value());

    // Bind image memory and create views
    for (uint32_t i = 0; i < num_textures; i++) {
        LocalTexture &gpu_texture = gpu_textures[i];
        VkDeviceSize offset = texture_offsets[i];

        REQ_VK(dev.dt.bindImageMemory(dev.hdl, gpu_texture.image,
                                      texture_store.memory.hdl,
                                      offset));

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
        const LocalTexture &gpu_texture = gpu_textures[i];
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
            const LocalTexture &gpu_texture = gpu_textures[i];
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

    array<VkBufferMemoryBarrier, 2> geometry_barriers;
    geometry_barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    geometry_barriers[0].pNext = nullptr;
    geometry_barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    geometry_barriers[0].dstAccessMask = 0;
    geometry_barriers[0].srcQueueFamilyIndex = dev.transferQF;

    if (need_acceleration_structure_) {
        geometry_barriers[0].dstQueueFamilyIndex = dev.computeQF;
    } else {
        geometry_barriers[0].dstQueueFamilyIndex = dev.gfxQF;
    }

    geometry_barriers[0].buffer = data.buffer;
    geometry_barriers[0].offset = 0;
    geometry_barriers[0].size = staged.hdr.totalBytes;

    // Geometry & texture barrier execute.
    dev.dt.cmdPipelineBarrier(
        transferStageCommand, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
        1, geometry_barriers.data(),
        texture_barriers.size(), texture_barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(transferStageCommand));

    VkSubmitInfo copy_submit {};
    copy_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    copy_submit.waitSemaphoreCount = 0;
    copy_submit.pWaitSemaphores = nullptr;
    copy_submit.pWaitDstStageMask = nullptr;
    copy_submit.commandBufferCount = 1;
    copy_submit.pCommandBuffers = &transferStageCommand;
    copy_submit.signalSemaphoreCount = 1;
    copy_submit.pSignalSemaphores = &ownershipSemaphore;

    transferQueue.submit(dev, 1, &copy_submit, VK_NULL_HANDLE);

    optional<BLASData> blas_store;
    optional<LocalBuffer> blas_scratch;
    if (need_acceleration_structure_) {
        VkBufferDeviceAddressInfo addr_info;
        addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addr_info.pNext = nullptr;
        addr_info.buffer = data.buffer;
        VkDeviceAddress geometry_addr =
            dev.dt.getBufferDeviceAddressKHR(dev.hdl, &addr_info);
    
        // FIXME, each AccelBuildInfo has maxVertexIndex set to the number
        // of vertices in the entire scene, because each mesh is reindexed
        // into the global buffer. Probably results in massive waste of memory
        // during build - globally indexed meshes are nice in the RT shaders
        // but aren't necessary here.
        vector<AccelBuildInfo> geo_info;
        geo_info.reserve(staged.meshMetadata.size());
        for (const auto &mesh_info : staged.meshMetadata) {
            geo_info.push_back(AccelBuildInfo {
                geometry_addr,
                geometry_addr + staged.hdr.indexOffset +
                    mesh_info.indexOffset * sizeof(uint32_t),
                mesh_info.numTriangles, 
                staged.hdr.numVertices,
            });
        }

        REQ_VK(dev.dt.beginCommandBuffer(asBuildCommand, &begin_info));

        geometry_barriers[0].srcAccessMask = 0;
        geometry_barriers[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

        // Acquire geometry on compute queue for acceleration structure build
        dev.dt.cmdPipelineBarrier(asBuildCommand,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            0, 0, nullptr, 1, geometry_barriers.data(),
            0, nullptr);

        auto blas_result = makeBLASes(dev, alloc, impl_.vertexSize,
            geo_info, asBuildCommand);

        if (!blas_result.has_value()) {
            cerr << "OOM while constructing BLASes" << endl;
            fatalExit();
        }

        blas_store.emplace(move(get<0>(blas_result.value())));
        blas_scratch.emplace(move(get<1>(blas_result.value())));

        geometry_barriers[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        geometry_barriers[0].dstAccessMask = 0;
        geometry_barriers[0].srcQueueFamilyIndex = dev.computeQF;
        geometry_barriers[0].dstQueueFamilyIndex = dev.gfxQF;

        geometry_barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        geometry_barriers[1].pNext = nullptr;
        geometry_barriers[1].srcAccessMask =
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        geometry_barriers[1].dstAccessMask = 0;
        geometry_barriers[1].srcQueueFamilyIndex = dev.computeQF;
        geometry_barriers[1].dstQueueFamilyIndex = dev.gfxQF;
        geometry_barriers[1].buffer = blas_store->storage.buffer;
        geometry_barriers[1].offset = 0;
        geometry_barriers[1].size = get<2>(blas_result.value());

        // Transfer geometry and acceleration structure to graphics queue
        dev.dt.cmdPipelineBarrier(asBuildCommand,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            0, 0, nullptr,
            2, geometry_barriers.data(),
            0, nullptr);

        REQ_VK(dev.dt.endCommandBuffer(asBuildCommand));

        VkPipelineStageFlags sema_wait_mask =
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        VkSubmitInfo compute_submit {};
        compute_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        compute_submit.waitSemaphoreCount = 1;
        compute_submit.pWaitSemaphores = &ownershipSemaphore;
        compute_submit.pWaitDstStageMask = &sema_wait_mask;
        compute_submit.commandBufferCount = 1;
        compute_submit.pCommandBuffers = &asBuildCommand;
        compute_submit.signalSemaphoreCount = 1;
        compute_submit.pSignalSemaphores = &accelBuildSemaphore;
        computeQueue->submit(dev, 1, &compute_submit, VK_NULL_HANDLE);
    }

    // Start recording for graphics queue
    REQ_VK(dev.dt.beginCommandBuffer(gfxCopyCommand, &begin_info));

    // Finish moving geometry onto graphics queue family
    // geometry and textures need separate barriers due to different
    // dependent stages
    uint32_t num_geometry_barriers;
    geometry_barriers[0].srcAccessMask = 0;
    if (need_acceleration_structure_) {
        geometry_barriers[0].dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
            VK_ACCESS_INDEX_READ_BIT;
        geometry_barriers[1].srcAccessMask = 0;
        geometry_barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        num_geometry_barriers = 2;
    } else {
        geometry_barriers[0].dstAccessMask =
            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;

        num_geometry_barriers = 1;
    }

    VkPipelineStageFlags dst_geo_gfx_stage =
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;

    if (need_acceleration_structure_) {
        dst_geo_gfx_stage |= VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    }

    dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              dst_geo_gfx_stage,
                              0,
                              0, nullptr,
                              num_geometry_barriers, geometry_barriers.data(),
                              0, nullptr);

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

    REQ_VK(dev.dt.endCommandBuffer(gfxCopyCommand));

    VkSubmitInfo gfx_submit {};
    gfx_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    gfx_submit.waitSemaphoreCount = 1;
    if (need_acceleration_structure_) {
        gfx_submit.pWaitSemaphores = &accelBuildSemaphore;
    } else {
        gfx_submit.pWaitSemaphores = &ownershipSemaphore;
    }
    VkPipelineStageFlags sema_wait_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    gfx_submit.pWaitDstStageMask = &sema_wait_mask;
    gfx_submit.commandBufferCount = 1;
    gfx_submit.pCommandBuffers = &gfxCopyCommand;

    gfxQueue.submit(dev, 1, &gfx_submit, fence);

    waitForFenceInfinitely(dev, fence);
    resetFence(dev, fence);

    assert(material_metadata.numMaterials <= VulkanConfig::max_materials);

    DescriptorSet material_set = descriptorManager.makeSet();
    DescriptorSet cull_set = cullDescriptorManager.makeSet();
    optional<DescriptorSet> rt_set;
    if (need_acceleration_structure_) {
        rt_set.emplace(rtDescriptorManager.makeSet());
    }

    vector<VkWriteDescriptorSet> desc_updates;
    desc_updates.reserve(2 * material_metadata.texturesPerMaterial + 4);

    VkDescriptorBufferInfo material_buffer_info;
    vector<VkDescriptorImageInfo> descriptor_views;

#if 0
    // FIXME null descriptorManager feels a bit indirect
    if (material_set.hdl != VK_NULL_HANDLE &&
        material_metadata.numMaterials > 0) {
        // If there are textures the layout is
        // 0: sampler
        // 1 .. # textures: texture arrays
        // Final: material params
        descriptor_views.reserve(material_metadata.numMaterials *
                                 material_metadata.texturesPerMaterial);

        for (uint32_t material_texture_idx = 0;
             material_texture_idx < material_metadata.texturesPerMaterial;
             material_texture_idx++) {
            for (uint32_t mat_idx = 0;
                 mat_idx < material_metadata.numMaterials; mat_idx++) {
                VkImageView view = texture_views[material_metadata.textureIndices
                         [mat_idx * material_metadata.texturesPerMaterial +
                          material_texture_idx]];

                descriptor_views.push_back(
                    {VK_NULL_HANDLE,  // Immutable
                     view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            }
            VkWriteDescriptorSet desc_update;
            desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            desc_update.pNext = nullptr;
            desc_update.dstSet = material_set.hdl;
            desc_update.dstBinding = 1 + material_texture_idx;
            desc_update.dstArrayElement = 0;
            desc_update.descriptorCount = material_metadata.numMaterials;
            desc_update.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            desc_update.pImageInfo =
                descriptor_views.data() +
                material_texture_idx * material_metadata.numMaterials;
            desc_update.pBufferInfo = nullptr;
            desc_update.pTexelBufferView = nullptr;

            desc_updates.push_back(desc_update);

            if (need_acceleration_structure_) {
                desc_update.dstSet = rt_set->hdl;
                desc_update.dstBinding = 3 + material_texture_idx;

                desc_updates.push_back(desc_update);
            }
        }

        if (staged.hdr.materialBytes > 0) {
            uint32_t param_binding = 0;
            if (material_metadata.texturesPerMaterial > 0) {
                param_binding = 1 + material_metadata.texturesPerMaterial;
            }

            material_buffer_info.buffer = data.buffer;
            material_buffer_info.offset = staged.hdr.materialOffset;
            material_buffer_info.range = staged.hdr.materialBytes;

            VkWriteDescriptorSet desc_update;
            desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            desc_update.pNext = nullptr;
            desc_update.dstSet = material_set.hdl;
            desc_update.dstBinding = param_binding;
            desc_update.dstArrayElement = 0;
            desc_update.descriptorCount = 1;
            desc_update.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            desc_update.pImageInfo = nullptr;
            desc_update.pBufferInfo = &material_buffer_info;
            desc_update.pTexelBufferView = nullptr;

            desc_updates.push_back(desc_update);

            if (need_acceleration_structure_) {
                desc_update.dstSet = rt_set->hdl;
                desc_update.dstBinding = 2 + param_binding;
                desc_updates.push_back(desc_update);
            }
        }
    }

    VkDescriptorBufferInfo mesh_chunk_buffer_info;
    mesh_chunk_buffer_info.buffer = data.buffer;
    mesh_chunk_buffer_info.offset = staged.hdr.meshChunkOffset;
    mesh_chunk_buffer_info.range = staged.hdr.meshChunkBytes;

    VkWriteDescriptorSet cull_desc_update;
    cull_desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    cull_desc_update.pNext = nullptr;
    cull_desc_update.dstSet = cull_set.hdl;
    cull_desc_update.dstBinding = 0;
    cull_desc_update.dstArrayElement = 0;
    cull_desc_update.descriptorCount = 1;
    cull_desc_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    cull_desc_update.pImageInfo = nullptr;
    cull_desc_update.pBufferInfo = &mesh_chunk_buffer_info;
    cull_desc_update.pTexelBufferView = nullptr;
    desc_updates.push_back(cull_desc_update);
#endif

    VkDescriptorBufferInfo vertex_buffer_info;
    VkDescriptorBufferInfo index_buffer_info;
    if (need_acceleration_structure_) {
        vertex_buffer_info.buffer = data.buffer;
        vertex_buffer_info.offset = 0;
        vertex_buffer_info.range = staged.hdr.numVertices * impl_.vertexSize;

        VkWriteDescriptorSet rt_desc_update;
        rt_desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        rt_desc_update.pNext = nullptr;
        rt_desc_update.dstSet = rt_set->hdl;
        rt_desc_update.dstBinding = 0;
        rt_desc_update.dstArrayElement = 0;
        rt_desc_update.descriptorCount = 1;
        rt_desc_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        rt_desc_update.pImageInfo = nullptr;
        rt_desc_update.pBufferInfo = &vertex_buffer_info;
        rt_desc_update.pTexelBufferView = nullptr;
        desc_updates.push_back(rt_desc_update);

        index_buffer_info.buffer = data.buffer;
        index_buffer_info.offset = staged.hdr.indexOffset;
        index_buffer_info.range = staged.hdr.numIndices * sizeof(uint32_t);

        rt_desc_update.dstBinding = 1;
        rt_desc_update.pBufferInfo = &index_buffer_info;
        desc_updates.push_back(rt_desc_update);
    }

    dev.dt.updateDescriptorSets(dev.hdl, desc_updates.size(),
                                desc_updates.data(), 0, nullptr);

    return make_shared<Scene>(Scene {
        move(texture_store),
        move(material_set),
        move(cull_set),
        move(data),
        staged.hdr.indexOffset,
        move(staged.meshMetadata),
        staged.hdr.numMeshes,
        move(env_init),
        move(rt_set),
        move(blas_store),
    });
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
