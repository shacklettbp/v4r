#include "scene.hpp"
#include "loader_definitions.inl"

#include "asset_load.hpp"
#include "shader.hpp"
#include "utils.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace std;

namespace v4r {

Texture::~Texture()
{
    ktxTexture_Destroy(data);
}

EnvironmentInit::EnvironmentInit(
        const vector<pair<uint32_t, InstanceProperties>> &instances,
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
        auto &[mesh_idx, inst] = instances[cur_id];

        uint32_t inst_idx = transforms[mesh_idx].size();

        transforms[mesh_idx].push_back(inst.modelTransform);
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

EnvironmentState::EnvironmentState(const shared_ptr<Scene> &s,
                                   const glm::mat4 &proj)
    : scene(s),
      projection(proj),
      reverseIDMap(scene->envDefaults.reverseIDMap),
      freeIDs(),
      lights(s->envDefaults.lights),
      freeLightIDs(),
      lightIDs(s->envDefaults.lightIDs),
      lightReverseIDs(s->envDefaults.lightReverseIDs)
{}

template <typename VertexType>
static StagedScene stageScene(
        const vector<shared_ptr<Mesh>> &meshes,
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
    if (param_bytes.size() > 0) {
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
    MeshInfo *mesh_infos = reinterpret_cast<MeshInfo *>(
            staging_start + mesh_info_offset);

    for (uint32_t mesh_idx = 0; mesh_idx < meshes.size(); mesh_idx++) {
        const auto &generic_mesh = meshes[mesh_idx];
        auto mesh = static_cast<const MeshT *>(generic_mesh.get());
        VkDeviceSize vertex_bytes = sizeof(VertexType) * mesh->vertices.size();
        memcpy(cur_ptr, mesh->vertices.data(), vertex_bytes);

        mesh_infos[mesh_idx].vertexOffset = vertex_offset;

        cur_ptr += vertex_bytes;
        vertex_offset += mesh->vertices.size();
    }

    // Copy all indices
    uint32_t cur_mesh_index = 0;
    for (uint32_t mesh_idx = 0; mesh_idx < meshes.size(); mesh_idx++) {
        auto mesh = static_cast<const MeshT *>(meshes[mesh_idx].get());

        VkDeviceSize index_bytes =
            sizeof(uint32_t) * mesh->indices.size();
        memcpy(cur_ptr, mesh->indices.data(), index_bytes);

        mesh_infos[mesh_idx].indexOffset = cur_mesh_index;
        mesh_infos[mesh_idx].indexCount = 
            static_cast<uint32_t>(mesh->indices.size());

        cur_ptr += index_bytes;
        cur_mesh_index += mesh->indices.size();
    }

    // Optionally copy material params
    if (param_bytes.size() > 0) {
        memcpy(staging_start + material_offset, param_bytes.data(),
               param_bytes.size());
    }

    staging.flush(dev);

    return { 
        move(staging), 
        total_vertex_bytes,
        material_offset,
        mesh_info_offset,
        static_cast<uint32_t>(meshes.size()),
        total_bytes
    };
}

template <typename VertexType>
VertexMesh<VertexType>::VertexMesh(vector<VertexType> v,
                                   vector<uint32_t> i)
    : Mesh { move(i) },
      vertices(move(v))
{}

template <typename VertexType>
static shared_ptr<Mesh> makeSharedMesh(vector<VertexType> vertices,
                                       vector<uint32_t> indices)
{
    return shared_ptr<VertexMesh<VertexType>>(new VertexMesh<VertexType>(
        move(vertices),
        move(indices)
    ));
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
    auto suffix = gltf_path.substr(gltf_path.find('.') + 1);
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
static SceneDescription parseGLTFScene(string_view scene_path,
                                       const glm::mat4 &coordinate_txfm)
{
    auto raw_scene = gltfLoad(scene_path);

    constexpr bool need_materials = !is_same_v<MaterialParamsType,
                                               NoMaterial>;

    vector<shared_ptr<Material>> materials;
    vector<shared_ptr<Mesh>> geometry;
    geometry.reserve(raw_scene.meshes.size());

    if constexpr (need_materials) {
        materials =
            gltfParseMaterials<MaterialParamsType>(raw_scene);
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
        return parseGLTFScene<VertexType, MaterialParamsType>(
                scene_path, coordinate_txfm);
    } else {
        cerr << "Only GLTF is supported" << endl;
        abort();
    }
}


template <typename VertexType, typename MaterialParamsType>
LoaderImpl LoaderImpl::create()
{
    return {
        v4r::stageScene<VertexType>,
        v4r::parseScene<VertexType, MaterialParamsType>,
        v4r::loadMesh<VertexType>
    };
}

LoaderState::LoaderState(const DeviceState &d,
                         const LoaderImpl &impl,
                         const VkDescriptorSetLayout &scene_set_layout,
                         DescriptorManager::MakePoolType make_scene_pool,
                         const VkDescriptorSetLayout &
                            mesh_cull_scene_set_layout,
                         DescriptorManager::MakePoolType
                            make_mesh_cull_scene_pool,
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
      cullDescriptorManager(dev, mesh_cull_scene_set_layout,
                            make_mesh_cull_scene_pool),
      coordinateTransform(coordinate_transform),
      impl_(impl)
{}

shared_ptr<Scene> LoaderState::loadScene(string_view scene_path)
{
    SceneDescription desc = impl_.parseScene(scene_path,
                                             coordinateTransform);

    return makeScene(desc);
}

struct MaterialsInfo {
    vector<shared_ptr<Texture>> uniqueTextures;
    vector<uint8_t> packedParams;
    unordered_map<const Texture *, size_t> textureIndices;
    vector<size_t> paramOffsets;
};

static MaterialsInfo finalizeMaterials(
        const vector<shared_ptr<Material>> &materials)
{
    vector<shared_ptr<Texture>> textures;
    unordered_map<const Texture *, size_t> texture_tracker;
    vector<size_t> param_offsets;
    param_offsets.reserve(materials.size());

    uint64_t num_material_bytes = 0;
    for (const auto &material : materials) {
        num_material_bytes += material->paramBytes.size();
    }

    vector<uint8_t> packed_params(num_material_bytes);
    uint8_t *cur_param_ptr = packed_params.data();

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
        }

        param_offsets.push_back(cur_param_ptr - packed_params.data());
        cur_param_ptr += material->paramBytes.size();
    }

    return { textures, packed_params, texture_tracker, param_offsets };
}

shared_ptr<Scene> LoaderState::makeScene(
        const SceneDescription &scene_desc)
{
    const auto &materials = scene_desc.getMaterials();

    vector<LocalImage> gpu_textures;

    auto [cpu_textures, material_params, texture_indices, material_offsets] =
        finalizeMaterials(materials);

    // FIXME - custom loader or hacked loader that makes doing this mip
    // level by mip level possible
    VkDeviceSize total_texture_bytes = 0;
    for (const shared_ptr<Texture> &texture : cpu_textures) {
        ktxTexture *ktx = texture->data;
        assert(ktx->classId == ktxTexture2_c);

        ktxTexture2 *ktx2 = reinterpret_cast<ktxTexture2 *>(ktx);
        KTX_error_code res = ktxTexture2_TranscodeBasis(
            ktx2, KTX_TTF_BC7_RGBA, 0);
        ktxCheck(res);

        for (uint32_t level = 0; level < texture->numLevels; level++) {
            total_texture_bytes += ktxTexture_GetImageSize(ktx, level);
        }

        gpu_textures.emplace_back(alloc.makeTexture(texture->width,
                                                    texture->height,
                                                    texture->numLevels));
    }
    const uint32_t num_textures = cpu_textures.size();

    optional<HostBuffer> texture_staging;
    if (num_textures > 0) {
        texture_staging.emplace(alloc.makeStagingBuffer(total_texture_bytes));
    }

    // Copy all geometry into single buffer
    const auto &cpu_meshes = scene_desc.getMeshes();
    
    auto staged = impl_.stageScene(cpu_meshes, material_params, dev, alloc);

    LocalBuffer data = alloc.makeLocalBuffer(staged.totalBytes);

    // Start recording for transfer queue
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(transferStageCommand, &begin_info));

    // Copy vertex/index buffer onto GPU
    VkBufferCopy copy_settings {};
    copy_settings.size = staged.totalBytes;
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
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, gpu_texture.mipLevels,
            0, 1,
        };
    }

    if (num_textures > 0) {
        dev.dt.cmdPipelineBarrier(transferStageCommand,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 0, nullptr, 0, nullptr,
                                  texture_barriers.size(),
                                  texture_barriers.data());

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
                KTX_error_code res =
                    ktxTexture_GetImageOffset(ktx, level, 0, 0,
                                              &ktx_level_offset);
                ktxCheck(res);

                VkDeviceSize num_level_bytes =
                    ktxTexture_GetImageSize(ktx, level);

                memcpy(base_texture_staging + cur_staging_offset,
                       ktx_data + ktx_level_offset, 
                       num_level_bytes);

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
            dev.dt.cmdCopyBufferToImage(transferStageCommand,
                                        texture_staging->buffer,
                                        gpu_texture.image,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        num_levels,
                                        copy_infos.data());
        }

        // Flush staging buffer
        texture_staging->flush(dev);

        // Transfer queue relinquish texture barriers
        for (VkImageMemoryBarrier &barrier : texture_barriers) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;;
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
    geometry_barrier.size = staged.totalBytes;

    // Geometry & texture barrier execute.
    dev.dt.cmdPipelineBarrier(transferStageCommand,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr,
                              1, &geometry_barrier,
                              texture_barriers.size(),
                              texture_barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(transferStageCommand));

    VkSubmitInfo copy_submit{};
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
    geometry_barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                     VK_ACCESS_INDEX_READ_BIT;

    dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                              0, 0, nullptr,
                              1, &geometry_barrier,
                              0, nullptr);

    if (num_textures > 0)  {
        for (VkImageMemoryBarrier &barrier : texture_barriers) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = dev.transferQF;
            barrier.dstQueueFamilyIndex = dev.gfxQF;
        }

        // Finish acquiring mip level 0 on graphics queue and transition layout
        dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                  0, 0, nullptr, 0, nullptr,
                                  texture_barriers.size(),
                                  texture_barriers.data());
    }

    REQ_VK(dev.dt.endCommandBuffer(gfxCopyCommand));

    VkSubmitInfo gfx_submit{};
    gfx_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    gfx_submit.waitSemaphoreCount = 1;
    gfx_submit.pWaitSemaphores = &semaphore;
    VkPipelineStageFlags sema_wait_mask = 
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
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
        view_info.components = { 
            VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
            VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A
        };
        view_info.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, gpu_texture.mipLevels,
            0, 1
        };

        VkImageView view;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &view));

        texture_views.push_back(view);
    }

    assert(materials.size() <= VulkanConfig::max_materials);

    DescriptorSet material_set = descriptorManager.makeSet();

    // FIXME null descriptorManager feels a bit indirect
    if (material_set.hdl != VK_NULL_HANDLE && materials.size() > 0) {
        // If there are textures the layout is
        // 0: sampler
        // 1 .. # textures: texture arrays
        // Final: material params
        vector<VkDescriptorImageInfo> descriptor_views;
        const size_t textures_per_material = materials[0]->textures.size();
        descriptor_views.reserve(materials.size() * textures_per_material);
        vector<VkWriteDescriptorSet> desc_updates;
        desc_updates.reserve(textures_per_material + 1);

        for (size_t material_texture_idx = 0;
             material_texture_idx < textures_per_material;
             material_texture_idx++) {
            for (const auto &material : materials) {
                const auto &texture = material->textures[material_texture_idx];
                VkImageView view =
                    texture_views[texture_indices[texture.get()]];

                descriptor_views.push_back({
                    VK_NULL_HANDLE, // Immutable
                    view,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                });
            }
            VkWriteDescriptorSet desc_update;
            desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            desc_update.pNext = nullptr;
            desc_update.dstSet = material_set.hdl;
            desc_update.dstBinding = 1 + material_texture_idx;
            desc_update.dstArrayElement = 0;
            desc_update.descriptorCount = materials.size();
            desc_update.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            desc_update.pImageInfo = descriptor_views.data() +
                material_texture_idx * materials.size();
            desc_update.pBufferInfo = nullptr;
            desc_update.pTexelBufferView = nullptr;

            desc_updates.push_back(desc_update);
        }

        VkDescriptorBufferInfo material_buffer_info;
        if (material_params.size() > 0) {
            uint32_t param_binding = 0;
            if (textures_per_material > 0) {
                param_binding = 1 + textures_per_material;
            }

            material_buffer_info.buffer = data.buffer;
            material_buffer_info.offset = staged.paramBufferOffset;
            material_buffer_info.range = material_params.size();

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
        }

        dev.dt.updateDescriptorSets(dev.hdl, desc_updates.size(),
                                    desc_updates.data(), 0, nullptr);
    }

    DescriptorSet cull_set = cullDescriptorManager.makeSet();

    VkDescriptorBufferInfo mesh_info_buffer_info;
    mesh_info_buffer_info.buffer = data.buffer;
    mesh_info_buffer_info.offset = staged.meshInfoOffset;
    mesh_info_buffer_info.range =
        sizeof(MeshInfo) * staged.numMeshes;

    VkWriteDescriptorSet cull_desc_update;
    cull_desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    cull_desc_update.pNext = nullptr;
    cull_desc_update.dstSet = cull_set.hdl;
    cull_desc_update.dstBinding = 0;
    cull_desc_update.dstArrayElement = 0;
    cull_desc_update.descriptorCount = 1;
    cull_desc_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    cull_desc_update.pImageInfo = nullptr;
    cull_desc_update.pBufferInfo = &mesh_info_buffer_info;
    cull_desc_update.pTexelBufferView = nullptr;

    dev.dt.updateDescriptorSets(dev.hdl, 1, &cull_desc_update, 0, nullptr);

    return make_shared<Scene>(Scene {
        move(gpu_textures),
        move(texture_views),
        move(material_set),
        move(cull_set),
        move(data),
        staged.indexBufferOffset,
        staged.numMeshes,
        EnvironmentInit(scene_desc.getDefaultInstances(),
                        scene_desc.getDefaultLights(),
                        cpu_meshes.size())
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
