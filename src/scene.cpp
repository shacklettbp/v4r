#include "scene.hpp"

#include "asset_load.hpp"
#include "utils.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace std;

namespace v4r {

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
static StagedMeshes copyMeshesToStaging(
        const vector<shared_ptr<Mesh>> &meshes,
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

    HostBuffer geo_staging = alloc.makeStagingBuffer(total_geometry_bytes);

    vector<InlineMesh> inline_meshes;
    inline_meshes.reserve(meshes.size());

    // Copy all vertices
    uint32_t vertex_offset = 0;
    uint8_t *cur_ptr = reinterpret_cast<uint8_t *>(geo_staging.ptr);

    for (const auto &generic_mesh : meshes) {
        auto mesh = static_cast<const MeshT *>(generic_mesh.get());
        VkDeviceSize vertex_bytes = sizeof(VertexType) * mesh->vertices.size();
        memcpy(cur_ptr, mesh->vertices.data(), vertex_bytes);

        inline_meshes.emplace_back(InlineMesh {
            vertex_offset,
            0,
            0
        });

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

        inline_meshes[mesh_idx].startIndex = cur_mesh_index;
        inline_meshes[mesh_idx].numIndices = 
            static_cast<uint32_t>(mesh->indices.size());

        cur_ptr += index_bytes;
        cur_mesh_index += mesh->indices.size();
    }

    geo_staging.flush(dev);

    return { 
        move(geo_staging), 
        move(inline_meshes),
        total_vertex_bytes,
        total_geometry_bytes
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
static shared_ptr<Mesh> loadMesh(const string &geometry_path)
{
    Assimp::Importer importer;
    int flags = aiProcess_PreTransformVertices | aiProcess_Triangulate;
    const aiScene *raw_scene = importer.ReadFile(geometry_path.c_str(), flags);
    if (!raw_scene) {
        cerr << "Failed to load geometry file " << geometry_path << ": " <<
            importer.GetErrorString() << endl;
        fatalExit();
    }

    if (raw_scene->mNumMeshes == 0) {
        cerr << "No meshes in file " << geometry_path << endl;
        fatalExit();
    }

    // FIXME probably should just union all meshes into one mesh here
    aiMesh *raw_mesh = raw_scene->mMeshes[0];

    auto [vertices, indices] = assimpParseMesh<VertexType>(raw_mesh);

    return makeSharedMesh(move(vertices), move(indices));
}

template <typename MatDescType>
static shared_ptr<Material> makeSharedMaterial(MatDescType description);

template <>
shared_ptr<Material> makeSharedMaterial(
        UnlitRendererInputs::MaterialDescription description)
{
    return shared_ptr<Material>(new Material {
        { move(description.texture) }
    });
}

template <>
shared_ptr<Material> makeSharedMaterial(
        LitRendererInputs::MaterialDescription description)
{
    return shared_ptr<Material>(new Material {
        { move(description.texture) }
    });
}

template <typename VertexType, typename MaterialDescType>
static SceneDescription parseAssimpScene(const string &scene_path,
                                         const glm::mat4 &coordinate_txfm)
{
    Assimp::Importer importer;
    int flags = aiProcess_JoinIdenticalVertices | aiProcess_Triangulate;
    const aiScene *raw_scene = importer.ReadFile(scene_path.c_str(), flags);
    if (!raw_scene) {
        cerr << "Failed to load scene " << scene_path << ": " <<
            importer.GetErrorString() << endl;
        fatalExit();
    }

    vector<shared_ptr<Material>> materials;
    vector<shared_ptr<Mesh>> geometry;
    vector<uint32_t> mesh_materials;

    auto material_params = assimpParseMaterials<MaterialDescType>(raw_scene);
    materials.reserve(material_params.size());
    for (auto &&params : material_params) {
        materials.emplace_back(makeSharedMaterial(move(params)));
    }

    geometry.reserve(raw_scene->mNumMeshes);
    mesh_materials.reserve(raw_scene->mNumMeshes);
    for (uint32_t mesh_idx = 0; mesh_idx < raw_scene->mNumMeshes; mesh_idx++) {
        aiMesh *raw_mesh = raw_scene->mMeshes[mesh_idx];
        mesh_materials.push_back(raw_mesh->mMaterialIndex);

        auto [vertices, indices] = assimpParseMesh<VertexType>(raw_mesh);
        geometry.emplace_back(makeSharedMesh(move(vertices), move(indices)));
    }

    SceneDescription scene_desc(move(geometry), move(materials));

    assimpParseInstances(scene_desc, raw_scene, mesh_materials,
                         coordinate_txfm);

    return scene_desc;
}

template <typename VertexType, typename MatDescType>
LoaderHelper makeHelperPointers()
{
    return {
        copyMeshesToStaging<VertexType>,
        parseAssimpScene<VertexType, MatDescType>,
        loadMesh<VertexType>
    };
}

template <typename InputsType>
LoaderHelper getVertexType(const RenderFeatures &features)
{
    using MatDescType = typename InputsType::MaterialDescription;
    switch (features.colorSrc) {
        case RenderFeatures::MeshColor::None: {
            return makeHelperPointers<typename InputsType::NoColorVertex,
                                      MatDescType>();
        }
        case RenderFeatures::MeshColor::Vertex: {
            if constexpr (is_same_v<InputsType, UnlitRendererInputs>) {
                return makeHelperPointers<typename InputsType::ColoredVertex,
                                          MatDescType>();
            } else {
                unreachable();
            }
        }
        case RenderFeatures::MeshColor::Texture: {
            return makeHelperPointers<typename InputsType::TexturedVertex,
                                      MatDescType>();
        }
    }

    unreachable();
}

LoaderHelper makeLoaderHelper(const RenderFeatures &features)
{
    switch (features.pipeline) {
        case RenderFeatures::Pipeline::Unlit:
            return getVertexType<UnlitRendererInputs>(features);
        case RenderFeatures::Pipeline::Lit:
        case RenderFeatures::Pipeline::Shadowed:
            return getVertexType<LitRendererInputs>(features);
    }

    unreachable();
}

LoaderState::LoaderState(const DeviceState &d,
                         const RenderFeatures &features,
                         const PerSceneDescriptorConfig &scene_desc_cfg,
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
      descriptorManager(dev, scene_desc_cfg.layout),
      coordinateTransform(coordinate_transform),
      assetHelper(makeLoaderHelper(features))
{}

static uint32_t getMipLevels(const Texture &texture)
{
    return static_cast<uint32_t>(
        floor(log2(max(texture.width, texture.height)))) + 1;
}

static void generateMips(const DeviceState &dev,
                         const VkCommandBuffer copy_cmd,
                         const vector<LocalImage> &gpu_textures,
                         DynArray<VkImageMemoryBarrier> &barriers)
{
    for (size_t texture_idx = 0; texture_idx < gpu_textures.size();
            texture_idx++) {
        const LocalImage &gpu_texture = gpu_textures[texture_idx];
        VkImageMemoryBarrier &barrier = barriers[texture_idx];
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        for (uint32_t mip_level = 1; mip_level < gpu_texture.mipLevels;
                mip_level++) {
            VkImageBlit blit_spec {};
            
            // Src
            blit_spec.srcSubresource =
                { VK_IMAGE_ASPECT_COLOR_BIT, mip_level - 1, 0, 1 };
            blit_spec.srcOffsets[1] =
                { static_cast<int32_t>(
                        max(gpu_texture.width >> (mip_level - 1), 1u)),
                  static_cast<int32_t>(
                        max(gpu_texture.height >> (mip_level - 1), 1u)),
                  1 };

            // Dst
            blit_spec.dstSubresource =
                { VK_IMAGE_ASPECT_COLOR_BIT, mip_level, 0, 1 };
            blit_spec.dstOffsets[1] =
                { static_cast<int32_t>(
                        max(gpu_texture.width >> mip_level, 1u)),
                  static_cast<int32_t>(
                        max(gpu_texture.height >> mip_level, 1u)),
                  1 };

            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.subresourceRange.baseMipLevel = mip_level;

            dev.dt.cmdPipelineBarrier(copy_cmd,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      0, 0, nullptr, 0, nullptr,
                                      1, &barrier);

            dev.dt.cmdBlitImage(copy_cmd,
                                gpu_texture.image,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                gpu_texture.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &blit_spec, VK_FILTER_LINEAR);

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

            dev.dt.cmdPipelineBarrier(copy_cmd,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      0, 0, nullptr, 0, nullptr,
                                      1, &barrier);
        }
    }
}

shared_ptr<Scene> LoaderState::loadScene(
        const SceneDescription &scene_desc)
{
    vector<HostBuffer> texture_stagings;
    vector<LocalImage> gpu_textures;

    // Gather textures
    vector<shared_ptr<Texture>> cpu_textures;

    for (const auto &material : scene_desc.getMaterials()) {
        for (const auto & texture : material->textures) {
            cpu_textures.push_back(texture);
        }
    }

    // FIXME pack textures
    for (const shared_ptr<Texture> &texture : cpu_textures) {
        uint64_t texture_bytes = texture->width * texture->height *
            texture->num_channels * sizeof(uint8_t);

        HostBuffer texture_staging = alloc.makeStagingBuffer(texture_bytes);
        memcpy(texture_staging.ptr, texture->raw_image.data(), texture_bytes);
        texture_staging.flush(dev);

        texture_stagings.emplace_back(move(texture_staging));

        uint32_t mip_levels = getMipLevels(*texture);
        gpu_textures.emplace_back(alloc.makeTexture(texture->width,
                                                    texture->height,
                                                    mip_levels));
    }

    // Copy all geometry into single buffer
    const auto &cpu_meshes = scene_desc.getMeshes();
    
    auto staged = assetHelper.stageGeometry(cpu_meshes, dev, alloc);

    LocalBuffer geometry = alloc.makeGeometryBuffer(staged.totalBytes);

    // Start recording for transfer queue
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(transferStageCommand, &begin_info));

    // Copy vertex/index buffer onto GPU
    VkBufferCopy copy_settings {};
    copy_settings.size = staged.totalBytes;
    dev.dt.cmdCopyBuffer(transferStageCommand, staged.buffer.buffer,
                         geometry.buffer, 1, &copy_settings);

    // Set initial texture layouts
    DynArray<VkImageMemoryBarrier> barriers(gpu_textures.size());
    for (size_t i = 0; i < gpu_textures.size(); i++) {
        const LocalImage &gpu_texture = gpu_textures[i];
        VkImageMemoryBarrier &barrier = barriers[i];

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
            0, 1, 0, 1
        };
    }

    if (gpu_textures.size() > 0) {
        dev.dt.cmdPipelineBarrier(transferStageCommand,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 0, nullptr, 0, nullptr,
                                  barriers.size(), barriers.data());

        for (size_t i = 0; i < gpu_textures.size(); i++) {
            const HostBuffer &stage_buffer = texture_stagings[i];
            const LocalImage &gpu_texture = gpu_textures[i];
            VkBufferImageCopy copy_spec {};
            copy_spec.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy_spec.imageSubresource.mipLevel = 0;
            copy_spec.imageSubresource.baseArrayLayer = 0;
            copy_spec.imageSubresource.layerCount = 1;
            copy_spec.imageExtent =
                { gpu_texture.width, gpu_texture.height, 1 };

            dev.dt.cmdCopyBufferToImage(transferStageCommand,
                                        stage_buffer.buffer,
                                        gpu_texture.image,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        1, &copy_spec);
        }

        // Transfer queue relinquish mip level 0
        for (VkImageMemoryBarrier &barrier : barriers) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;;
            barrier.srcQueueFamilyIndex = dev.transferQF;
            barrier.dstQueueFamilyIndex = dev.gfxQF;
        }
    }

    // Transfer queue relinquish geometry (also barrier on geometry write)
    VkBufferMemoryBarrier geometry_barrier;
    geometry_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    geometry_barrier.pNext = nullptr;
    geometry_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    geometry_barrier.dstAccessMask = 0;
    geometry_barrier.srcQueueFamilyIndex = dev.transferQF;
    geometry_barrier.dstQueueFamilyIndex = dev.gfxQF;
    geometry_barrier.buffer = geometry.buffer;
    geometry_barrier.offset = 0;
    geometry_barrier.size = staged.totalBytes;

    // Geometry & texture barrier execute.
    dev.dt.cmdPipelineBarrier(transferStageCommand,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr,
                              1, &geometry_barrier,
                              barriers.size(), barriers.data());

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
    geometry_barrier.srcAccessMask = 0;
    geometry_barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                     VK_ACCESS_INDEX_READ_BIT;
    dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                              0, 0, nullptr,
                              1, &geometry_barrier,
                              0, nullptr);

    if (gpu_textures.size() > 0)  {
        // Finish acquiring mip level 0 on graphics queue and transition layout
        for (VkImageMemoryBarrier &barrier : barriers) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcQueueFamilyIndex = dev.transferQF;
            barrier.dstQueueFamilyIndex = dev.gfxQF;
        }

        dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 0, nullptr, 0, nullptr,
                                  barriers.size(), barriers.data());

        generateMips(dev, gfxCopyCommand, gpu_textures, barriers);

        // Final layout transition for textures
        for (size_t texture_idx = 0; texture_idx < gpu_textures.size();
                texture_idx++) {
            const LocalImage &gpu_texture = gpu_textures[texture_idx];
            VkImageMemoryBarrier &barrier = barriers[texture_idx];

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = gpu_texture.mipLevels;
        }

        dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                  0, 0, nullptr, 0, nullptr,
                                  barriers.size(), barriers.data());

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
    vector<VkDescriptorImageInfo> view_infos;
    texture_views.reserve(gpu_textures.size());
    view_infos.reserve(gpu_textures.size());
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
        view_infos.emplace_back(VkDescriptorImageInfo {
            VK_NULL_HANDLE, // Immutable
            view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        });
    }

    assert(gpu_textures.size() <= VulkanConfig::max_textures);

    DescriptorSet texture_set = descriptorManager.makeSet();

    // FIXME null descriptorManager feels a bit indirect
    if (texture_set.hdl != VK_NULL_HANDLE) {
        VkWriteDescriptorSet desc_update;
        desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        desc_update.pNext = nullptr;
        desc_update.dstSet = texture_set.hdl;
        desc_update.dstBinding = 0;
        desc_update.dstArrayElement = 0;
        desc_update.descriptorCount = view_infos.size();
        desc_update.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        desc_update.pImageInfo = view_infos.data();
        desc_update.pBufferInfo = nullptr;
        desc_update.pTexelBufferView = nullptr;
        dev.dt.updateDescriptorSets(dev.hdl, 1, &desc_update, 0, nullptr);
    }

    return make_shared<Scene>(Scene {
        move(gpu_textures),
        move(texture_views),
        move(texture_set),
        move(geometry),
        staged.indexBufferOffset,
        move(staged.meshPositions),
        EnvironmentInit(scene_desc.getDefaultInstances(),
                        scene_desc.getDefaultLights(),
                        cpu_meshes.size())
    });
}

shared_ptr<Texture> LoaderState::loadTexture(const vector<uint8_t> &raw)
{
    return readSDRTexture(raw.data(), raw.size());
}


template <typename MatDescType>
shared_ptr<Material> LoaderState::makeMaterial(MatDescType description)
{
    return makeSharedMaterial(move(description));
}

template shared_ptr<Material> LoaderState::makeMaterial(
        UnlitRendererInputs::MaterialDescription);
template shared_ptr<Material> LoaderState::makeMaterial(
        LitRendererInputs::MaterialDescription);

template <typename VertexType>
shared_ptr<Mesh> LoaderState::makeMesh(vector<VertexType> vertices,
                                       vector<uint32_t> indices)
{
    return makeSharedMesh(move(vertices), move(indices));
}

template shared_ptr<Mesh> LoaderState::makeMesh(
        vector<UnlitRendererInputs::NoColorVertex>, vector<uint32_t>);
template shared_ptr<Mesh> LoaderState::makeMesh(
        vector<UnlitRendererInputs::ColoredVertex>, vector<uint32_t>);
template shared_ptr<Mesh> LoaderState::makeMesh(
        vector<UnlitRendererInputs::TexturedVertex>, vector<uint32_t>);
template shared_ptr<Mesh> LoaderState::makeMesh(
        vector<LitRendererInputs::NoColorVertex>, vector<uint32_t>);
template shared_ptr<Mesh> LoaderState::makeMesh(
        vector<LitRendererInputs::TexturedVertex>, vector<uint32_t>);

}
