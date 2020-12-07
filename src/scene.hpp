#ifndef SCENE_HPP_INCLUDED
#define SCENE_HPP_INCLUDED

#include <v4r/config.hpp>
#include <v4r/assets.hpp>

#include <filesystem>
#include <list>
#include <mutex>
#include <string_view>
#include <unordered_map>

#include "descriptors.hpp"
#include "utils.hpp"
#include "vulkan_handles.hpp"
#include "vulkan_memory.hpp"

// Forward declare ktxTexture as kind of an opaque backing data type
struct ktxTexture;

namespace v4r {

struct PerSceneDescriptorConfig {
    VkSampler textureSampler;
    VkDescriptorSetLayout layout;
};

struct Mesh {
    std::vector<uint32_t> indices;
};

template <typename VertexType>
struct VertexMesh : public Mesh {
    std::vector<VertexType> vertices;

    VertexMesh(std::vector<VertexType> vertices,
               std::vector<uint32_t> indices);
};

template <typename VertexType>
struct VertexImpl;

struct Texture {
    uint32_t width;
    uint32_t height;
    uint32_t numLevels;

    ktxTexture *data;

    ~Texture();
};

namespace MaterialParam {
struct DiffuseColorTexture {
    std::shared_ptr<Texture> value;
};

struct DiffuseColorUniform {
    glm::vec4 value;
};

using AlbedoColorTexture = DiffuseColorTexture;
using AlbedoColorUniform = DiffuseColorUniform;

struct SpecularColorTexture {
    std::shared_ptr<Texture> value;
};

struct SpecularColorUniform {
    glm::vec4 value;
};

struct ShininessUniform {
    float value;
};
};

struct Material {
    std::vector<std::shared_ptr<Texture>> textures;
    std::vector<uint8_t> paramBytes;
};

template <typename MaterialParamsType>
struct MaterialImpl;

struct EnvironmentInit {
    EnvironmentInit(const std::vector<InstanceProperties> &instances,
                    const std::vector<LightProperties> &lights,
                    uint32_t num_meshes);

    std::vector<std::vector<glm::mat4x3>> transforms;
    std::vector<std::vector<uint32_t>> materials;
    std::vector<std::pair<uint32_t, uint32_t>> indexMap;
    std::vector<std::vector<uint32_t>> reverseIDMap;

    std::vector<LightProperties> lights;
    std::vector<uint32_t> lightIDs;
    std::vector<uint32_t> lightReverseIDs;
};

struct TextureData {
    TextureData(const DeviceState &d, MemoryAllocator &a);
    TextureData(const TextureData &) = delete;
    TextureData(TextureData &&);
    ~TextureData();

    std::vector<LocalTexture> textures;
    std::vector<VkImageView> views;
    MemoryChunk memory;
    const DeviceState &dev;
    MemoryAllocator &alloc;
};

struct AccelBuildInfo {
    VkDeviceAddress vertexAddress;
    VkDeviceAddress indexAddress;
    uint32_t numTriangles;
    uint32_t maxVertexIndex;
};

struct MeshInfo {
    uint32_t indexOffset;
    uint32_t chunkOffset;
    uint32_t numTriangles;
    uint32_t numVertices;
    uint32_t numChunks;
};

struct BLAS {
    VkAccelerationStructureKHR accelStruct;
    VkDeviceAddress devAddr;
};

class BLASData {
public:
    BLASData(const BLASData &) = delete;
    BLASData(BLASData &&) = default;
    ~BLASData();

    const DeviceState &dev;
    std::vector<BLAS> accelStructs;
    LocalBuffer storage;
};

struct Scene {
    TextureData textures;
    DescriptorSet materialSet;
    DescriptorSet cullSet;
    LocalBuffer data;
    VkDeviceSize indexOffset;
    std::vector<MeshInfo> meshMetadata;
    uint32_t numMeshes;
    EnvironmentInit envDefaults;
    std::optional<DescriptorSet> rtSet;
    std::optional<BLASData> blases;
};

struct RTEnvironmentState {
    VkAccelerationStructureKHR tlas;
    DescriptorSet tlasSet;
    LocalBuffer storage;
};

class EnvironmentState {
public:
    EnvironmentState(const DeviceState &d, MemoryAllocator &alloc,
                     const std::shared_ptr<Scene> &s, const glm::mat4 &proj,
                     bool enable_rt, DescriptorSet &&rt_desc_set,
                     VkCommandBuffer build_cmd, const QueueState &build_queue);
    EnvironmentState(const EnvironmentState &) = delete;
    EnvironmentState(EnvironmentState &&);
    ~EnvironmentState();

    std::shared_ptr<Scene> scene;
    glm::mat4 projection;
    FrustumBounds frustumBounds;

    std::vector<std::vector<uint32_t>> reverseIDMap;
    std::vector<uint32_t> freeIDs;

    std::vector<LightProperties> lights;
    std::vector<uint32_t> freeLightIDs;
    std::vector<uint32_t> lightIDs;
    std::vector<uint32_t> lightReverseIDs;

    const DeviceState &dev;
    std::optional<RTEnvironmentState> rtState;
};

struct MaterialMetadata {
    std::vector<std::filesystem::path> textures;
    uint32_t numMaterials;
    uint32_t texturesPerMaterial;
    std::vector<uint32_t> textureIndices;
};

struct StagingHeader {
    // Geometry data
    uint32_t numVertices;
    uint32_t numIndices;
    uint64_t indexOffset;
    uint64_t meshletBufferOffset;
    uint64_t meshletBufferBytes;
    uint64_t meshletOffset;
    uint64_t meshletBytes;
    uint64_t meshChunkOffset;
    uint64_t meshChunkBytes;

    // Material data
    uint64_t materialOffset;
    uint64_t materialBytes;

    uint64_t totalBytes;
    uint32_t numMeshes;
};

struct StagedScene {
    HostBuffer buffer;
    StagingHeader hdr;
    std::vector<MeshInfo> meshMetadata;
};

struct SceneLoadInfo {
    StagedScene scene;
    MaterialMetadata materialMetadata;
    EnvironmentInit envInit;
};

struct LoaderImpl {
    std::add_pointer_t<StagedScene(const std::vector<std::shared_ptr<Mesh>> &,
                                   const std::vector<uint8_t> &param_bytes,
                                   const DeviceState &,
                                   MemoryAllocator &)>
        stageScene;

    std::add_pointer_t<SceneDescription(std::string_view, const glm::mat4 &)>
        parseScene;

    std::add_pointer_t<std::shared_ptr<Mesh>(std::string_view)> loadMesh;

    std::add_pointer_t<
        SceneLoadInfo(std::string_view, const glm::mat4 &, MemoryAllocator &)>
        loadPreprocessedScene;

    uint32_t vertexSize;

    template <typename VertexType, typename MaterialParamsType>
    static LoaderImpl create();
};

class LoaderState {
public:
    LoaderState(const DeviceState &dev,
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
                const glm::mat4 &coordinateTransform,
                const QueueState *compute_queue,
                bool need_acceleration_structure);

    std::shared_ptr<Scene> loadScene(std::string_view scene_path);

    std::shared_ptr<Scene> makeScene(const SceneDescription &desc);

    std::shared_ptr<Texture> loadTexture(const std::vector<uint8_t> &raw);

    template <typename MaterialParamsType>
    std::shared_ptr<Material> makeMaterial(MaterialParamsType params);

    std::shared_ptr<Mesh> loadMesh(std::string_view geometry_path);

    template <typename VertexType>
    std::shared_ptr<Mesh> makeMesh(std::vector<VertexType> vertices,
                                   std::vector<uint32_t> indices);

    const DeviceState &dev;

    const QueueState &transferQueue;
    const QueueState &gfxQueue;

    const VkCommandPool gfxPool;
    const VkCommandBuffer gfxCopyCommand;

    const VkCommandPool transferPool;
    const VkCommandBuffer transferStageCommand;

    const VkSemaphore ownershipSemaphore;
    const VkFence fence;

    MemoryAllocator &alloc;
    DescriptorManager descriptorManager;
    DescriptorManager cullDescriptorManager;
    DescriptorManager rtDescriptorManager;

    glm::mat4 coordinateTransform;

    const QueueState *computeQueue;
    const VkCommandPool computePool;
    const VkCommandBuffer asBuildCommand;
    const VkSemaphore accelBuildSemaphore;

private:
    std::shared_ptr<Scene> makeScene(SceneLoadInfo load_info);

    const LoaderImpl impl_;
    bool need_acceleration_structure_;
};

}

#endif
