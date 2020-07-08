#ifndef SCENE_HPP_INCLUDED
#define SCENE_HPP_INCLUDED

#include <list>
#include <mutex>
#include <unordered_map>

#include <v4r/config.hpp>
#include <v4r/assets.hpp>

#include "descriptors.hpp"
#include "utils.hpp"
#include "vulkan_handles.hpp"
#include "vulkan_memory.hpp"

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

struct InlineMesh {
    uint32_t vertexOffset;
    uint32_t startIndex;
    uint32_t numIndices;
};

struct Texture {
    uint32_t width;
    uint32_t height;
    uint32_t num_channels;

    ManagedArray<uint8_t> raw_image;
};

struct Material {
    std::vector<std::shared_ptr<Texture>> textures;
};

struct BlinnPhongMaterial : Material {
    float shininess;
};

struct EnvironmentInit {
    EnvironmentInit(
            const std::vector<std::pair<uint32_t, InstanceProperties>>
                &instances,
            const std::vector<LightProperties> &lights,
            uint32_t num_meshes);

    std::vector<std::vector<glm::mat4>> transforms;
    std::vector<std::vector<uint32_t>> materials;
    std::vector<std::pair<uint32_t, uint32_t>> indexMap;
    std::vector<std::vector<uint32_t>> reverseIDMap;

    std::vector<LightProperties> lights;
    std::vector<uint32_t> lightIDs;
    std::vector<uint32_t> lightReverseIDs;
};

struct Scene {
    std::vector<LocalImage> textures;
    std::vector<VkImageView> texture_views;
    DescriptorSet textureSet;
    LocalBuffer geometry;
    VkDeviceSize indexOffset;
    std::vector<InlineMesh> meshes;
    EnvironmentInit envDefaults;
};

class EnvironmentState {
public:
    EnvironmentState(const std::shared_ptr<Scene> &s, const glm::mat4 &proj);

    std::shared_ptr<Scene> scene;
    glm::mat4 projection;

    std::vector<std::vector<uint32_t>> reverseIDMap;
    std::vector<uint32_t> freeIDs;

    std::vector<LightProperties> lights;
    std::vector<uint32_t> freeLightIDs;
    std::vector<uint32_t> lightIDs;
    std::vector<uint32_t> lightReverseIDs;
};

struct StagedMeshes {
    HostBuffer buffer;
    std::vector<InlineMesh> meshPositions;
    VkDeviceSize indexBufferOffset;
    VkDeviceSize totalBytes;
};

struct LoaderHelper {
    std::add_pointer_t<
        StagedMeshes(const std::vector<std::shared_ptr<Mesh>> &,
                     const DeviceState &,
                     MemoryAllocator &)>
            stageGeometry;

    std::add_pointer_t<
        SceneDescription(const std::string &, const glm::mat4 &)>
            parseScene;

    std::add_pointer_t<
        std::shared_ptr<Mesh>(const std::string &)>
            loadMesh;
};

class LoaderState {
public:
    LoaderState(const DeviceState &dev,
                const RenderFeatures &features,
                const PerSceneDescriptorConfig &scene_desc_cfg,
                MemoryAllocator &alc,
                QueueManager &queue_manager,
                const glm::mat4 &coordinateTransform);

    std::shared_ptr<Scene> loadScene(
            const SceneDescription &scene_desc);

    std::shared_ptr<Texture> loadTexture(
            const std::vector<uint8_t> &raw);

    template <typename MatDescType>
    std::shared_ptr<Material> makeMaterial(MatDescType description);

    template <typename VertexType>
    std::shared_ptr<Mesh> makeMesh(std::vector<VertexType> vertices,
                                   std::vector<uint32_t> indices);
                
    const DeviceState &dev;

    const VkCommandPool gfxPool;
    const QueueState &gfxQueue;
    const VkCommandBuffer gfxCopyCommand;

    const VkCommandPool transferPool;
    const QueueState &transferQueue;
    const VkCommandBuffer transferStageCommand;

    const VkSemaphore semaphore;
    const VkFence fence;

    MemoryAllocator &alloc;
    DescriptorManager descriptorManager;

    glm::mat4 coordinateTransform;

    LoaderHelper assetHelper;
};

}

#endif
