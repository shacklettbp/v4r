#ifndef SCENE_HPP_INCLUDED
#define SCENE_HPP_INCLUDED

#include <list>
#include <mutex>
#include <unordered_map>

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

template <typename VertexT>
struct Mesh {
    using VertexType = VertexT;

    std::vector<VertexType> vertices;
    std::vector<uint32_t> indices;

    Mesh(std::vector<VertexType> &&verts,
             std::vector<uint32_t> &&idxs)
        : vertices(move(verts)),
          indices(move(idxs))
    {}
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

template <typename ParamsType>
struct Material {
    ParamsType params;
};

struct EnvironmentInit {
    EnvironmentInit(const std::vector<std::vector<InstanceProperties>> &insts);

    std::vector<std::vector<glm::mat4>> transforms;
    std::vector<std::vector<uint32_t>> materials;
    std::vector<std::pair<uint32_t, uint32_t>> indexMap;
    std::vector<std::vector<uint32_t>> reverseIDMap;
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
};

class LoaderState {
public:
    LoaderState(const DeviceState &dev,
                const PerSceneDescriptorConfig &scene_desc_cfg,
                MemoryAllocator &alc,
                QueueManager &queue_manager,
                const glm::mat4 &coordinateTransform);

    template <typename PipelineType>
    std::shared_ptr<Scene> loadScene(
            const SceneDescription<PipelineType> &scene_desc);
                
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
};

}

#endif
