#ifndef VULKAN_STATE_HPP_INCLUDED
#define VULKAN_STATE_HPP_INCLUDED

#include <array>
#include <atomic>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <v4r/config.hpp>

#include "utils.hpp"
#include "vulkan_handles.hpp"
#include "vulkan_memory.hpp"

namespace v4r {

struct Vertex {
    glm::vec3 position;
    glm::vec2 uv;
    glm::vec3 color;
};

struct Texture {
    uint32_t width;
    uint32_t height;
    uint32_t num_channels;

    ManagedArray<uint8_t> raw_image;
};

struct Material {
    std::optional<uint64_t> ambientTexture;
    glm::vec4 ambientColor;

    std::optional<uint64_t> diffuseTexture;
    glm::vec4 diffuseColor;

    std::optional<uint64_t> specularTexture;
    glm::vec4 specularColor;

    float shininess;
};

struct SceneMesh {
    uint32_t startIndex;
    uint32_t numIndices;
    size_t materialIndex;
};

struct SceneAssets {
    std::list<Texture> textures;
    std::vector<Material> materials;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    std::vector<SceneMesh> meshes;
};

struct SceneState {
    std::vector<LocalTexture> textures;
    std::vector<VkImageView> texture_views;
    std::vector<Material> materials;
    LocalBuffer geometry;
    VkDeviceSize indexOffset;
    std::vector<SceneMesh> meshes;
};

struct DescriptorConfig {
    VkSampler textureSampler;
    VkDescriptorSetLayout layout;
};

struct PoolState {
    PoolState(VkDescriptorPool p)
        : pool(p), numActive(0)
    {}

    VkDescriptorPool pool;
    std::atomic_uint64_t numActive;
};

struct DescriptorSet {
    ~DescriptorSet() { pool.numActive--; };

    VkDescriptorSet hdl;
    PoolState &pool;
};

class DescriptorTracker {
public:
    DescriptorTracker(const DeviceState &dev, const DescriptorConfig &cfg);
    ~DescriptorTracker();

    DescriptorSet makeDescriptorSet();

private:
    const DeviceState &dev;
    const VkDescriptorSetLayout &layout_;

    std::list<PoolState> free_pools_;
    std::list<PoolState> used_pools_;
};

struct FramebufferConfig {
public:
    uint32_t width;
    uint32_t height;

    VkFormat colorFmt;
    VkImageCreateInfo colorCreationSettings;
    VkDeviceSize colorMemorySize;
    uint32_t colorMemoryTypeIdx;

    VkFormat depthFmt;
    VkImageCreateInfo depthCreationSettings;
    VkDeviceSize depthMemorySize;
    uint32_t depthMemoryTypeIdx;
};

struct FramebufferState {
public:
    VkImage colorImg;
    VkDeviceMemory colorMem;

    VkImage depthImg;
    VkDeviceMemory depthMem;

    std::array<VkImageView, 2> attachmentViews; 

    VkFramebuffer hdl;
};

// FIXME separate out things like the layout, cache (maybe renderpass)
// into a PipelineConfig type struct
struct PipelineState {
    VkRenderPass renderPass;

    std::array<VkShaderModule, 2> shaders;

    VkPipelineCache pipelineCache;
    VkPipelineLayout gfxLayout;
    VkPipeline gfxPipeline;
};

struct CommandStreamState {
public:
    CommandStreamState(const InstanceState &inst,
                       const DeviceState &dev,
                       const DescriptorConfig &desc_cfg,
                       const PipelineState &pl,
                       MemoryAllocator &alc);
    CommandStreamState(const CommandStreamState &) = delete;
    CommandStreamState(CommandStreamState &&) = default;

    SceneState loadScene(SceneAssets &&assets);

    const InstanceState &inst;
    const DeviceState &dev;
    const PipelineState &pipeline;

    const VkCommandPool gfxPool;
    const VkQueue gfxQueue;
    const VkCommandBuffer gfxCopyCommand;

    const VkCommandPool transferPool;
    const VkQueue transferQueue;
    const VkCommandBuffer transferStageCommand;
    const VkSemaphore copySemaphore;
    const VkFence copyFence;

    MemoryAllocator &alloc;
    const DescriptorTracker descriptorTracker;
};

struct VulkanState {
public:
    VulkanState(const RenderConfig &cfg);
    VulkanState(const VulkanState &) = delete;
    VulkanState(VulkanState &&) = delete;

    const RenderConfig cfg;

    const InstanceState inst;
    const DeviceState dev;
    MemoryAllocator alloc;
    const FramebufferConfig fbCfg;
    const DescriptorConfig descCfg;
    const PipelineState pipeline;
    const FramebufferState fb;
};

}

#endif
