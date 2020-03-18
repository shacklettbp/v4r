#ifndef VULKAN_STATE_HPP_INCLUDED
#define VULKAN_STATE_HPP_INCLUDED

#include <array>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <v4r/config.hpp>

#include "vulkan_handles.hpp"
#include "vulkan_memory.hpp"

namespace v4r {

struct Vertex {
    glm::vec3 position;
    glm::vec2 uv;
    glm::vec3 color;
};

struct SceneMesh {
    uint32_t startIndex;
    uint32_t numIndices;
};

struct SceneAssets {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    std::vector<SceneMesh> meshes;
};

struct SceneState {
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
    CommandStreamState(const DeviceState &dev,
                       const FramebufferConfig &fb_cfg,
                       const PipelineState &pl);
    CommandStreamState(const CommandStreamState &) = delete;
    CommandStreamState(CommandStreamState &&) = default;

    const DeviceState &dev;
    const PipelineState &pipeline;
    const VkCommandPool gfxPool;
    const VkQueue gfxQueue;

    const FramebufferState fb;
};

struct VulkanState {
public:
    VulkanState(const RenderConfig &cfg);
    VulkanState(const VulkanState &) = delete;
    VulkanState(VulkanState &&) = delete;

    SceneState loadScene(const SceneAssets &assets) const;

    const RenderConfig cfg;

    const InstanceState inst;
    const DeviceState dev;
    MemoryAllocator alloc;
    const FramebufferConfig fbCfg;
    const PipelineState pipeline;
};

}

#endif
