#ifndef VULKAN_STATE_HPP_INCLUDED
#define VULKAN_STATE_HPP_INCLUDED

#include <array>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <v4r/config.hpp>

#include "dispatch.hpp"

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

struct SceneLoaderConfig {
public:
    uint32_t stageMemTypeIdx;
    uint32_t sceneMemTypeIdx;
};

struct SceneState {
};

struct FramebufferConfig {
public:
    uint32_t width;
    uint32_t height;

    VkFormat depthFmt;
    VkFormat colorFmt;

    VkImageCreateInfo colorCreationSettings;
    VkMemoryAllocateInfo colorMemorySettings;

    VkImageCreateInfo depthCreationSettings;
    VkMemoryAllocateInfo depthMemorySettings;
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

struct DeviceState {
public:
    uint32_t gfxQF;
    uint32_t computeQF;
    uint32_t transferQF;

    const VkPhysicalDevice phy;
    const VkDevice hdl;
    const DeviceDispatch dt;

    DeviceState() = delete;
};

struct InstanceState {
public:
    const VkInstance hdl;
    const InstanceDispatch dt;

    InstanceState();

    DeviceState makeDevice(uint32_t gpu_id) const;
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
    const FramebufferConfig fbCfg;
    const SceneLoaderConfig sceneLoaderCfg;
    const PipelineState pipeline;
};

}

#endif
