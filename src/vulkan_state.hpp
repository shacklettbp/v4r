#ifndef VULKAN_STATE_HPP_INCLUDED
#define VULKAN_STATE_HPP_INCLUDED

#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include <v4r/config.hpp>

#include "dispatch.hpp"

namespace v4r {

struct FramebufferConfig {
public:
    VkFormat depthFmt;
    VkFormat colorFmt;

    VkImageCreateInfo colorCreationSettings;
    VkMemoryAllocateInfo colorMemorySettings;
};

struct FramebufferState {
public:
    VkImage image;
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
                       const FramebufferConfig &fb_cfg);
    CommandStreamState(const CommandStreamState &) = delete;
    CommandStreamState(CommandStreamState &&) = default;

    const DeviceState &dev;
    const VkCommandPool gfxPool;
    const VkQueue gfxQueue;

    const FramebufferState fb;
};

struct VulkanState {
public:
    VulkanState(const RenderConfig &cfg);
    VulkanState(const VulkanState &) = delete;
    VulkanState(VulkanState &&) = delete;

    const RenderConfig cfg;

    const InstanceState inst;
    const DeviceState dev;
    const FramebufferConfig fbCfg;
};

}

#endif
