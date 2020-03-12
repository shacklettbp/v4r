#ifndef VULKAN_STATE_HPP_INCLUDED
#define VULKAN_STATE_HPP_INCLUDED

#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include "dispatch.hpp"

namespace v4r {

struct DeviceState {
public:
    uint32_t gfxQF;
    uint32_t computeQF;
    uint32_t transferQF;

    VkFormat depthFmt;

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
private:
    VkFormat getDeviceDepthFormat(VkPhysicalDevice phy) const;
};

struct CommandStreamState {
public:
    CommandStreamState(const DeviceState &dev);
    CommandStreamState(const CommandStreamState &) = delete;
    CommandStreamState(CommandStreamState &&) = default;

    const DeviceState &dev;
    const VkCommandPool gfxPool;
    const VkQueue gfxQueue;
};


struct VulkanState {
public:
    VulkanState(uint32_t gpu_id);
    VulkanState(const VulkanState &) = delete;
    VulkanState(VulkanState &&) = delete;

    const InstanceState inst;
    const DeviceState dev;
};
}

#endif
