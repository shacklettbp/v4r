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

struct VulkanThreadState;

struct VulkanState {
public:
    VulkanState(uint32_t gpu_id);

    const InstanceState inst;
    const DeviceState dev;
};

struct VulkanThreadState {
public:
    VulkanThreadState(const DeviceState &dev);

    const DeviceState &dev;
    const VkCommandPool gfxPool;
};

}

#endif
