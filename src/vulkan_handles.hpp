#ifndef VULKAN_HANDLES_HPP_INCLUDED
#define VULKAN_HANDLES_HPP_INCLUDED

#include <vulkan/vulkan.h>

#include "dispatch.hpp"

namespace v4r {

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

}

#endif
