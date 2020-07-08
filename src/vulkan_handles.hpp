#ifndef VULKAN_HANDLES_HPP_INCLUDED
#define VULKAN_HANDLES_HPP_INCLUDED

#include <array>

#include <vulkan/vulkan.h>

#include "dispatch.hpp"

namespace v4r {

using DeviceUUID = std::array<uint8_t, VK_UUID_SIZE>;

struct DeviceState {
public:
    uint32_t gfxQF;
    uint32_t computeQF;
    uint32_t transferQF;

    uint32_t numGraphicsQueues;
    uint32_t numComputeQueues;
    uint32_t numTransferQueues;

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

    DeviceState makeDevice(const DeviceUUID &uuid,
                           uint32_t desired_gfx_queues,
                           uint32_t desired_compute_queues,
                           uint32_t desired_transfer_queues) const;
private:
    const VkDebugUtilsMessengerEXT debug_;

    InstanceState(bool enable_validation);

    VkPhysicalDevice findPhysicalDevice(const DeviceUUID &uuid) const;
};

}

#endif
