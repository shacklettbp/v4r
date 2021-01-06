#ifndef VULKAN_HANDLES_HPP_INCLUDED
#define VULKAN_HANDLES_HPP_INCLUDED

#include <array>
#include <optional>
#include <vector>

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

    uint32_t rtRecursionDepth;
    VkDeviceSize rtShaderGroupBaseAlignment;
    VkDeviceSize rtShaderGroupHandleSize;

    const VkPhysicalDevice phy;
    const VkDevice hdl;
    const DeviceDispatch dt;

    DeviceState() = delete;
    DeviceState(const DeviceState &) = delete;
    DeviceState(DeviceState &&) = default;
};

struct InstanceState {
public:
    const VkInstance hdl;
    const InstanceDispatch dt;

    InstanceState(bool need_present,
                  const std::vector<const char *> &extra_exts);
    InstanceState(const InstanceState &) = delete;
    InstanceState(InstanceState &&) = default;

    DeviceState makeDevice(const DeviceUUID &uuid,
                           bool enable_rt,
                           uint32_t desired_gfx_queues,
                           uint32_t desired_compute_queues,
                           uint32_t desired_transfer_queues,
                           std::add_pointer_t<
                               VkBool32(VkInstance,
                                        VkPhysicalDevice,
                                        uint32_t)> present_check) const;
private:
    const VkDebugUtilsMessengerEXT debug_;

    InstanceState(bool enable_validation, bool need_present,
                  const std::vector<const char *> &extra_exts);

    VkPhysicalDevice findPhysicalDevice(const DeviceUUID &uuid) const;
};

}

#endif
