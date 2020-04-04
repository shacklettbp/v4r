#include "vulkan_handles.hpp"

#include "utils.hpp"
#include "vulkan_config.hpp"
#include "vk_utils.hpp"

#include <iostream>
#include <vector>

using namespace std;

extern "C" {
    VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
            const VkInstanceCreateInfo *, const VkAllocationCallbacks *,
            VkInstance *);
}

namespace v4r {

static const char *instance_extensions[] = {
};

static const char *device_extensions[] = {
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME
};

static constexpr uint32_t num_instance_extensions =
    sizeof(instance_extensions) / sizeof(const char *);

static constexpr uint32_t num_device_extensions =
    sizeof(device_extensions) / sizeof(const char *);

static VkInstance createInstance()
{
    VkApplicationInfo app_info {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "v4r";
    app_info.pEngineName = "v4r";
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo inst_info {};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo = &app_info;

    if (num_instance_extensions > 0) {
        inst_info.enabledExtensionCount = num_instance_extensions;
        inst_info.ppEnabledExtensionNames = instance_extensions;
    }

    VkInstance inst;
    REQ_VK(vkCreateInstance(&inst_info, nullptr, &inst));

    return inst;
}

InstanceState::InstanceState()
    : hdl(createInstance()),
      dt(hdl)
{}

static void fillQueueInfo(VkDeviceQueueCreateInfo &info, uint32_t idx,
                          const vector<float> &priorities)
{
    info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    info.queueFamilyIndex = idx;
    info.queueCount = priorities.size();
    info.pQueuePriorities = priorities.data();
}
DeviceState InstanceState::makeDevice(const uint32_t gpu_id) const
{
    uint32_t num_gpus;
    REQ_VK(dt.enumeratePhysicalDevices(hdl, &num_gpus, nullptr));

    DynArray<VkPhysicalDevice> phys(num_gpus);
    REQ_VK(dt.enumeratePhysicalDevices(hdl, &num_gpus, phys.data()));

    if (num_gpus <= gpu_id) {
        cerr << "Not enough GPUs found by vulkan" << endl;
        fatalExit();
    }

    VkPhysicalDevice phy = phys[gpu_id];

    VkPhysicalDeviceProperties2 props;
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = nullptr;
    dt.getPhysicalDeviceProperties2(phy, &props);

    VkPhysicalDeviceFeatures2 feats;
    feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats.pNext = nullptr;
    dt.getPhysicalDeviceFeatures2(phy, &feats);

    uint32_t num_queue_families;
    dt.getPhysicalDeviceQueueFamilyProperties2(phy, &num_queue_families,
                                                  nullptr);

    if (num_queue_families == 0) {
        cerr << "GPU doesn't have any queue families" << endl;
        fatalExit();
    }

    DynArray<VkQueueFamilyProperties2> queue_family_props(num_queue_families);
    for (auto &qf : queue_family_props) {
        qf.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        qf.pNext = nullptr;
    }

    dt.getPhysicalDeviceQueueFamilyProperties2(phy, &num_queue_families,
                                               queue_family_props.data());

    // Currently only finds dedicated transfer, compute, and gfx queues
    // FIXME implement more flexiblity in queue choices
    optional<uint32_t> compute_queue_family;
    optional<uint32_t> gfx_queue_family;
    optional<uint32_t> transfer_queue_family;
    for (uint32_t i = 0; i < num_queue_families; i++) {
        const auto &qf = queue_family_props[i];
        auto &qf_prop = qf.queueFamilyProperties;

        if (!transfer_queue_family &&
            (qf_prop.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(qf_prop.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {

            transfer_queue_family = i;
        } else if (!compute_queue_family &&
                   (qf_prop.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                   !(qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {

            compute_queue_family = i;;
        } else if (!gfx_queue_family &&
                   (qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {

            gfx_queue_family = i;
        }

        if (transfer_queue_family && compute_queue_family &&
            gfx_queue_family) {
            break;
        }
    }

    if (!compute_queue_family || !gfx_queue_family || !transfer_queue_family) {
        cerr << "GPU does not support required separate queues" << endl;
        fatalExit();
    }

    const uint32_t num_gfx_queues =
        min(VulkanConfig::num_desired_gfx_queues,
            queue_family_props[*gfx_queue_family].
                queueFamilyProperties.queueCount);
    const uint32_t num_compute_queues =
        min(VulkanConfig::num_desired_compute_queues,
            queue_family_props[*compute_queue_family].
                queueFamilyProperties.queueCount);
    const uint32_t num_transfer_queues =
        min(VulkanConfig::num_desired_transfer_queues,
        queue_family_props[*transfer_queue_family].
            queueFamilyProperties.queueCount);

    array<VkDeviceQueueCreateInfo, 3> queue_infos {};
    vector<float> gfx_pris(num_gfx_queues, VulkanConfig::gfx_priority);
    vector<float> compute_pris(num_compute_queues,
                               VulkanConfig::compute_priority);
    vector<float> transfer_pris(num_transfer_queues,
                                VulkanConfig::transfer_priority);
    fillQueueInfo(queue_infos[0], *gfx_queue_family, gfx_pris);
    fillQueueInfo(queue_infos[1], *compute_queue_family, compute_pris);
    fillQueueInfo(queue_infos[2], *transfer_queue_family, transfer_pris);

    VkDeviceCreateInfo dev_create_info {};
    dev_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_create_info.queueCreateInfoCount = 3;
    dev_create_info.pQueueCreateInfos = queue_infos.data();
    dev_create_info.enabledExtensionCount = num_device_extensions;
    dev_create_info.ppEnabledExtensionNames = device_extensions;

    // Would enable sampler anisotropy here but current habitat
    // does not use it
    dev_create_info.pEnabledFeatures = nullptr;
    VkPhysicalDeviceFeatures2 requested_features {};
    requested_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    requested_features.pNext = nullptr;
    requested_features.features.samplerAnisotropy = false;
    dev_create_info.pNext = &requested_features;

    VkDevice dev;
    REQ_VK(dt.createDevice(phy, &dev_create_info, nullptr, &dev));

    return DeviceState {
        *gfx_queue_family,
        *compute_queue_family,
        *transfer_queue_family,
        num_gfx_queues,
        num_compute_queues,
        num_transfer_queues,
        phy,
        dev,
        DeviceDispatch(dev)
    };
}

}
