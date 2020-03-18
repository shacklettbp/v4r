#include "vk_utils.hpp"

#include <cstdlib>
#include <iostream>

using namespace std;

namespace v4r {

uint32_t findMemoryTypeIndex(uint32_t allowed_type_bits,
        VkMemoryPropertyFlags required_props,
        VkPhysicalDeviceMemoryProperties2 &mem_props)
{
    uint32_t num_mems = mem_props.memoryProperties.memoryTypeCount;

    for (uint32_t idx = 0; idx < num_mems; idx++) {
        uint32_t mem_type_bits = (1 << idx);
        if (!(allowed_type_bits & mem_type_bits)) continue;

        VkMemoryPropertyFlags supported_props =
            mem_props.memoryProperties.memoryTypes[idx].propertyFlags;

        if ((required_props & supported_props) == required_props) {
            return idx;
        }
    }

    cerr << "Failed to find desired memory type" << endl;
    fatalExit();
}

void printVkError(VkResult res, const char *msg)
{
#define ERR_CASE(val) case VK_##val: cerr << #val; break

    cerr << msg << ": ";
    switch (res) {
        ERR_CASE(NOT_READY);
        ERR_CASE(TIMEOUT);
        ERR_CASE(EVENT_SET);
        ERR_CASE(EVENT_RESET);
        ERR_CASE(INCOMPLETE);
        ERR_CASE(ERROR_OUT_OF_HOST_MEMORY);
        ERR_CASE(ERROR_OUT_OF_DEVICE_MEMORY);
        ERR_CASE(ERROR_INITIALIZATION_FAILED);
        ERR_CASE(ERROR_DEVICE_LOST);
        ERR_CASE(ERROR_MEMORY_MAP_FAILED);
        ERR_CASE(ERROR_LAYER_NOT_PRESENT);
        ERR_CASE(ERROR_EXTENSION_NOT_PRESENT);
        ERR_CASE(ERROR_FEATURE_NOT_PRESENT);
        ERR_CASE(ERROR_INCOMPATIBLE_DRIVER);
        ERR_CASE(ERROR_TOO_MANY_OBJECTS);
        ERR_CASE(ERROR_FORMAT_NOT_SUPPORTED);
        ERR_CASE(ERROR_FRAGMENTED_POOL);
        ERR_CASE(ERROR_UNKNOWN);
        ERR_CASE(ERROR_OUT_OF_POOL_MEMORY);
        ERR_CASE(ERROR_INVALID_EXTERNAL_HANDLE);
        ERR_CASE(ERROR_FRAGMENTATION);
        ERR_CASE(ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
        ERR_CASE(ERROR_SURFACE_LOST_KHR);
        ERR_CASE(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        ERR_CASE(SUBOPTIMAL_KHR);
        ERR_CASE(ERROR_OUT_OF_DATE_KHR);
        ERR_CASE(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        ERR_CASE(ERROR_VALIDATION_FAILED_EXT);
        ERR_CASE(ERROR_INVALID_SHADER_NV);
        ERR_CASE(ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
        ERR_CASE(ERROR_NOT_PERMITTED_EXT);
        ERR_CASE(ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
        default: cerr << "New vulkan error"; break;
    }
    cerr << endl;
#undef ERR_CASE
}

}
