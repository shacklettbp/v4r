#ifndef VK_UTILS_HPP_INCLUDED
#define VK_UTILS_HPP_INCLUDED

#include <string>

#include <vulkan/vulkan.h>

#include "utils.hpp"

namespace v4r {

uint32_t findMemoryTypeIndex(uint32_t allowed_type_bits,
        VkMemoryPropertyFlags required_props,
        VkPhysicalDeviceMemoryProperties2 &mem_props);

void printVkError(VkResult res, const char *msg);

static inline VkResult checkVk(VkResult res, const char *msg,
                               bool fatal = true) noexcept
{
    if (res != VK_SUCCESS) {
        printVkError(res, msg);
        if (fatal) {
            fatalExit();
        }
    }

    return res;
}

#define STRINGIFY_HELPER(m) #m
#define STRINGIFY(m) STRINGIFY_HELPER(m)

#define LOC_APPEND(m) m ": " __FILE__ " @ " STRINGIFY(__LINE__)
#define REQ_VK(expr) checkVk((expr), LOC_APPEND(#expr))
#define CHK_VK(expr) checkVk((expr), LOC_APPEND(#expr), false)

}

#endif
