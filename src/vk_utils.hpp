#ifndef VK_UTILS_HPP_INCLUDED
#define VK_UTILS_HPP_INCLUDED

#include <string>

#include <vulkan/vulkan.h>

namespace v4r {

[[noreturn]] void fatalExit() noexcept;

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

#define LINE_SAFE(m) #m
#define LINE_STR(m) LINE_SAFE(m)
#define LOC_APPEND(m) m ": " __FILE__ " @ " LINE_STR(__LINE__)
#define REQ_VK(expr) checkVk((expr), LOC_APPEND(#expr))
#define CHK_VK(expr) checkVk((expr), LOC_APPEND(#expr), false)

}

#endif
