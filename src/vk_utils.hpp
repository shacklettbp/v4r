#ifndef VK_UTILS_HPP_INCLUDED
#define VK_UTILS_HPP_INCLUDED

#include <deque>
#include <mutex>
#include <string>

#include "vulkan_handles.hpp"
#include "utils.hpp"

namespace v4r {

class QueueState {
public:
    inline QueueState(VkQueue queue_hdl, bool shared);

    inline void submit(const DeviceState &dev, uint32_t submit_count,
                       const VkSubmitInfo *pSubmits, VkFence fence) const;

    inline void bindSubmit(const DeviceState &dev, uint32_t submit_count,
        const VkBindSparseInfo *pSubmits, VkFence fence) const;

    inline bool presentSubmit(const DeviceState &dev,
                              const VkPresentInfoKHR *present_info) const;

private:
    VkQueue queue_hdl_;

    bool shared_;
    mutable std::mutex mutex_;
};

inline VkCommandPool makeCmdPool(const DeviceState &dev, uint32_t qf_idx);

inline VkCommandBuffer makeCmdBuffer(const DeviceState &dev,
        VkCommandPool pool,
        VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

inline VkQueue makeQueue(const DeviceState &dev,
                         uint32_t qf_idx, uint32_t queue_idx);

inline VkSemaphore makeBinarySemaphore(const DeviceState &dev);

inline VkSemaphore makeBinaryExternalSemaphore(const DeviceState &dev);
int exportBinarySemaphore(const DeviceState &dev, VkSemaphore semaphore);

inline VkFence makeFence(const DeviceState &dev, bool pre_signal=false);

inline void waitForFenceInfinitely(const DeviceState &dev, VkFence fence);

inline void resetFence(const DeviceState &dev, VkFence fence);

inline uint32_t getWorkgroupSize(uint32_t num_items);

inline VkDeviceSize alignOffset(VkDeviceSize offset, VkDeviceSize alignment);

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

#ifndef VK_UTILS_INL_INCLUDED
#include "vk_utils.inl"
#endif

#endif
