#ifndef V4R_INTERNAL_UTILS_HPP_INCLUDED
#define V4R_INTERNAL_UTILS_HPP_INCLUDED

#include "vulkan_handles.hpp"

#include <vulkan/vulkan.h>
#include "cuda_runtime.h"

namespace v4r {

struct SyncState {
    const DeviceState &dev;
    const VkFence fence;
    cudaExternalSemaphore_t cudaSemaphore;
};

}

#endif
