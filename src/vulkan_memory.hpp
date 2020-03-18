#ifndef VULKAN_MEMORY_HPP_INCLUDED
#define VULKAN_MEMORY_HPP_INCLUDED

#include "vulkan_handles.hpp"

namespace v4r {

class MemoryAllocator;

class StageBuffer {
public:
    VkBuffer buffer;
    void *ptr;

    ~StageBuffer();
private:
    StageBuffer(MemoryAllocator &alloc);

    VkDeviceMemory mem_;
    MemoryAllocator &alloc_;

    friend class MemoryAllocator;
};

class LocalBuffer {
public:
    VkBuffer buffer;

    ~LocalBuffer();

private:
    LocalBuffer(MemoryAllocator &alloc);

    MemoryAllocator &alloc_;

    friend class MemoryAllocator;
};

struct MemoryTypeIndices {
    uint32_t stageBuffer;
    uint32_t localGeometryBuffer;
};

class MemoryAllocator {
public:
    MemoryAllocator(const DeviceState &dev, const InstanceState &inst);
    MemoryAllocator(const MemoryAllocator &) = delete;
    MemoryAllocator(MemoryAllocator &&) = default;

    StageBuffer makeStagingBuffer(VkDeviceSize num_bytes);
    LocalBuffer makeGeometryBuffer(VkDeviceSize num_bytes);

private:
    const DeviceState &dev;
    MemoryTypeIndices type_indices_;
};

}

#endif
