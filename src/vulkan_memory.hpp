#ifndef VULKAN_MEMORY_HPP_INCLUDED
#define VULKAN_MEMORY_HPP_INCLUDED

#include "vulkan_handles.hpp"

namespace v4r {

class MemoryAllocator;

template<bool host_mapped>
class BufferDeleter {
public:
    BufferDeleter(VkDeviceMemory mem, MemoryAllocator &alloc)
        : mem_(mem), alloc_(alloc)
    {}

    void operator()(VkBuffer buffer) const;

private:
    VkDeviceMemory mem_;

    MemoryAllocator &alloc_;
};

class StageBuffer {
public:
    VkBuffer buffer;
    void *ptr;

    ~StageBuffer();
private:
    StageBuffer(VkBuffer buf, void *p,
                BufferDeleter<true> deleter);

    BufferDeleter<true> deleter_;
    friend class MemoryAllocator;
};

class LocalBuffer {
public:
    VkBuffer buffer;

    ~LocalBuffer();

private:
    LocalBuffer(VkBuffer buf, BufferDeleter<false> deleter);

    BufferDeleter<false> deleter_;
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

    template<bool> friend class BufferDeleter;
};

}

#endif
