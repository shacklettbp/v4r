#ifndef VULKAN_MEMORY_HPP_INCLUDED
#define VULKAN_MEMORY_HPP_INCLUDED

#include "vulkan_handles.hpp"

namespace v4r {

class MemoryAllocator;

template<bool host_mapped>
class AllocDeleter {
public:
    AllocDeleter(VkDeviceMemory mem, MemoryAllocator &alloc)
        : mem_(mem), alloc_(alloc)
    {}

    void operator()(VkBuffer buffer) const;
    void operator()(VkImage image) const;

    void clear();

private:
    VkDeviceMemory mem_;

    MemoryAllocator &alloc_;
};

class HostBuffer {
public:
    HostBuffer(const HostBuffer &) = delete;
    HostBuffer(HostBuffer &&o);
    ~HostBuffer();

    void flush(const DeviceState &dev);

    VkBuffer buffer;
    void *ptr;
private:
    HostBuffer(VkBuffer buf, void *p,
               VkMappedMemoryRange mem_range,
               AllocDeleter<true> deleter);

    const VkMappedMemoryRange mem_range_;

    AllocDeleter<true> deleter_;
    friend class MemoryAllocator;
};

class LocalBuffer {
public:
    LocalBuffer(const LocalBuffer &) = delete;
    LocalBuffer(LocalBuffer &&o);
    ~LocalBuffer();

    VkBuffer buffer;
private:
    LocalBuffer(VkBuffer buf, AllocDeleter<false> deleter);

    AllocDeleter<false> deleter_;
    friend class MemoryAllocator;
};

class LocalTexture {
public:
    LocalTexture(const LocalTexture &) = delete;
    LocalTexture(LocalTexture &&o);
    ~LocalTexture();

    uint32_t width;
    uint32_t height;
    uint32_t mipLevels;
    VkImage image;
private:
    LocalTexture(uint32_t width, uint32_t height, uint32_t mip_levels,
                 VkImage image, AllocDeleter<false> deleter);

    AllocDeleter<false> deleter_;
    friend class MemoryAllocator;
};

struct MemoryTypeIndices {
    uint32_t stageBuffer;
    uint32_t uniformBuffer;
    uint32_t localGeometryBuffer;
    uint32_t precomputedMipmapTexture;
    uint32_t runtimeMipmapTexture;
};

class MemoryAllocator {
public:
    MemoryAllocator(const DeviceState &dev, const InstanceState &inst);
    MemoryAllocator(const MemoryAllocator &) = delete;
    MemoryAllocator(MemoryAllocator &&) = default;

    HostBuffer makeStagingBuffer(VkDeviceSize num_bytes);
    HostBuffer makeUniformBuffer(VkDeviceSize num_bytes);

    LocalBuffer makeGeometryBuffer(VkDeviceSize num_bytes);

    LocalTexture makeTexture(const VkImageCreateInfo &img_info,
                             bool precomputed_mipmaps=false);

private:
    const DeviceState &dev;
    MemoryTypeIndices type_indices_;

    template<bool> friend class AllocDeleter;
};

}

#endif
