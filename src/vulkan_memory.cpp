#include "vulkan_memory.hpp"

#include "vk_utils.hpp"

#include <cstring>
#include <iostream>

using namespace std;

namespace v4r {

namespace MemoryUsageFlags {
    static const VkBufferUsageFlags stage =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    static const VkBufferUsageFlags uniform =
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    static const VkBufferUsageFlags geometry =
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    static const VkImageUsageFlags precomputedMipmapTexture =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT;

    static const VkImageUsageFlags runtimeMipmapTexture =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT;
}

template<bool host_mapped>
void AllocDeleter<host_mapped>::operator()(VkBuffer buffer) const
{
    if (mem_ == VK_NULL_HANDLE) return;

    const DeviceState &dev = alloc_.dev;

    if constexpr(host_mapped) {
        dev.dt.unmapMemory(dev.hdl, mem_);
    }

    dev.dt.freeMemory(dev.hdl, mem_, nullptr);

    dev.dt.destroyBuffer(dev.hdl, buffer, nullptr);
}

template<>
void AllocDeleter<false>::operator()(VkImage image) const
{
    if (mem_ == VK_NULL_HANDLE) return;

    const DeviceState &dev = alloc_.dev;

    dev.dt.freeMemory(dev.hdl, mem_, nullptr);
    dev.dt.destroyImage(dev.hdl, image, nullptr);
}

template<bool host_mapped>
void AllocDeleter<host_mapped>::clear()
{
    mem_ = VK_NULL_HANDLE;
}

HostBuffer::HostBuffer(VkBuffer buf, void *p,
                       VkMappedMemoryRange mem_range,
                       AllocDeleter<true> deleter)
    : buffer(buf), ptr(p),
      mem_range_(mem_range),
      deleter_(deleter)
{}

HostBuffer::HostBuffer(HostBuffer &&o)
    : buffer(o.buffer),
      ptr(o.ptr),
      mem_range_(o.mem_range_),
      deleter_(o.deleter_)
{
    o.deleter_.clear();
}

HostBuffer::~HostBuffer()
{
    deleter_(buffer);
}

void HostBuffer::flush(const DeviceState &dev)
{
    dev.dt.flushMappedMemoryRanges(dev.hdl, 1, &mem_range_);
}

LocalBuffer::LocalBuffer(VkBuffer buf,
                         AllocDeleter<false> deleter)
    : buffer(buf),
      deleter_(deleter)
{}

LocalBuffer::LocalBuffer(LocalBuffer &&o)
    : buffer(o.buffer),
      deleter_(o.deleter_)
{
    o.deleter_.clear();
}

LocalBuffer::~LocalBuffer()
{
    deleter_(buffer);
}

LocalTexture::LocalTexture(uint32_t w, uint32_t h, uint32_t mip_levels,
                           VkImage img, AllocDeleter<false> deleter)
    : width(w), height(h), mipLevels(mip_levels),
      image(img),
      deleter_(deleter)
{}

LocalTexture::LocalTexture(LocalTexture &&o)
    : width(o.width), height(o.height), mipLevels(o.mipLevels),
      image(o.image),
      deleter_(o.deleter_)
{
    o.deleter_.clear();
}

LocalTexture::~LocalTexture()
{
    deleter_(image);
}

static VkMemoryRequirements getBufferMemReqs(VkBufferUsageFlags usage_flags,
                                             const DeviceState &dev)
{
    VkBufferCreateInfo buffer_info;
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = nullptr;
    buffer_info.flags = 0;
    buffer_info.size = 1;
    buffer_info.usage = usage_flags;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer test_buffer;
    REQ_VK(dev.dt.createBuffer(dev.hdl, &buffer_info, nullptr,
                               &test_buffer));

    VkMemoryRequirements reqs;
    dev.dt.getBufferMemoryRequirements(dev.hdl, test_buffer, &reqs);

    dev.dt.destroyBuffer(dev.hdl, test_buffer, nullptr);

    return reqs;
}

static VkMemoryRequirements getColorMemReqs(VkImageUsageFlags usage_flags,
                                            const DeviceState &dev)
{
    VkImageCreateInfo img_info;
    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.pNext = nullptr;
    img_info.flags = 0;
    img_info.imageType = VK_IMAGE_TYPE_2D;
    img_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    img_info.extent = { 1, 1, 1 };
    img_info.mipLevels = 1;
    img_info.arrayLayers = 1;
    img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage = usage_flags;
    img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    img_info.queueFamilyIndexCount = 0;
    img_info.pQueueFamilyIndices = nullptr;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage test_image;
    REQ_VK(dev.dt.createImage(dev.hdl, &img_info, nullptr, &test_image));

    VkMemoryRequirements reqs;
    dev.dt.getImageMemoryRequirements(dev.hdl, test_image, &reqs);

    dev.dt.destroyImage(dev.hdl, test_image, nullptr);

    return reqs;
}

static MemoryTypeIndices findTypeIndices(const DeviceState &dev,
                                         const InstanceState &inst)
{
    VkPhysicalDeviceMemoryProperties2 dev_mem_props;
    dev_mem_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    dev_mem_props.pNext = nullptr;
    inst.dt.getPhysicalDeviceMemoryProperties2(dev.phy, &dev_mem_props);

    VkMemoryRequirements stage_reqs =
        getBufferMemReqs(MemoryUsageFlags::stage, dev);

    uint32_t stage_type_idx = findMemoryTypeIndex(
            stage_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            dev_mem_props);

    VkMemoryRequirements uniform_reqs =
        getBufferMemReqs(MemoryUsageFlags::uniform, dev);

    uint32_t uniform_type_idx = findMemoryTypeIndex(
            uniform_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            dev_mem_props);

    VkMemoryRequirements geometry_reqs =
            getBufferMemReqs(MemoryUsageFlags::geometry, dev);

    uint32_t geometry_type_idx = findMemoryTypeIndex(
            geometry_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements texture_precomp_mip_reqs =
        getColorMemReqs(MemoryUsageFlags::precomputedMipmapTexture, dev);

    uint32_t texture_precomp_idx = findMemoryTypeIndex(
            texture_precomp_mip_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements texture_runtime_mip_reqs =
        getColorMemReqs(MemoryUsageFlags::runtimeMipmapTexture, dev);

    uint32_t texture_runtime_idx = findMemoryTypeIndex(
            texture_runtime_mip_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    return MemoryTypeIndices {
        stage_type_idx,
        uniform_type_idx,
        geometry_type_idx,
        texture_precomp_idx,
        texture_runtime_idx
    };
}

MemoryAllocator::MemoryAllocator(const DeviceState &d,
                                 const InstanceState &inst)
    : dev(d),
      type_indices_(findTypeIndices(dev, inst))
{}

HostBuffer MemoryAllocator::makeStagingBuffer(VkDeviceSize num_bytes)
{
    VkBufferCreateInfo buffer_info;
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = nullptr;
    buffer_info.flags = 0;
    buffer_info.size = num_bytes;
    buffer_info.usage = MemoryUsageFlags::stage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    REQ_VK(dev.dt.createBuffer(dev.hdl, &buffer_info, nullptr, &buffer));

    VkMemoryRequirements reqs;
    dev.dt.getBufferMemoryRequirements(dev.hdl, buffer, &reqs);

    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_indices_.stageBuffer;

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindBufferMemory(dev.hdl, buffer, memory, 0));

    void *mapped_ptr;
    REQ_VK(dev.dt.mapMemory(dev.hdl, memory, 0, num_bytes, 0, &mapped_ptr));

    VkMappedMemoryRange range;
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.pNext = nullptr;
    range.memory = memory,
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;

    return HostBuffer(buffer, mapped_ptr, range,
                      AllocDeleter<true>(memory, *this));
}

HostBuffer MemoryAllocator::makeUniformBuffer(VkDeviceSize num_bytes)
{
    VkBufferCreateInfo buffer_info;
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = nullptr;
    buffer_info.flags = 0;
    buffer_info.size = num_bytes;
    buffer_info.usage = MemoryUsageFlags::uniform;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    REQ_VK(dev.dt.createBuffer(dev.hdl, &buffer_info, nullptr, &buffer));

    VkMemoryRequirements reqs;
    dev.dt.getBufferMemoryRequirements(dev.hdl, buffer, &reqs);

    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_indices_.uniformBuffer;

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindBufferMemory(dev.hdl, buffer, memory, 0));

    void *mapped_ptr;
    REQ_VK(dev.dt.mapMemory(dev.hdl, memory, 0, num_bytes, 0, &mapped_ptr));

    VkMappedMemoryRange range;
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.pNext = nullptr;
    range.memory = memory,
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;

    return HostBuffer(buffer, mapped_ptr, range,
                      AllocDeleter<true>(memory, *this));
}

LocalBuffer MemoryAllocator::makeGeometryBuffer(VkDeviceSize num_bytes)
{
    VkBufferCreateInfo buffer_info;
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = nullptr;
    buffer_info.flags = 0;
    buffer_info.size = num_bytes;
    buffer_info.usage = MemoryUsageFlags::geometry;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    REQ_VK(dev.dt.createBuffer(dev.hdl, &buffer_info, nullptr, &buffer));

    VkMemoryRequirements reqs;
    dev.dt.getBufferMemoryRequirements(dev.hdl, buffer, &reqs);

    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_indices_.localGeometryBuffer;

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindBufferMemory(dev.hdl, buffer, memory, 0));

    return LocalBuffer(buffer, AllocDeleter<false>(memory, *this));
}

LocalTexture MemoryAllocator::makeTexture(const VkImageCreateInfo &img_info,
                                          bool precomputed_mipmaps)
{
    VkImage texture_img;
    REQ_VK(dev.dt.createImage(dev.hdl, &img_info, nullptr, &texture_img));

    VkMemoryRequirements reqs;
    dev.dt.getImageMemoryRequirements(dev.hdl, texture_img, &reqs);

    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.allocationSize = reqs.size;
    if (precomputed_mipmaps) {
        alloc.memoryTypeIndex = type_indices_.precomputedMipmapTexture;
    } else {
        alloc.memoryTypeIndex = type_indices_.runtimeMipmapTexture;
    }

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindImageMemory(dev.hdl, texture_img, memory, 0));

    return LocalTexture(img_info.extent.width, img_info.extent.height,
                        img_info.mipLevels, texture_img,
                        AllocDeleter<false>(memory, *this));
}

}
