#include "vulkan_memory.hpp"

#include "vk_utils.hpp"

using namespace std;

namespace v4r {

namespace BufferUsageFlags {
    static const VkBufferUsageFlags stage =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    static const VkBufferUsageFlags geometry =
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
}

template<bool host_mapped>
void BufferDeleter<host_mapped>::operator()(VkBuffer buffer) const
{
    const DeviceState &dev = alloc_.dev;

    if constexpr(host_mapped) {
        dev.dt.unmapMemory(dev.hdl, mem_);
    }

    dev.dt.freeMemory(dev.hdl, mem_, nullptr);

    dev.dt.destroyBuffer(dev.hdl, buffer, nullptr);
}

StageBuffer::StageBuffer(VkBuffer buf, void *p,
                         BufferDeleter<true> deleter)
    : buffer(buf), ptr(p),
      deleter_(deleter)
{}

StageBuffer::~StageBuffer()
{
    deleter_(buffer);
}

LocalBuffer::LocalBuffer(VkBuffer buf,
                         BufferDeleter<false> deleter)
    : buffer(buf),
      deleter_(deleter)
{}

LocalBuffer::~LocalBuffer()
{
    deleter_(buffer);
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

static MemoryTypeIndices findTypeIndices(const DeviceState &dev,
                                         const InstanceState &inst)
{
    VkPhysicalDeviceMemoryProperties2 dev_mem_props;
    dev_mem_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    dev_mem_props.pNext = nullptr;
    inst.dt.getPhysicalDeviceMemoryProperties2(dev.phy, &dev_mem_props);

    VkMemoryRequirements stage_reqs =
        getBufferMemReqs(BufferUsageFlags::stage, dev);

    uint32_t stage_type_idx = findMemoryTypeIndex(stage_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            dev_mem_props);

    VkMemoryRequirements geometry_reqs =
            getBufferMemReqs(BufferUsageFlags::geometry, dev);

    uint32_t geometry_type_idx = findMemoryTypeIndex(geometry_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    return MemoryTypeIndices {
        stage_type_idx,
        geometry_type_idx
    };
}

MemoryAllocator::MemoryAllocator(const DeviceState &d,
                                 const InstanceState &inst)
    : dev(d),
      type_indices_(findTypeIndices(dev, inst))
{}

StageBuffer MemoryAllocator::makeStagingBuffer(VkDeviceSize num_bytes)
{
    VkBufferCreateInfo buffer_info;
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = nullptr;
    buffer_info.flags = 0;
    buffer_info.size = num_bytes;
    buffer_info.usage = BufferUsageFlags::stage;
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

    return StageBuffer(buffer, mapped_ptr,
                       BufferDeleter<true>(memory, *this));
}

LocalBuffer MemoryAllocator::makeGeometryBuffer(VkDeviceSize num_bytes)
{
    VkBufferCreateInfo buffer_info;
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = nullptr;
    buffer_info.flags = 0;
    buffer_info.size = num_bytes;
    buffer_info.usage = BufferUsageFlags::geometry;
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

    return LocalBuffer(buffer, BufferDeleter<false>(memory, *this));
}

}
