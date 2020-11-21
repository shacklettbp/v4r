#include "vulkan_memory.hpp"

#include "vk_utils.hpp"

#include <cstring>
#include <iostream>

using namespace std;

namespace v4r {

namespace BufferFlags {
    static constexpr VkBufferUsageFlags stageUsage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    static constexpr VkBufferUsageFlags shaderUsage =
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_RAY_TRACING_BIT_KHR;

    static constexpr VkBufferUsageFlags hostGenericUsage =
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | shaderUsage;

    static constexpr VkBufferUsageFlags geometryUsage =
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    static constexpr VkBufferUsageFlags indirectUsage =
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

    static constexpr VkBufferUsageFlags localGenericUsage =
        geometryUsage | shaderUsage |
        VK_BUFFER_USAGE_RAY_TRACING_BIT_KHR;

    static constexpr VkBufferUsageFlags dedicatedUsage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
};

namespace ImageFlags {
    static constexpr VkFormatFeatureFlags precomputedMipmapTextureReqs =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

    static constexpr VkImageUsageFlags precomputedMipmapTextureUsage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;

    static constexpr VkImageUsageFlags runtimeMipmapTextureUsage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;

    static constexpr VkFormatFeatureFlags runtimeMipmapTextureReqs =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT |
        VK_FORMAT_FEATURE_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

    static constexpr VkImageUsageFlags colorAttachmentUsage = 
        //VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    static constexpr VkFormatFeatureFlags colorAttachmentReqs =
        //VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;

    static constexpr VkImageUsageFlags depthAttachmentUsage = 
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    static constexpr VkFormatFeatureFlags depthAttachmentReqs =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
};

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

void HostBuffer::flush(const DeviceState &dev,
                       VkDeviceSize offset,
                       VkDeviceSize num_bytes)
{
    VkMappedMemoryRange sub_range = mem_range_;
    sub_range.offset = offset;
    sub_range.size = num_bytes;
    dev.dt.flushMappedMemoryRanges(dev.hdl, 1, &sub_range);
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

LocalImage::LocalImage(uint32_t w, uint32_t h, uint32_t mip_levels,
                       VkImage img, AllocDeleter<false> deleter)
    : width(w), height(h), mipLevels(mip_levels),
      image(img),
      deleter_(deleter)
{}

LocalImage::LocalImage(LocalImage &&o)
    : width(o.width), height(o.height), mipLevels(o.mipLevels),
      image(o.image),
      deleter_(o.deleter_)
{
    o.deleter_.clear();
}

LocalImage::~LocalImage()
{
    deleter_(image);
}

static VkFormatProperties2 getFormatProperties(const InstanceState &inst,
                                               VkPhysicalDevice phy,
                                               VkFormat fmt)
{
    VkFormatProperties2 props;
    props.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
    props.pNext = nullptr;

    inst.dt.getPhysicalDeviceFormatProperties2(phy, fmt, &props);
    return props;
}

template<size_t N>
static VkFormat chooseFormat(VkPhysicalDevice phy, const InstanceState &inst,
                             VkFormatFeatureFlags required_features,
                             const array<VkFormat, N> &desired_formats)
{
    for (auto fmt : desired_formats) {
        VkFormatProperties2 props = getFormatProperties(inst, phy, fmt);
        if ((props.formatProperties.optimalTilingFeatures &
                    required_features) == required_features) {
            return fmt;
        }
    }

    cerr << "Unable to find required features in given formats" << endl;
    fatalExit();
}

pair<VkBuffer, VkMemoryRequirements> makeUnboundBuffer(const DeviceState &dev,
        VkDeviceSize num_bytes, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo buffer_info;
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = nullptr;
    buffer_info.flags = 0;
    buffer_info.size = num_bytes;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    REQ_VK(dev.dt.createBuffer(dev.hdl, &buffer_info, nullptr, &buffer));

    VkMemoryRequirements reqs;
    dev.dt.getBufferMemoryRequirements(dev.hdl, buffer, &reqs);

    return pair(buffer, reqs);
}

pair<VkImage, VkMemoryRequirements> makeUnboundImage(const DeviceState &dev,
        uint32_t width, uint32_t height, uint32_t mip_levels,
        VkFormat format, VkImageUsageFlags usage)
{
    VkImageCreateInfo img_info;
    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.pNext = nullptr;
    img_info.flags = 0;
    img_info.imageType = VK_IMAGE_TYPE_2D;
    img_info.format = format;
    img_info.extent = { width, height, 1 };
    img_info.mipLevels = mip_levels;
    img_info.arrayLayers = 1;
    img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage = usage;
    img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    img_info.queueFamilyIndexCount = 0;
    img_info.pQueueFamilyIndices = nullptr;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage img;
    REQ_VK(dev.dt.createImage(dev.hdl, &img_info, nullptr, &img));

    VkMemoryRequirements reqs;
    dev.dt.getImageMemoryRequirements(dev.hdl, img, &reqs);

    return pair(img, reqs);
}


static uint32_t findMemoryTypeIndex(uint32_t allowed_type_bits,
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

static VkMemoryRequirements getBufferMemReqs(const DeviceState &dev,
                                             VkBufferUsageFlags usage_flags)
{
    auto [test_buffer, reqs] = makeUnboundBuffer(dev, 1, usage_flags);

    dev.dt.destroyBuffer(dev.hdl, test_buffer, nullptr);

    return reqs;
}

static VkMemoryRequirements getImageMemReqs(const DeviceState &dev,
                                            VkFormat format,
                                            VkImageUsageFlags usage_flags)
{
    auto [test_image, reqs] = makeUnboundImage(dev, 1, 1, 1, format,
                                               usage_flags);

    dev.dt.destroyImage(dev.hdl, test_image, nullptr);

    return reqs;
}

static MemoryTypeIndices findTypeIndices(const DeviceState &dev,
                                         const InstanceState &inst,
                                         const ResourceFormats &formats)
{
    VkPhysicalDeviceMemoryProperties2 dev_mem_props;
    dev_mem_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    dev_mem_props.pNext = nullptr;
    inst.dt.getPhysicalDeviceMemoryProperties2(dev.phy, &dev_mem_props);

    VkMemoryRequirements stage_reqs = 
        getBufferMemReqs(dev, BufferFlags::stageUsage);

    uint32_t stage_type_idx = findMemoryTypeIndex(
            stage_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            dev_mem_props);

    VkMemoryRequirements shader_reqs =
        getBufferMemReqs(dev, BufferFlags::shaderUsage);

    uint32_t shader_type_idx = findMemoryTypeIndex(
            shader_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            dev_mem_props);

    VkMemoryRequirements host_generic_reqs =
        getBufferMemReqs(dev, BufferFlags::hostGenericUsage);

    uint32_t host_generic_type_idx = findMemoryTypeIndex(
            host_generic_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            dev_mem_props);

    VkMemoryRequirements geometry_reqs =
        getBufferMemReqs(dev, BufferFlags::geometryUsage);

    uint32_t geometry_type_idx = findMemoryTypeIndex(
            geometry_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements indirect_reqs =
        getBufferMemReqs(dev, BufferFlags::geometryUsage);

    uint32_t indirect_type_idx = findMemoryTypeIndex(
            indirect_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements local_generic_reqs =
        getBufferMemReqs(dev, BufferFlags::localGenericUsage);

    uint32_t local_generic_type_idx = findMemoryTypeIndex(
            local_generic_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements dedicated_reqs =
        getBufferMemReqs(dev, BufferFlags::dedicatedUsage);

    uint32_t dedicated_type_idx = findMemoryTypeIndex(
            dedicated_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements texture_precomp_mip_reqs =
        getImageMemReqs(dev, formats.hdrTexture,
                        ImageFlags::precomputedMipmapTextureUsage);

    uint32_t texture_precomp_idx = findMemoryTypeIndex(
            texture_precomp_mip_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements texture_runtime_mip_reqs =
        getImageMemReqs(dev, formats.sdrTexture,
                        ImageFlags::runtimeMipmapTextureUsage);

    uint32_t texture_runtime_idx = findMemoryTypeIndex(
            texture_runtime_mip_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements color_attachment_reqs =
        getImageMemReqs(dev, formats.colorAttachment,
                        ImageFlags::colorAttachmentUsage);

    uint32_t color_attachment_idx = findMemoryTypeIndex(
            color_attachment_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements depth_attachment_reqs =
        getImageMemReqs(dev, formats.depthAttachment,
                        ImageFlags::depthAttachmentUsage);

    uint32_t depth_attachment_idx = findMemoryTypeIndex(
            depth_attachment_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    return MemoryTypeIndices {
        stage_type_idx,
        shader_type_idx,
        host_generic_type_idx,
        geometry_type_idx,
        indirect_type_idx,
        local_generic_type_idx,
        dedicated_type_idx,
        texture_precomp_idx,
        texture_runtime_idx,
        color_attachment_idx,
        depth_attachment_idx
    };
}

static Alignments getMemoryAlignments(const InstanceState &inst,
                                      VkPhysicalDevice phy)
{
    VkPhysicalDeviceProperties2 props {};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    inst.dt.getPhysicalDeviceProperties2(phy, &props);

    return Alignments {
        props.properties.limits.minUniformBufferOffsetAlignment,
        props.properties.limits.minStorageBufferOffsetAlignment
    };
}

MemoryAllocator::MemoryAllocator(const DeviceState &d,
                                 const InstanceState &inst)
    : dev(d),
      formats_ {
          chooseFormat(dev.phy, inst,
                       ImageFlags::precomputedMipmapTextureReqs,
                       array { VK_FORMAT_BC7_UNORM_BLOCK }),
          chooseFormat(dev.phy, inst,
                       ImageFlags::precomputedMipmapTextureReqs,
                       array { VK_FORMAT_R16G16B16A16_SFLOAT }),
          chooseFormat(dev.phy, inst,
                       ImageFlags::colorAttachmentReqs,
                       array { VK_FORMAT_R8G8B8A8_UNORM }),
          chooseFormat(dev.phy, inst,
                       ImageFlags::depthAttachmentReqs,
                       array { VK_FORMAT_D32_SFLOAT,
                               VK_FORMAT_D32_SFLOAT_S8_UINT }),
          chooseFormat(dev.phy, inst,
                       ImageFlags::colorAttachmentReqs,
                       array { VK_FORMAT_R32_SFLOAT })
      },
      type_indices_(findTypeIndices(dev, inst, formats_)),
      alignments_(getMemoryAlignments(inst, dev.phy))
{}

HostBuffer MemoryAllocator::makeHostBuffer(VkDeviceSize num_bytes,
                                           VkBufferUsageFlags usage,
                                           uint32_t mem_idx)
{
    auto [buffer, reqs] = makeUnboundBuffer(dev, num_bytes, usage);

    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = mem_idx;

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


HostBuffer MemoryAllocator::makeStagingBuffer(VkDeviceSize num_bytes)
{
    return makeHostBuffer(num_bytes, BufferFlags::stageUsage,
                          type_indices_.stageBuffer);
}

HostBuffer MemoryAllocator::makeShaderBuffer(VkDeviceSize num_bytes)
{
    return makeHostBuffer(num_bytes, BufferFlags::shaderUsage, 
                          type_indices_.shaderBuffer);
}

HostBuffer MemoryAllocator::makeHostBuffer(VkDeviceSize num_bytes)
{
    return makeHostBuffer(num_bytes, BufferFlags::hostGenericUsage,
                          type_indices_.hostGenericBuffer);
}

LocalBuffer MemoryAllocator::makeLocalBuffer(VkDeviceSize num_bytes,
                                             VkBufferUsageFlags usage,
                                             uint32_t mem_idx)
{
    auto [buffer, reqs] = makeUnboundBuffer(dev, num_bytes,
                                            usage);

    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = mem_idx;

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindBufferMemory(dev.hdl, buffer, memory, 0));

    return LocalBuffer(buffer, AllocDeleter<false>(memory, *this));
}

LocalBuffer MemoryAllocator::makeLocalBuffer(VkDeviceSize num_bytes)
{
    return makeLocalBuffer(num_bytes, BufferFlags::localGenericUsage,
                           type_indices_.localGenericBuffer);
}

LocalBuffer MemoryAllocator::makeGeometryBuffer(VkDeviceSize num_bytes)
{
    return makeLocalBuffer(num_bytes, BufferFlags::geometryUsage,
                           type_indices_.localGeometryBuffer);
}

LocalBuffer MemoryAllocator::makeIndirectBuffer(VkDeviceSize num_bytes)
{
    return makeLocalBuffer(num_bytes, BufferFlags::indirectUsage,
                           type_indices_.localIndirectBuffer);
}


pair<LocalBuffer, VkDeviceMemory> MemoryAllocator::makeDedicatedBuffer(
        VkDeviceSize num_bytes)
{
    auto [buffer, reqs] = makeUnboundBuffer(dev, num_bytes,
                                            BufferFlags::dedicatedUsage);

    VkMemoryDedicatedAllocateInfo dedicated;
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.pNext = nullptr;
    dedicated.image = VK_NULL_HANDLE;
    dedicated.buffer = buffer;
    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &dedicated;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_indices_.dedicatedBuffer;

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindBufferMemory(dev.hdl, buffer, memory, 0));

    return pair(LocalBuffer(buffer, AllocDeleter<false>(memory, *this)),
                memory);
}

LocalImage MemoryAllocator::makeTexture(uint32_t width, uint32_t height,
                                        uint32_t mip_levels,
                                        bool precomputed_mipmaps)
{
    assert(precomputed_mipmaps == true);

    auto [texture_img, reqs] = makeUnboundImage(dev, width, height, mip_levels,
            formats_.sdrTexture,
            precomputed_mipmaps ? ImageFlags::precomputedMipmapTextureUsage :
                ImageFlags::runtimeMipmapTextureUsage);

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

    return LocalImage(width, height,
                      mip_levels, texture_img,
                      AllocDeleter<false>(memory, *this));
}

LocalImage MemoryAllocator::makeDedicatedImage(uint32_t width, uint32_t height,
                                               uint32_t mip_levels,
                                               VkFormat format,
                                               VkImageUsageFlags usage,
                                               uint32_t type_idx)
{
    auto [img, reqs] = makeUnboundImage(dev, width, height, mip_levels,
                                        format, usage);
    
    VkMemoryDedicatedAllocateInfo dedicated;
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.pNext = nullptr;
    dedicated.image = img;
    dedicated.buffer = VK_NULL_HANDLE;
    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &dedicated;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_idx;

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindImageMemory(dev.hdl, img, memory, 0));

    return LocalImage(width, height, mip_levels, img,
                      AllocDeleter<false>(memory, *this));
}

LocalImage MemoryAllocator::makeColorAttachment(uint32_t width,
                                                uint32_t height)
{
    return makeDedicatedImage(width, height, 1, formats_.colorAttachment,
                              ImageFlags::colorAttachmentUsage,
                              type_indices_.colorAttachment);
}

LocalImage MemoryAllocator::makeDepthAttachment(uint32_t width,
                                                uint32_t height)
{
    return makeDedicatedImage(width, height, 1, formats_.depthAttachment,
                              ImageFlags::depthAttachmentUsage,
                              type_indices_.depthAttachment);
}

LocalImage MemoryAllocator::makeLinearDepthAttachment(uint32_t width,
                                                      uint32_t height)
{
    return makeDedicatedImage(width, height, 1, formats_.linearDepthAttachment,
                              ImageFlags::colorAttachmentUsage,
                              type_indices_.colorAttachment);
}

static VkDeviceSize alignOffset(VkDeviceSize offset, VkDeviceSize alignment)
{
    return ((offset + alignment - 1) / alignment) * alignment;
}

VkDeviceSize MemoryAllocator::alignUniformBufferOffset(
        VkDeviceSize offset) const
{
    return alignOffset(offset, alignments_.uniformBuffer);
}

VkDeviceSize MemoryAllocator::alignStorageBufferOffset(
        VkDeviceSize offset) const
{
    return alignOffset(offset, alignments_.storageBuffer);
}

VkDeviceMemory MemoryAllocator::allocateAccelerationStructureMemory(
    VkAccelerationStructureKHR as)
{
    VkAccelerationStructureMemoryRequirementsInfoKHR as_mem_info;
    as_mem_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_KHR;
    as_mem_info.pNext = nullptr;
    as_mem_info.type =
        VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_KHR;
    as_mem_info.buildType = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR;
    as_mem_info.accelerationStructure = as;

    VkMemoryRequirements2 as_mem_reqs;
    as_mem_reqs.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    as_mem_reqs.pNext = nullptr;
    dev.dt.getAccelerationStructureMemoryRequirementsKHR(dev.hdl, &as_mem_info,
                                                         &as_mem_reqs);

    VkMemoryAllocateInfo alloc_info;
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = nullptr;
    alloc_info.allocationSize = as_mem_reqs.memoryRequirements.size;
    // FIXME
    alloc_info.memoryTypeIndex = type_indices_.localGeometryBuffer;

    VkDeviceMemory as_memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc_info, nullptr, &as_memory));

    return as_memory;
}

LocalBuffer MemoryAllocator::makeAccelerationStructureScratchBuffer(
    VkAccelerationStructureKHR as)
{
    VkAccelerationStructureMemoryRequirementsInfoKHR as_mem_info;
    as_mem_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_KHR;
    as_mem_info.pNext = nullptr;
    as_mem_info.type =
        VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_KHR;
    as_mem_info.buildType = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR;
    as_mem_info.accelerationStructure = as;

    VkMemoryRequirements2 as_mem_reqs;
    as_mem_reqs.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    as_mem_reqs.pNext = nullptr;
    dev.dt.getAccelerationStructureMemoryRequirementsKHR(dev.hdl, &as_mem_info,
                                                         &as_mem_reqs);

    return makeLocalBuffer(as_mem_reqs.memoryRequirements.size,
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_RAY_TRACING_BIT_KHR,
                           type_indices_.localGeometryBuffer);
}

}
