#include "vulkan_memory.hpp"

#include "vk_utils.hpp"
#include "vulkan_config.hpp"

#include <cstring>
#include <iostream>

using namespace std;

namespace v4r {

namespace BufferFlags {
    static constexpr VkBufferUsageFlags commonUsage =
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    static constexpr VkBufferUsageFlags stageUsage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    static constexpr VkBufferUsageFlags geometryUsage =
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    static constexpr VkBufferUsageFlags shaderUsage =
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    static constexpr VkBufferUsageFlags paramUsage =
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    static constexpr VkBufferUsageFlags hostUsage =
        stageUsage | shaderUsage | paramUsage;

    static constexpr VkBufferUsageFlags indirectUsage =
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

    static constexpr VkBufferUsageFlags localUsage =
        commonUsage | geometryUsage | shaderUsage | indirectUsage;

    static constexpr VkBufferUsageFlags dedicatedUsage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
};

namespace ImageFlags {
    static constexpr VkFormatFeatureFlags textureReqs =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

    static constexpr VkImageUsageFlags textureUsage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;

    static constexpr VkImageUsageFlags colorAttachmentUsage = 
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    static constexpr VkFormatFeatureFlags colorAttachmentReqs =
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
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

static pair<VkBuffer, VkMemoryRequirements> makeUnboundBuffer(
    const DeviceState &dev, VkDeviceSize num_bytes, VkBufferUsageFlags usage)
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

static VkImage makeImage(
    const DeviceState &dev, uint32_t width, uint32_t height,
    uint32_t mip_levels, VkFormat format, VkImageUsageFlags usage,
    VkImageCreateFlags img_flags = 0)
{
    VkImageCreateInfo img_info;
    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.pNext = nullptr;
    img_info.flags = img_flags;
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

    return img;
}

uint32_t findMemoryTypeIndex(uint32_t allowed_type_bits,
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

static VkMemoryRequirements getImageMemReqs(const DeviceState &dev,
                                            VkImage img)
{
    VkMemoryRequirements reqs;
    dev.dt.getImageMemoryRequirements(dev.hdl, img, &reqs);

    return reqs;
}

static VkSparseImageMemoryRequirements getSparseTextureMemReqs(
        const DeviceState &dev, VkImage img)
{
    uint32_t num_requirements;
    dev.dt.getImageSparseMemoryRequirements(dev.hdl, img, &num_requirements,
                                            nullptr);

    DynArray<VkSparseImageMemoryRequirements> reqs(num_requirements);
    dev.dt.getImageSparseMemoryRequirements(dev.hdl, img, &num_requirements,
                                            reqs.data());

    for (uint32_t i = 0; i < num_requirements; i++) {
        if (reqs[i].formatProperties.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT) {
            return reqs[i];
        }
    }

    cerr << "Unable to get memory requirements for sparse texture" << endl;
    fatalExit();
}

static MemoryTypeIndices findTypeIndices(const DeviceState &dev,
                                         const InstanceState &inst,
                                         const ResourceFormats &formats)
{
    auto get_generic_buffer_reqs = [&](VkBufferUsageFlags usage_flags) {
        auto [test_buffer, reqs] = makeUnboundBuffer(dev, 1, usage_flags);

        dev.dt.destroyBuffer(dev.hdl, test_buffer, nullptr);

        return reqs;
    };

    auto get_generic_image_reqs = [&](VkFormat format,
                                      VkImageUsageFlags usage_flags,
                                      VkImageCreateFlags img_flags = 0) {
        VkImage test_image = makeImage(dev, 1, 1, 1, format, usage_flags,
                                       img_flags);

        VkMemoryRequirements reqs = getImageMemReqs(dev, test_image);

        dev.dt.destroyImage(dev.hdl, test_image, nullptr);

        return reqs;
    };

    VkPhysicalDeviceMemoryProperties2 dev_mem_props;
    dev_mem_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    dev_mem_props.pNext = nullptr;
    inst.dt.getPhysicalDeviceMemoryProperties2(dev.phy, &dev_mem_props);

    VkMemoryRequirements host_generic_reqs = 
        get_generic_buffer_reqs(BufferFlags::hostUsage);

    uint32_t host_type_idx = findMemoryTypeIndex(
            host_generic_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            dev_mem_props);

    VkMemoryRequirements buffer_local_reqs =
        get_generic_buffer_reqs(BufferFlags::localUsage);

    VkMemoryRequirements tex_local_reqs =
        get_generic_image_reqs(formats.sdrTexture, ImageFlags::textureUsage,
                               VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                               VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT);

    uint32_t local_type_idx = findMemoryTypeIndex(
            buffer_local_reqs.memoryTypeBits & tex_local_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements dedicated_reqs =
        get_generic_buffer_reqs(BufferFlags::dedicatedUsage);

    uint32_t dedicated_type_idx = findMemoryTypeIndex(
            dedicated_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements color_attachment_reqs =
        get_generic_image_reqs(formats.colorAttachment,
                               ImageFlags::colorAttachmentUsage);

    uint32_t color_attachment_idx = findMemoryTypeIndex(
            color_attachment_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    VkMemoryRequirements depth_attachment_reqs =
        get_generic_image_reqs(formats.depthAttachment,
                               ImageFlags::depthAttachmentUsage);

    uint32_t depth_attachment_idx = findMemoryTypeIndex(
            depth_attachment_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    return MemoryTypeIndices {
        host_type_idx,
        local_type_idx,
        dedicated_type_idx,
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

static SparseAttributes getSparseAttributes(const DeviceState &dev,
                                            VkFormat texture_format)
{
    VkImage test_img = makeImage(dev, 1, 1, 1,
        texture_format, ImageFlags::textureUsage,
        VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
        VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT);

    auto mem_reqs = getImageMemReqs(dev, test_img);
    auto sparse_reqs = getSparseTextureMemReqs(dev, test_img);

    dev.dt.destroyImage(dev.hdl, test_img, nullptr);

    glm::u32vec2 tile_dim(
        sparse_reqs.formatProperties.imageGranularity.width,
        sparse_reqs.formatProperties.imageGranularity.height);
    uint32_t tile_bytes = mem_reqs.alignment;
    uint32_t tail_bytes = sparse_reqs.imageMipTailSize;

    // 28.4.2 Miptail size is guaranteed to be an integer multiple of the
    // sparse block size in bytes FIXME add support for multiples
    if (tile_bytes != tail_bytes) {
        cerr << "Mip tail larger than sparse block size not supported" << endl;
        fatalExit();
    }

    return SparseAttributes {
        tile_dim,
        tile_bytes,
    };
}

MemoryAllocator::MemoryAllocator(const DeviceState &d,
                                 const InstanceState &inst,
                                 VkDeviceSize memory_budget,
                                 bool use_dynamic_blocks)
    : dev(d),
      formats_ {
          chooseFormat(dev.phy, inst,
                       ImageFlags::textureReqs,
                       array { VK_FORMAT_BC7_UNORM_BLOCK }),
          chooseFormat(dev.phy, inst,
                       ImageFlags::textureReqs,
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
      alignments_(getMemoryAlignments(inst, dev.phy)),
      sparse_(getSparseAttributes(dev, formats_.sdrTexture)),
      texture_memory_(),
      freelist_store_([&]() {
          uint32_t num_full_backing_allocations =
              memory_budget / VulkanConfig::texture_backing_size;

          uint32_t tiles_per_allocation =
              VulkanConfig::texture_backing_size / sparse_.tileBytes;

          VkDeviceSize overflow = memory_budget -
              num_full_backing_allocations *
                  VulkanConfig::texture_backing_size;

          uint32_t extra_tiles = overflow / sparse_.tileBytes;

          assert(VulkanConfig::texture_backing_size % sparse_.tileBytes == 0);

          return num_full_backing_allocations * tiles_per_allocation +
              extra_tiles;
      }()),
      freelist_head_(Head {
          0,
          0,
      })
{
    VkDeviceSize rounded_memory_budget =
        (memory_budget / sparse_.tileBytes) * sparse_.tileBytes;
    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.memoryTypeIndex = type_indices_.local;

    VkDeviceSize allocated_memory = 0;
    VkDeviceSize assigned_memory = 0;
    for (uint32_t chunk_id = 0; chunk_id < freelist_store_.size();
         chunk_id++) {
        if (assigned_memory == allocated_memory) {
            alloc.allocationSize =
                min(VulkanConfig::texture_backing_size,
                    rounded_memory_budget - allocated_memory);

            allocated_memory += alloc.allocationSize;

            VkDeviceMemory memory;
            REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
            texture_memory_.push_back(memory);
        }

        freelist_store_[chunk_id].chunk = MemoryChunk {
            texture_memory_.back(),
            uint32_t(assigned_memory % VulkanConfig::texture_backing_size),
            chunk_id,
        };

        assigned_memory += sparse_.tileBytes;

        freelist_store_[chunk_id].next.store(chunk_id + 1,
                                             memory_order_relaxed);
    }

    freelist_store_[freelist_store_.size() - 1].next.store(~0U,
        memory_order_relaxed);

    atomic_thread_fence(memory_order_release);
}

HostBuffer MemoryAllocator::makeHostBuffer(VkDeviceSize num_bytes,
                                           VkBufferUsageFlags usage)
{
    auto [buffer, reqs] = makeUnboundBuffer(dev, num_bytes, usage);

    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_indices_.host;

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
    return makeHostBuffer(num_bytes, BufferFlags::stageUsage);
}

HostBuffer MemoryAllocator::makeParamBuffer(VkDeviceSize num_bytes)
{
    return makeHostBuffer(num_bytes,
                          BufferFlags::commonUsage |
                              BufferFlags::shaderUsage |
                              BufferFlags::paramUsage);
}

LocalBuffer MemoryAllocator::makeLocalBuffer(VkDeviceSize num_bytes,
                                             VkBufferUsageFlags usage)
{
    auto [buffer, reqs] = makeUnboundBuffer(dev, num_bytes,
                                            usage);

    VkMemoryAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = type_indices_.local;

    VkDeviceMemory memory;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &alloc, nullptr, &memory));
    REQ_VK(dev.dt.bindBufferMemory(dev.hdl, buffer, memory, 0));

    return LocalBuffer(buffer, AllocDeleter<false>(memory, *this));
}

LocalBuffer MemoryAllocator::makeLocalBuffer(VkDeviceSize num_bytes)
{
    return makeLocalBuffer(num_bytes, BufferFlags::localUsage);
}

LocalBuffer MemoryAllocator::makeGeometryBuffer(VkDeviceSize num_bytes)
{
    return makeLocalBuffer(num_bytes,
                           BufferFlags::commonUsage |
                               BufferFlags::geometryUsage);
}

LocalBuffer MemoryAllocator::makeIndirectBuffer(VkDeviceSize num_bytes)
{
    return makeLocalBuffer(num_bytes,
                           BufferFlags::commonUsage |
                               BufferFlags::shaderUsage |
                               BufferFlags::indirectUsage);
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

VkDeviceSize MemoryAllocator::getMipTailOffset(VkImage image) const
{
    auto sparse_reqs = getSparseTextureMemReqs(dev, image);

    return sparse_reqs.imageMipTailOffset;
}

SparseTexture MemoryAllocator::makeTexture(uint32_t width, uint32_t height,
                                           uint32_t mip_levels)
{
    VkImage texture_img = makeImage(dev, width, height, mip_levels,
            formats_.sdrTexture, ImageFlags::textureUsage,
            VK_IMAGE_CREATE_SPARSE_BINDING_BIT | 
            VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT);

    VkImageViewCreateInfo view_info;
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.pNext = nullptr;
    view_info.flags = 0;
    view_info.image = texture_img;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = formats_.sdrTexture;
    view_info.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                            VK_COMPONENT_SWIZZLE_B,
                            VK_COMPONENT_SWIZZLE_A};
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                  mip_levels, 0, 1};

    VkImageView view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &view));

    return SparseTexture {
        width,
        height,
        mip_levels,
        texture_img,
        view,
    };
}

void MemoryAllocator::destroyTexture(SparseTexture &&texture)
{
    dev.dt.destroyImageView(dev.hdl, texture.view, nullptr);
    dev.dt.destroyImage(dev.hdl, texture.image, nullptr);
}

optional<MemoryChunk> MemoryAllocator::getChunk()
{
    Head cur_head = freelist_head_.load(memory_order_acquire);
    Head new_head;

    do {
        uint32_t cur_idx = cur_head.index;
        if (cur_idx == ~0U) return optional<MemoryChunk>();

        uint32_t next_idx =
            freelist_store_[cur_idx].next.load(std::memory_order_relaxed);

        new_head.index = next_idx;
        new_head.counter = cur_head.counter + 1;
    } while (!freelist_head_.compare_exchange_weak(cur_head,
                                                   new_head,
                                                   memory_order_release,
                                                   memory_order_acquire));

    return freelist_store_[cur_head.index].chunk;
}

void MemoryAllocator::returnChunk(MemoryChunk memory)
{
    FreeChunk &freelist_node = freelist_store_[memory.chunkID];

    Head cur_head = freelist_head_.load(memory_order_relaxed);

    Head new_head = { 
        memory.chunkID,
        0,
    };

    do {
        freelist_node.next.store(cur_head.index, memory_order_relaxed);
        new_head.counter = cur_head.counter + 1;
    } while (!freelist_head_.compare_exchange_weak(cur_head,
                                                   new_head,
                                                   memory_order_release,
                                                   memory_order_relaxed));
}

LocalImage MemoryAllocator::makeDedicatedImage(uint32_t width, uint32_t height,
                                               uint32_t mip_levels,
                                               VkFormat format,
                                               VkImageUsageFlags usage,
                                               uint32_t type_idx)
{
    auto img = makeImage(dev, width, height, mip_levels, format, usage);
    auto reqs = getImageMemReqs(dev, img);
    
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

}
