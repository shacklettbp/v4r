#ifndef VULKAN_STATE_HPP_INCLUDED
#define VULKAN_STATE_HPP_INCLUDED

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_precision.hpp>

#include <v4r/config.hpp>
#include <v4r/environment.hpp>

#include "descriptors.hpp"
#include "scene.hpp"
#include "utils.hpp"
#include "vulkan_handles.hpp"
#include "vulkan_memory.hpp"

namespace v4r {

// FIXME unify with shader
struct ViewInfo {
    glm::mat4 projection;
    glm::mat4 view;
};

struct PerRenderDescriptorConfig {
    VkDescriptorSetLayout layout;

    VkDeviceSize bytesPerTransform;
    VkDeviceSize totalTransformBytes;

    VkDeviceSize viewOffset;

    VkDeviceSize materialIndicesOffset;
    VkDeviceSize totalMaterialIndexBytes;

    VkDeviceSize lightsOffset;
    VkDeviceSize totalLightParamBytes;

    VkDeviceSize totalParamBytes;

    std::add_pointer_t<
        VkDescriptorPool(const DeviceState &, uint32_t)> makePool;
};

struct FramebufferConfig {
public:
    uint32_t numImagesWidePerBatch;
    uint32_t numImagesTallPerBatch;

    uint32_t width;
    uint32_t height;

    uint64_t colorLinearBytesPerFrame;
    uint64_t depthLinearBytesPerFrame;
    uint64_t linearBytesPerFrame;

    uint64_t totalLinearBytes;

    std::vector<VkClearValue> clearValues;
};

struct FramebufferState {
public:
    std::vector<LocalImage> attachments;
    std::vector<VkImageView> attachmentViews; 

    VkFramebuffer hdl;

    LocalBuffer resultBuffer;
    VkDeviceMemory resultMem;
};

struct PipelineConfig {
public:
    std::vector<VkVertexInputBindingDescription> inputBindings;
    std::vector<VkVertexInputAttributeDescription> inputAttrs;

    std::vector<std::pair<const std::string, VkShaderStageFlagBits>> shaders;
    std::vector<VkDescriptorSetLayout> descLayouts;
};

// FIXME separate out things like the layout, cache (maybe renderpass)
// into PipelineManager
struct PipelineState {
    std::vector<VkShaderModule> shaders;

    VkPipelineCache pipelineCache;
    VkPipelineLayout gfxLayout;
    VkPipeline gfxPipeline;
};

struct PerFrameState {
    VkSemaphore semaphore;
    VkFence fence;
    std::array<VkCommandBuffer, 2> commands;
    
    glm::u32vec2 baseFBOffset;
    DynArray<glm::u32vec2> batchFBOffsets;

    VkDeviceSize colorBufferOffset;
    VkDeviceSize depthBufferOffset;
};

class CommandStreamState {
public:
    CommandStreamState(const RenderFeatures &features,
                       const InstanceState &inst,
                       const DeviceState &dev,
                       const PerRenderDescriptorConfig &per_render_cfg,
                       VkDescriptorSet per_render_descriptor,
                       VkRenderPass render_pass,
                       const PipelineState &pipeline,
                       const FramebufferConfig &fb_cfg,
                       const FramebufferState &fb,
                       MemoryAllocator &alc,
                       QueueManager &queue_manager,
                       uint32_t batch_size,
                       uint32_t render_width,
                       uint32_t render_height,
                       uint32_t stream_idx,
                       uint32_t num_frames_inflight);

    CommandStreamState(const CommandStreamState &) = delete;
    CommandStreamState(CommandStreamState &&) = default;

    uint32_t render(const std::vector<Environment> &envs);

    VkDeviceSize getColorOffset(uint32_t frame_idx) const
    {
        return frame_states_[frame_idx].colorBufferOffset; 
    }

    VkDeviceSize getDepthOffset(uint32_t frame_idx) const
    { 
        return frame_states_[frame_idx].depthBufferOffset;
    }

    VkFence getFence(uint32_t frame_idx) const
    {
        return frame_states_[frame_idx].fence;
    }

    int getSemaphoreFD(uint32_t frame_idx) const;

    const InstanceState &inst;
    const DeviceState &dev;

    const PipelineState &pipeline;

    VkCommandPool gfxPool; // FIXME move all command pools into VulkanState
    const QueueState &gfxQueue;

    MemoryAllocator &alloc;

private:
    const FramebufferConfig &fb_cfg_;
    const FramebufferState &fb_;
    VkRenderPass render_pass_;
    VkDescriptorSet per_render_descriptor_;
    HostBuffer per_render_buffer_;
    glm::mat4 *transform_ptr_;
    VkDeviceSize bytes_per_txfm_;
    ViewInfo *view_ptr_;
    uint32_t *material_ptr_;
    LightProperties *light_ptr_;
    uint32_t *num_lights_ptr_;

    glm::u32vec2 render_size_;
    glm::u32vec2 render_extent_;
    std::vector<PerFrameState> frame_states_;
    uint32_t cur_frame_;
};

class VulkanState {
public:
    VulkanState(const RenderConfig &cfg, const DeviceUUID &uuid);
    VulkanState(const VulkanState &) = delete;
    VulkanState(VulkanState &&) = delete;

    LoaderState makeLoader();
    CommandStreamState makeStream();

    int getFramebufferFD() const;
    uint64_t getFramebufferBytes() const;

    const RenderConfig cfg;

    const InstanceState inst;
    const DeviceState dev;

    QueueManager queueMgr;
    MemoryAllocator alloc;

    const FramebufferConfig fbCfg;
    const PerRenderDescriptorConfig streamDescCfg;
    const PerSceneDescriptorConfig sceneDescCfg;
    VkDescriptorPool renderDescriptorPool;
    VkRenderPass renderPass;
    const PipelineState pipeline;
    const FramebufferState fb;

    std::atomic_uint32_t numLoaders;
    std::atomic_uint32_t numStreams;
};

}

#endif
