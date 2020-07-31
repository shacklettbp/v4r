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
#include "shader.hpp"
#include "utils.hpp"
#include "vulkan_handles.hpp"
#include "vulkan_memory.hpp"

namespace v4r {

struct FramebufferConfig {
public:
    uint32_t imgWidth;
    uint32_t imgHeight;

    uint32_t numImagesWidePerBatch;
    uint32_t numImagesTallPerBatch;

    uint32_t frameWidth;
    uint32_t frameHeight;
    uint32_t totalWidth;
    uint32_t totalHeight;

    uint64_t colorLinearBytesPerFrame;
    uint64_t depthLinearBytesPerFrame;
    uint64_t linearBytesPerFrame;

    uint64_t totalLinearBytes;

    bool colorOutput;
    bool depthOutput;

    std::vector<VkClearValue> clearValues;
};

// FIXME separate out things like the layout, cache (maybe renderpass)
// into PipelineManager
struct PipelineState {
    std::vector<VkShaderModule> shaders;

    VkPipelineCache pipelineCache;
    VkPipelineLayout gfxLayout;
    VkPipeline gfxPipeline;
};

struct ParamBufferConfig {
    VkDeviceSize totalTransformBytes;

    VkDeviceSize viewOffset;
    VkDeviceSize totalViewBytes;

    VkDeviceSize materialIndicesOffset;
    VkDeviceSize totalMaterialIndexBytes;

    VkDeviceSize lightsOffset;
    VkDeviceSize totalLightParamBytes;

    VkDeviceSize totalParamBytes;
};

struct RenderState {
    ParamBufferConfig paramPositions;

    VkDescriptorSetLayout frameDescriptorLayout;
    VkDescriptorPool frameDescriptorPool;

    VkDescriptorSetLayout sceneDescriptorLayout;
    VkSampler textureSampler;

    VkRenderPass renderPass;
};

template <typename PipelineType>
struct PipelineProps;

template <typename PipelineType>
struct PipelineImpl {
    static FramebufferConfig getFramebufferConfig(
            uint32_t batch_size, uint32_t img_width, uint32_t img_height,
            uint32_t num_streams);

    static RenderState makeRenderState(const DeviceState &dev,
                                       uint32_t batch_size,
                                       uint32_t num_streams,
                                       MemoryAllocator &alloc);

    static PipelineState makePipeline(const DeviceState &dev,
                                      const FramebufferConfig &fb_cfg,
                                      const RenderState &render_state);
};

struct FramebufferState {
public:
    std::vector<LocalImage> attachments;
    std::vector<VkImageView> attachmentViews; 

    VkFramebuffer hdl;

    LocalBuffer resultBuffer;
    VkDeviceMemory resultMem;
};

struct PerFrameState {
    VkSemaphore semaphore;
    VkFence fence;
    std::array<VkCommandBuffer, 2> commands;
    
    glm::u32vec2 baseFBOffset;
    DynArray<glm::u32vec2> batchFBOffsets;

    VkDeviceSize colorBufferOffset;
    VkDeviceSize depthBufferOffset;

    VkDescriptorSet frameSet;

    DynArray<VkBuffer> vertexBuffers;
    DynArray<VkDeviceSize> vertexOffsets;
    glm::mat4x3 *transformPtr;
    ViewInfo *viewPtr;
    uint32_t *materialPtr;
    LightProperties *lightPtr;
    uint32_t *numLightsPtr;
};

class CommandStreamState {
public:
    CommandStreamState(const InstanceState &inst,
                       const DeviceState &dev,
                       const FramebufferConfig &fb_cfg,
                       const RenderState &render_state,
                       const PipelineState &pipeline,
                       const FramebufferState &fb,
                       MemoryAllocator &alc,
                       QueueManager &queue_manager,
                       uint32_t batch_size,
                       uint32_t stream_idx,
                       uint32_t num_frames_inflight,
                       bool cpu_sync);

    CommandStreamState(const CommandStreamState &) = delete;
    CommandStreamState(CommandStreamState &&) = default;

    template <typename Fn>
    inline uint32_t render(const std::vector<Environment> &envs,
                           Fn &&submit_func);

    uint32_t render(const std::vector<Environment> &envs);

    VkImage getColorImage(uint32_t) const
    {
        return fb_.attachments[0].image;
    }

    glm::u32vec2 getFBOffset(uint32_t frame_idx) const
    {
        return frame_states_[frame_idx].baseFBOffset;
    }

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

    uint32_t getCurrentFrame() const {
        return cur_frame_;
    }

    uint32_t getNumFrames() const {
        return frame_states_.size();
    }

    glm::u32vec2 getFrameExtent() const {
        return render_extent_;
    }

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
    HostBuffer per_render_buffer_;

    glm::u32vec2 render_size_;
    glm::u32vec2 render_extent_;
    std::vector<PerFrameState> frame_states_;
    uint32_t cur_frame_;
};

struct CoreVulkanHandles {
    InstanceState inst;
    DeviceState dev;
};

class VulkanState {
public:
    template <typename PipelineType>
    VulkanState(const RenderConfig &cfg,
                const RenderFeatures<PipelineType> &features,
                const DeviceUUID &uuid);

    template <typename PipelineType,
              typename ImplType = PipelineImpl<PipelineType>>
    VulkanState(const RenderConfig &cfg,
                const RenderFeatures<PipelineType> &features,
                CoreVulkanHandles &&handles);

    LoaderState makeLoader();
    CommandStreamState makeStream();

    int getFramebufferFD() const;
    uint64_t getFramebufferBytes() const;

    glm::u32vec2 getImageDimensions() const
    {
        return glm::u32vec2(fbCfg.imgWidth, fbCfg.imgHeight);
    }

    bool isDoubleBuffered() const { return double_buffered_; }

    const InstanceState inst;
    const DeviceState dev;

    QueueManager queueMgr;
    MemoryAllocator alloc;

    const FramebufferConfig fbCfg;
    const RenderState renderState;
    const PipelineState pipeline;
    const FramebufferState fb;

    const glm::mat4 globalTransform;

private:
    const LoaderImpl loader_impl_;

    std::atomic_uint32_t num_loaders_;
    std::atomic_uint32_t num_streams_;

    const uint32_t max_num_loaders_;
    const uint32_t max_num_streams_;

    const uint32_t batch_size_;
    const bool double_buffered_;
    const bool cpu_sync_;
};

}

#ifndef VULKAN_STATE_INL_INCLUDED
#include "vulkan_state.inl"
#endif

#endif
