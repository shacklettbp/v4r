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

    uint32_t miniBatchSize;
    uint32_t numImagesWidePerMiniBatch;
    uint32_t numImagesTallPerMiniBatch;

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

struct RTBindingTable {
    HostBuffer sbt;
    VkStridedDeviceAddressRegionKHR raygenEntry;
    VkStridedDeviceAddressRegionKHR missEntry;
    VkStridedDeviceAddressRegionKHR hitEntry;
    VkStridedDeviceAddressRegionKHR callableEntry;
};

struct RasterPipelineState {
    VkPipelineLayout gfxLayout;
    VkPipeline gfxPipeline;

    VkPipelineLayout meshCullLayout;
    VkPipeline meshCullPipeline;
};

struct RTPipelineState {
    VkPipelineLayout layout;
    VkPipeline pipeline;
    
    RTBindingTable bindingTable;
};

// FIXME separate out things like the layout, cache (maybe renderpass)
// into PipelineManager
struct PipelineState {
    std::vector<VkShaderModule> shaders;

    VkPipelineCache pipelineCache;

    std::optional<RasterPipelineState> rasterState;
    std::optional<RTPipelineState> rtState;
};

struct ParamBufferConfig {
    VkDeviceSize totalTransformBytes;

    VkDeviceSize viewOffset;
    VkDeviceSize totalViewBytes;

    VkDeviceSize materialIndicesOffset;
    VkDeviceSize totalMaterialIndexBytes;

    VkDeviceSize lightsOffset;
    VkDeviceSize totalLightParamBytes;

    VkDeviceSize cullInputOffset;
    VkDeviceSize totalCullInputBytes;

    VkDeviceSize totalParamBytes;

    VkDeviceSize countIndirectOffset;
    VkDeviceSize totalCountIndirectBytes;

    VkDeviceSize drawIndirectOffset;
    VkDeviceSize totalDrawIndirectBytes;

    VkDeviceSize totalIndirectBytes;
};

struct RenderState {
    ParamBufferConfig paramPositions;

    VkDescriptorSetLayout meshCullDescriptorLayout;
    VkDescriptorPool meshCullDescriptorPool;

    VkDescriptorSetLayout meshCullSceneDescriptorLayout;
    DescriptorManager::MakePoolType makeMeshCullScenePool;

    VkDescriptorSetLayout frameDescriptorLayout;
    VkDescriptorPool frameDescriptorPool;

    VkDescriptorSetLayout sceneDescriptorLayout;
    DescriptorManager::MakePoolType makeScenePool;

    VkDescriptorSetLayout rtDescriptorLayout;
    DescriptorManager::MakePoolType makeRTDescriptorPool;

    VkDescriptorSetLayout rtImageDescriptorLayout;
    VkDescriptorPool rtImageDescriptorPool;

    VkDescriptorSetLayout rtSceneDescriptorLayout;
    DescriptorManager::MakePoolType makeRTScenePool;

    VkSampler textureSampler;

    VkRenderPass renderPass;
};

template <typename PipelineType>
struct PipelineProps;

template <typename PipelineType>
struct PipelineImpl {
    static FramebufferConfig getFramebufferConfig(
            uint32_t batch_size, uint32_t img_width, uint32_t img_height,
            uint32_t num_streams, const RenderOptions &opts);

    static RenderState makeRenderState(const DeviceState &dev,
                                       uint32_t batch_size,
                                       uint32_t num_streams,
                                       const RenderOptions &opts,
                                       MemoryAllocator &alloc);

    static PipelineState makePipeline(const DeviceState &dev,
                                      const FramebufferConfig &fb_cfg,
                                      const RenderState &render_state,
                                      bool use_raster,
                                      bool use_rt,
                                      MemoryAllocator &alloc);
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
    VkFence fence;
    std::array<VkCommandBuffer, 2> commands;
    // indirectDrawBuffer starts with batch_size draw counts,
    // followed by the actual indirect draw commands
    VkDeviceSize indirectCountBaseOffset;
    VkDeviceSize indirectCountTotalBytes;
    VkDeviceSize indirectBaseOffset;
    DynArray<uint32_t> drawOffsets;
    DynArray<uint32_t> maxNumDraws;
    
    glm::u32vec2 baseFBOffset;
    DynArray<glm::u32vec2> batchFBOffsets;

    VkDeviceSize colorBufferOffset;
    VkDeviceSize depthBufferOffset;

    VkDescriptorSet cullSet;
    VkDescriptorSet frameSet;
    VkDescriptorSet rtSet;

    DynArray<VkBuffer> vertexBuffers;
    DynArray<VkDeviceSize> vertexOffsets;
    glm::mat4x3 *transformPtr;
    ViewInfo *viewPtr;
    uint32_t *materialPtr;
    LightProperties *lightPtr;
    uint32_t *numLightsPtr;
    DrawInput *drawPtr;
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
                       QueueState &graphics_queue,
                       uint32_t batch_size,
                       uint32_t stream_idx,
                       uint32_t num_frames_inflight,
                       bool cpu_sync,
                       bool enable_rt);

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

    uint32_t getCurrentFrame() const
    {
        return cur_frame_;
    }

    uint32_t getNumFrames() const
    {
        return frame_states_.size();
    }

    glm::u32vec2 getFrameExtent() const
    {
        return per_batch_render_size_;
    }

    EnvironmentState makeEnvironment(const std::shared_ptr<Scene> &scene,
                                     const glm::mat4 &perspective);

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
    LocalBuffer indirect_draw_buffer_;

    DescriptorManager rt_desc_mgr_;

    uint32_t mini_batch_size_;
    uint32_t num_mini_batches_;
    glm::u32vec2 per_elem_render_size_;
    glm::u32vec2 per_minibatch_render_size_;
    glm::u32vec2 per_batch_render_size_;
    std::vector<PerFrameState> frame_states_;
    uint32_t cur_frame_;
    bool enable_rt_;
    VkCommandBuffer tlas_build_cmd_;
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

    template <typename PipelineType>
    VulkanState(const RenderConfig &cfg,
                const RenderFeatures<PipelineType> &features,
                CoreVulkanHandles &&handles);

    template <typename PipelineType>
    VulkanState(const RenderConfig &cfg,
                const RenderFeatures<PipelineType> &features,
                bool use_raster,
                bool use_rt,
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

    DynArray<QueueState> transfer_queues_;
    DynArray<QueueState> graphics_queues_;
    DynArray<QueueState> compute_queues_;

    const uint32_t batch_size_;
    const bool double_buffered_;
    const bool cpu_sync_;
    const bool enable_rt_;
};

}

#ifndef VULKAN_STATE_INL_INCLUDED
#include "vulkan_state.inl"
#endif

#endif
