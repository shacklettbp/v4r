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

struct PerRenderDescriptorConfig {
    VkDescriptorSetLayout layout;
};

struct FramebufferConfig {
public:
    uint32_t numImagesWidePerBatch;
    uint32_t numImagesTallPerBatch;

    uint32_t width;
    uint32_t height;

    uint64_t colorLinearBytes;
    uint64_t depthLinearBytes;
    uint64_t totalLinearBytes;
};

struct FramebufferState {
public:
    LocalImage color;
    LocalImage depth;
    LocalImage linearDepth;

    std::array<VkImageView, 3> attachmentViews; 

    VkFramebuffer hdl;

    LocalBuffer resultBuffer;
    VkDeviceMemory resultMem;
};

template <size_t NumInputs, size_t NumAttrs,
          size_t NumShaders, size_t NumDescriptorLayouts>
struct PipelineConfig {
public:
    std::array<VkVertexInputBindingDescription, NumInputs> inputBindings;
    std::array<VkVertexInputAttributeDescription, NumAttrs> inputAttrs;

    std::array<std::pair<const char *, VkShaderStageFlagBits>, NumShaders>
        shaders;
    std::array<VkDescriptorSetLayout, NumDescriptorLayouts> descLayouts;

    PipelineConfig(
            typename std::add_rvalue_reference<decltype(inputBindings)>::type
                input_bindings,
            typename std::add_rvalue_reference<decltype(inputAttrs)>::type
                input_attrs,
            typename std::add_rvalue_reference<decltype(shaders)>::type
                s,
            typename std::add_rvalue_reference<decltype(descLayouts)>::type
                desc_layouts)
        : inputBindings(std::move(input_bindings)), 
          inputAttrs(std::move(input_attrs)),
          shaders(std::move(s)),
          descLayouts(std::move(desc_layouts))
    {}
};

// Deduction guide!
template<typename B, typename A, typename S, typename D>
PipelineConfig(B, A, S, D) ->
    PipelineConfig<ArraySize<B>::value, ArraySize<A>::value,
                   ArraySize<S>::value, ArraySize<D>::value>;

// FIXME separate out things like the layout, cache (maybe renderpass)
// into PipelineManager
struct PipelineState {
    std::array<VkShaderModule, 2> shaders;

    VkPipelineCache pipelineCache;
    VkPipelineLayout gfxLayout;
    VkPipeline gfxPipeline;
};

class CommandStreamState {
public:
    CommandStreamState(const InstanceState &inst,
                       const DeviceState &dev,
                       const PerRenderDescriptorConfig &desc_cfg,
                       VkRenderPass render_pass,
                       const PipelineState &textured_pipeline,
                       const PipelineState &vertex_color_pipeline,
                       const FramebufferConfig &fb_cfg,
                       const FramebufferState &fb,
                       MemoryAllocator &alc,
                       QueueManager &queue_manager,
                       uint32_t batch_size,
                       uint32_t render_width,
                       uint32_t render_height,
                       uint32_t stream_idx);
    CommandStreamState(const CommandStreamState &) = delete;
    CommandStreamState(CommandStreamState &&) = default;

    void render(const std::vector<Environment> &envs);

    VkDeviceSize getColorOffset() const { return color_buffer_offset_; }
    VkDeviceSize getDepthOffset() const { return depth_buffer_offset_; }
    int getSemaphoreFD() const;

    const InstanceState &inst;
    const DeviceState &dev;

    VkRenderPass renderPass;
    const PipelineState &texturedPipeline;
    const PipelineState &vertexColorPipeline;
    const FramebufferState &fb;

    const VkCommandPool gfxPool;
    const QueueState &gfxQueue;

    MemoryAllocator &alloc;
    const VkSemaphore semaphore;

private:
    VkCommandBuffer render_cmd_;
    VkCommandBuffer copy_cmd_;
    VkDescriptorPool per_render_pool_;
    VkDescriptorSet per_render_descriptor_;
    HostBuffer transform_ssbo_;
    HostBuffer material_params_ssbo_;

    glm::u32vec2 fb_pos_;
    glm::u32vec2 render_size_;
    glm::u32vec2 render_extent_;
    VkDeviceSize color_buffer_offset_;
    VkDeviceSize depth_buffer_offset_;
    std::vector<glm::u32vec2> batch_offsets_;
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
    VkRenderPass renderPass;
    const PipelineState texturedPipeline;
    const PipelineState vertexColorPipeline;
    const FramebufferState fb;

    std::atomic_uint32_t numLoaders;
    std::atomic_uint32_t numStreams;
};

}

#endif
