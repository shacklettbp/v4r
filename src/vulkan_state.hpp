#ifndef VULKAN_STATE_HPP_INCLUDED
#define VULKAN_STATE_HPP_INCLUDED

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_precision.hpp>

#include <v4r/config.hpp>

#include "descriptors.hpp"
#include "utils.hpp"
#include "vulkan_handles.hpp"
#include "vulkan_memory.hpp"

namespace v4r {

struct PerViewUBO {
    glm::mat4 vp;
};

struct PushConstants {
    glm::mat4 modelTransform;
};

struct TexturedVertex {
    glm::vec3 position;
    glm::vec2 uv;
};

struct ColoredVertex {
    glm::vec3 position;
    glm::u8vec3 color;
};

struct Texture {
    uint32_t width;
    uint32_t height;
    uint32_t num_channels;

    ManagedArray<uint8_t> raw_image;
};

struct Material {
    std::optional<uint64_t> ambientTexture;
    glm::vec4 ambientColor;

    std::optional<uint64_t> diffuseTexture;
    glm::vec4 diffuseColor;

    std::optional<uint64_t> specularTexture;
    glm::vec4 specularColor;

    float shininess;
};

struct SceneMesh {
    uint32_t startIndex;
    uint32_t numIndices;
    size_t materialIndex;
};

struct ObjectInstance {
    glm::mat4 modelTransform;
    uint64_t meshIndex;
};

struct SceneAssets {
    std::vector<Texture> textures;
    std::vector<Material> materials;

    std::vector<TexturedVertex> textured_vertices;
    std::vector<ColoredVertex> colored_vertices;
    std::vector<uint32_t> indices;

    std::vector<SceneMesh> meshes;
    std::vector<ObjectInstance> instances;
};

struct SceneState {
    std::vector<LocalImage> textures;
    std::vector<VkImageView> texture_views;
    std::vector<Material> materials;
    DescriptorSet textureSet;
    LocalBuffer geometry;
    VkDeviceSize indexOffset;
    std::vector<SceneMesh> meshes;
    std::vector<ObjectInstance> instances;
};

struct PerSceneDescriptorConfig {
    VkSampler textureSampler;
    VkDescriptorSetLayout layout;
};

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

class QueueState {
public:
    QueueState(VkQueue queue_hdl);

    void incrUsers() {
        num_users_++;
    }

    void submit(const DeviceState &dev, uint32_t submit_count,
                const VkSubmitInfo *pSubmits, VkFence fence) const;

private:
    VkQueue queue_hdl_;
    uint32_t num_users_;
    mutable std::mutex mutex_;
};

class QueueManager {
public:
    QueueManager(const DeviceState &dev);

    QueueState & allocateGraphicsQueue() {
        return allocateQueue(dev.gfxQF, gfx_queues_,
                             cur_gfx_idx_, dev.numGraphicsQueues);
    }

    QueueState & allocateTransferQueue()
    { 
        return allocateQueue(dev.transferQF, transfer_queues_,
                             cur_transfer_idx_, dev.numTransferQueues);
    }

private:
    QueueState & allocateQueue(uint32_t qf_idx,
                               std::deque<QueueState> &queues,
                               uint32_t &cur_queue_idx,
                               uint32_t max_queues);

    const DeviceState &dev;
    std::deque<QueueState> gfx_queues_;
    uint32_t cur_gfx_idx_;
    std::deque<QueueState> transfer_queues_;
    uint32_t cur_transfer_idx_;

    std::mutex alloc_mutex_;
};

class LoaderState {
public:
    LoaderState(const DeviceState &dev,
                const PerSceneDescriptorConfig &scene_desc_cfg,
                MemoryAllocator &alc,
                QueueManager &queue_manager);

    SceneState loadScene(SceneAssets &&assets);
                
    const DeviceState &dev;

    const VkCommandPool gfxPool;
    const QueueState &gfxQueue;
    const VkCommandBuffer gfxCopyCommand;

    const VkCommandPool transferPool;
    const QueueState &transferQueue;
    const VkCommandBuffer transferStageCommand;

    const VkSemaphore semaphore;
    const VkFence fence;

    MemoryAllocator &alloc;
    DescriptorManager descriptorManager;
};

struct TransformPointers {
    glm::mat4 *instances;
    glm::mat4 *view;
};

class SceneRenderState {
public:
    SceneRenderState(const DeviceState &dev,
                     VkCommandBuffer render_cmd,
                     VkDescriptorSet desc_set,
                     const glm::u32vec2 &fb_offset,
                     const glm::u32vec2 &render_size,
                     MemoryAllocator &alloc);

    void setInstanceTransformBuffer(HostBuffer &&buffer);

    void setProjection(const glm::mat4 &projection);

    glm::mat4 *getViewPtr();
    glm::mat4 *getInstanceTransformsPtr();

    void record(const SceneState &scene, const PipelineState &pipeline,
                const FramebufferState &fb, VkRenderPass render_pass);

    void updateVP();
    void flushInstanceTransforms();

    VkCommandBuffer getCommand() const { return render_cmd_; }

private:
    const DeviceState &dev;
    VkCommandBuffer render_cmd_;
    glm::u32vec2 fb_offset_;
    glm::u32vec2 render_size_;
    VkDescriptorSet desc_set_;
    HostBuffer vp_ubo_;
    glm::mat4 projection_;
    glm::mat4 view_;
    std::optional<HostBuffer> transform_ssbo_;
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

    TransformPointers setSceneRenderState(uint32_t batch_idx,
                                          const glm::mat4 &projection,
                                          const SceneState &scene);

    void render();

    VkDeviceSize getColorOffset() const { return color_buffer_offset_; }
    VkDeviceSize getDepthOffset() const { return depth_buffer_offset_; }

    const InstanceState &inst;
    const DeviceState &dev;

    VkRenderPass renderPass;
    const PipelineState &texturedPipeline;
    const PipelineState &vertexColorPipeline;
    const FramebufferState &fb;

    const VkCommandPool gfxPool;
    const QueueState &gfxQueue;
    const VkCommandBuffer copyCommand;

    MemoryAllocator &alloc;

    VkFence fence; // FIXME remove

private:
    glm::u32vec2 fb_pos_;
    VkDeviceSize color_buffer_offset_;
    VkDeviceSize depth_buffer_offset_;
    VkDescriptorPool batch_desc_pool_;
    std::vector<VkCommandBuffer> commands_;
    std::vector<SceneRenderState> batch_state_;
};

struct VulkanState {
public:
    VulkanState(const RenderConfig &cfg);
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
