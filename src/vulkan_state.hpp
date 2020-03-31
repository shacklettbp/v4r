#ifndef VULKAN_STATE_HPP_INCLUDED
#define VULKAN_STATE_HPP_INCLUDED

#include <array>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <v4r/config.hpp>

#include "descriptors.hpp"
#include "utils.hpp"
#include "vulkan_handles.hpp"
#include "vulkan_memory.hpp"

namespace v4r {

struct PerStreamUBO {
    glm::mat4 vp;
};

struct PushConstants {
    glm::mat4 modelTransform;
};

struct Vertex {
    glm::vec3 position;
    glm::vec2 uv;
    glm::vec3 color;
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
    std::list<Texture> textures;
    std::vector<Material> materials;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    std::vector<SceneMesh> meshes;
    std::vector<ObjectInstance> instances;
};

struct SceneState {
    std::vector<LocalTexture> textures;
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

struct PerStreamDescriptorConfig {
    VkDescriptorSetLayout layout;
};

struct FramebufferConfig {
public:
    uint32_t width;
    uint32_t height;

    VkFormat colorFmt;
    VkImageCreateInfo colorCreationSettings;
    VkDeviceSize colorMemorySize;
    uint32_t colorMemoryTypeIdx;

    VkFormat depthFmt;
    VkImageCreateInfo depthCreationSettings;
    VkDeviceSize depthMemorySize;
    uint32_t depthMemoryTypeIdx;
};

struct FramebufferState {
public:
    VkImage colorImg;
    VkDeviceMemory colorMem;

    VkImage depthImg;
    VkDeviceMemory depthMem;

    std::array<VkImageView, 2> attachmentViews; 

    VkFramebuffer hdl;
};

// FIXME separate out things like the layout, cache (maybe renderpass)
// into a PipelineConfig type struct
struct PipelineState {
    VkRenderPass renderPass;

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

class StreamDescriptorState {
public:
    StreamDescriptorState(const DeviceState &dev,
                          const PerStreamDescriptorConfig &cfg,
                          MemoryAllocator &alloc);

    void bind(const DeviceState &dev, VkCommandBuffer cmd_buf,
              VkPipelineLayout pipe_layout);

    void update(const DeviceState &dev, const PerStreamUBO &data);


private:
    VkDescriptorPool pool_;
    VkDescriptorSet desc_set_;
    HostBuffer ubo_;
};

struct CommandStreamState {
public:
    CommandStreamState(const InstanceState &inst,
                       const DeviceState &dev,
                       const PerStreamDescriptorConfig &stream_desc_cfg,
                       const PerSceneDescriptorConfig &scene_desc_cfg,
                       const PipelineState &pl,
                       const FramebufferState &fb,
                       MemoryAllocator &alc,
                       QueueManager &queue_manager,
                       uint32_t render_width,
                       uint32_t render_height,
                       uint32_t stream_idx);
    CommandStreamState(const CommandStreamState &) = delete;
    CommandStreamState(CommandStreamState &&) = default;

    SceneState loadScene(SceneAssets &&assets);
    VkBuffer render(const SceneState &scene);

    const InstanceState &inst;
    const DeviceState &dev;
    const PipelineState &pipeline;
    const FramebufferState &fb;

    const VkCommandPool gfxPool;
    const QueueState &gfxQueue;
    const VkCommandBuffer gfxCopyCommand;
    const VkCommandBuffer gfxRenderCommand;
    const VkFence renderFence;

    const VkCommandPool transferPool;
    const QueueState &transferQueue;
    const VkCommandBuffer transferStageCommand;
    const VkSemaphore copySemaphore;
    const VkFence copyFence;

    MemoryAllocator &alloc;
    DescriptorManager descriptorManager;
    StreamDescriptorState streamDescState;

private:
    uint32_t fb_x_pos_;
    uint32_t fb_y_pos_;
    uint32_t render_width_;
    uint32_t render_height_;
};

struct VulkanState {
public:
    VulkanState(const RenderConfig &cfg);
    VulkanState(const VulkanState &) = delete;
    VulkanState(VulkanState &&) = delete;

    CommandStreamState makeStreamState();

    const RenderConfig cfg;

    const InstanceState inst;
    const DeviceState dev;
    MemoryAllocator alloc;
    QueueManager queueMgr;

    const FramebufferConfig fbCfg;
    const PerStreamDescriptorConfig streamDescCfg;
    const PerSceneDescriptorConfig sceneDescCfg;
    const PipelineState pipeline;
    const FramebufferState fb;

    uint32_t numStreams;
};

}

#endif
