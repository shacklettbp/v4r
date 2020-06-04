#include "vulkan_state.hpp"

#include "utils.hpp"
#include "vk_utils.hpp"
#include "vulkan_config.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <optional>
#include <vector>

#include <glm/gtx/string_cast.hpp>

using namespace std;

namespace v4r {

static PerSceneDescriptorConfig getSceneDescriptorConfig(
        const DeviceState &dev)
{
    VkSamplerCreateInfo sampler_info;
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.pNext = nullptr;
    sampler_info.flags = 0;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.mipLodBias = 0;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.maxAnisotropy = 0;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.minLod = 0;
    sampler_info.maxLod = VK_LOD_CLAMP_NONE;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;

    VkSampler sampler;
    REQ_VK(dev.dt.createSampler(dev.hdl, &sampler_info, nullptr, &sampler));

    return PerSceneDescriptorConfig {
        sampler,
        PerSceneDescriptorLayout::makeSetLayout(dev, nullptr, &sampler)
    };
}

static PerRenderDescriptorConfig getRenderDescriptorConfig(
        const DeviceState &dev)
{
    return PerRenderDescriptorConfig {
        PerRenderDescriptorLayout::makeSetLayout(dev, nullptr)
    };
}

static FramebufferConfig getFramebufferConfig(const RenderConfig &cfg)
{
    uint32_t batch_fb_images_wide = ceil(sqrt(cfg.batchSize));
    while (cfg.batchSize % batch_fb_images_wide != 0) {
        batch_fb_images_wide++;
    }

    uint32_t batch_fb_images_tall = (cfg.batchSize / batch_fb_images_wide);
    assert(batch_fb_images_wide * batch_fb_images_tall == cfg.batchSize);

    uint32_t fb_width = cfg.imgWidth * batch_fb_images_wide * cfg.numStreams;
    uint32_t fb_height = cfg.imgHeight * batch_fb_images_tall;

    uint64_t color_bytes = 4 * sizeof(uint8_t) * fb_width * fb_height;
    uint64_t depth_bytes = sizeof(float) * fb_width * fb_height;

    return FramebufferConfig {
        batch_fb_images_wide,
        batch_fb_images_tall,
        fb_width,
        fb_height,
        color_bytes,
        depth_bytes,
        color_bytes + depth_bytes
    };
}

static VkCommandPool makeCmdPool(const DeviceState &dev, uint32_t qf_idx)
{
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = qf_idx;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool pool;
    REQ_VK(dev.dt.createCommandPool(dev.hdl, &pool_info, nullptr, &pool));
    return pool;
}

static VkCommandBuffer makeCmdBuffer(const DeviceState &dev,
        VkCommandPool pool,
        VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY)
{
    VkCommandBufferAllocateInfo info;
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.pNext = nullptr;
    info.commandPool = pool;
    info.level = level;
    info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    REQ_VK(dev.dt.allocateCommandBuffers(dev.hdl, &info, &cmd));

    return cmd;
}

static VkQueue makeQueue(const DeviceState &dev,
                         uint32_t qf_idx, uint32_t queue_idx)
{
    VkDeviceQueueInfo2 queue_info;
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2;
    queue_info.pNext = nullptr;
    queue_info.flags = 0;
    queue_info.queueFamilyIndex = qf_idx;
    queue_info.queueIndex = queue_idx;

    VkQueue queue;
    dev.dt.getDeviceQueue2(dev.hdl, &queue_info, &queue);

    return queue;
}

static VkSemaphore makeBinarySemaphore(const DeviceState &dev)
{
    VkSemaphoreCreateInfo sema_info;
    sema_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sema_info.pNext = nullptr;
    sema_info.flags = 0;

    VkSemaphore sema;
    REQ_VK(dev.dt.createSemaphore(dev.hdl, &sema_info, nullptr, &sema));

    return sema;
}

static VkFence makeFence(const DeviceState &dev, bool pre_signal=false)
{
    VkFenceCreateInfo fence_info;
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.pNext = nullptr;
    if (pre_signal) {
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    } else {
        fence_info.flags = 0;
    }
    
    VkFence fence;
    REQ_VK(dev.dt.createFence(dev.hdl, &fence_info, nullptr, &fence));

    return fence;
}

static void waitForFenceInfinitely(const DeviceState &dev, VkFence fence)
{
    VkResult res;
    while ((res = dev.dt.waitForFences(dev.hdl, 1,
                                       &fence, VK_TRUE,
                                       ~0ull)) != VK_SUCCESS) {
        if (res != VK_TIMEOUT) {
            REQ_VK(res);
        }
    }
}

static VkRenderPass makeRenderPass(const DeviceState &dev,
                                   const ResourceFormats &fmts)
{
    array<VkAttachmentDescription, 3> attachment_descs {{
        {
            0,
            fmts.colorAttachment,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        },
        {
            0,
            fmts.linearDepthAttachment,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        },
        {
            0,
            fmts.depthAttachment,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        }
    }};

    array<VkAttachmentReference, 3> attachment_refs {{
        { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
        { 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
        { 2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL }
    }};

    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount =
        static_cast<uint32_t>(attachment_refs.size() - 1);
    subpass_desc.pColorAttachments = &attachment_refs[0];
    subpass_desc.pDepthStencilAttachment = &attachment_refs[2];

    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount =
        static_cast<uint32_t>(attachment_descs.size());
    render_pass_info.pAttachments = attachment_descs.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass_desc;
    render_pass_info.dependencyCount = 0;
    render_pass_info.pDependencies = nullptr;

    VkRenderPass render_pass;
    REQ_VK(dev.dt.createRenderPass(dev.hdl, &render_pass_info,
                                   nullptr, &render_pass));

    return render_pass;
}

static FramebufferState makeFramebuffer(const DeviceState &dev,
                                        MemoryAllocator &alloc,
                                        const FramebufferConfig &fb_cfg,
                                        VkRenderPass render_pass)
{
    LocalImage color = alloc.makeColorAttachment(fb_cfg.width, fb_cfg.height);
    LocalImage depth = alloc.makeDepthAttachment(fb_cfg.width, fb_cfg.height);
    LocalImage linear_depth =
        alloc.makeLinearDepthAttachment(fb_cfg.width, fb_cfg.height);

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = color.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = alloc.getFormats().colorAttachment;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    array<VkImageView, 3> views;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info,
                                  nullptr, &views[0]));

    view_info.image = linear_depth.image;
    view_info.format = alloc.getFormats().linearDepthAttachment;

    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info,
                                  nullptr, &views[1]));

    view_info.image = depth.image;
    view_info.format = alloc.getFormats().depthAttachment;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info,
                                  nullptr, &views[2]));

    VkFramebufferCreateInfo fb_info;
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;
    fb_info.flags = 0;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = static_cast<uint32_t>(views.size());
    fb_info.pAttachments = views.data();
    fb_info.width = fb_cfg.width;
    fb_info.height = fb_cfg.height;
    fb_info.layers = 1;

    VkFramebuffer fb_handle;
    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fb_info, nullptr, &fb_handle));

    auto [result_buffer, result_mem] =
        alloc.makeDedicatedBuffer(fb_cfg.totalLinearBytes);

    return FramebufferState {
        move(color),
        move(depth),
        move(linear_depth),
        views,
        fb_handle,
        move(result_buffer),
        result_mem
    };
}

static VkShaderModule loadShader(const DeviceState &dev,
                                 const string &base_name)
{
    const string full_path = string(STRINGIFY(SHADER_DIR)) + base_name;
    
    ifstream shader_file(full_path, ios::binary | ios::ate);

    streampos fend = shader_file.tellg();
    shader_file.seekg(0, ios::beg);
    streampos fbegin = shader_file.tellg();
    size_t file_size = fend - fbegin;

    if (file_size == 0) {
        cerr << "Empty shader file" << endl;
        fatalExit();
    }

    DynArray<char> shader_code(file_size);
    shader_file.read(shader_code.data(), file_size);
    shader_file.close();

    VkShaderModuleCreateInfo shader_info;
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.pNext = nullptr;
    shader_info.flags = 0;
    shader_info.codeSize = file_size;
    shader_info.pCode = reinterpret_cast<const uint32_t *>(shader_code.data());

    VkShaderModule shader_module;
    REQ_VK(dev.dt.createShaderModule(dev.hdl, &shader_info, nullptr,
                                     &shader_module));

    return shader_module;
}

// FIXME this stuff should be coming out of shader reflection probably
// PipelineManager would create pipelines from shader descriptions
// and each scene or mesh would reference a pipeline (or pipeline idx)
template <typename... LayoutType>
static auto getTexturedPipelineConfig(LayoutType... desc_layout)
{
    // Vertices
    array<VkVertexInputBindingDescription, 1> input_bindings {{
        {
            0,
            sizeof(TexturedVertex),
            VK_VERTEX_INPUT_RATE_VERTEX
        }
    }};

    array<VkVertexInputAttributeDescription, 2> input_attrs {{
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,
          offsetof(TexturedVertex, position) },
        { 1, 0, VK_FORMAT_R32G32_SFLOAT,
          offsetof(TexturedVertex, uv) }
    }};

    array shaders {
        pair("texture_unlit.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
        pair("texture_unlit.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
    };

    array desc_layouts {
        desc_layout
        ...
    };

    return PipelineConfig {
        move(input_bindings),
        move(input_attrs),
        move(shaders),
        move(desc_layouts)
    };
}

template <typename... LayoutType>
static auto getVertexColorPipelineConfig(LayoutType... desc_layout)
{
    // Vertices
    array<VkVertexInputBindingDescription, 1> input_bindings {{
        {
            0,
            sizeof(ColoredVertex),
            VK_VERTEX_INPUT_RATE_VERTEX
        }
    }};

    array<VkVertexInputAttributeDescription, 2> input_attrs {{
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,
          offsetof(ColoredVertex, position) },
        { 1, 0, VK_FORMAT_R8G8B8_UNORM,
          offsetof(ColoredVertex, color) }
    }};

    array shaders {
        pair("color_unlit.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
        pair("color_unlit.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
    };

    array desc_layouts {
        desc_layout
        ...
    };

    return PipelineConfig {
        move(input_bindings),
        move(input_attrs),
        move(shaders),
        move(desc_layouts)
    };
}

template<typename ConfigType>
static PipelineState makePipeline(const DeviceState &dev,
                                  const FramebufferConfig &fb_cfg,
                                  VkRenderPass render_pass,
                                  const ConfigType &cfg)
                                  
{
    // Pipeline cache (FIXME)
    VkPipelineCacheCreateInfo pcache_info {};
    pcache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkPipelineCache pipeline_cache;
    REQ_VK(dev.dt.createPipelineCache(dev.hdl, &pcache_info,
                                      nullptr, &pipeline_cache));

    array<VkShaderModule, ArraySize<decltype(cfg.shaders)>::value>
        shader_modules;
    array<VkPipelineShaderStageCreateInfo, shader_modules.size()>
        shader_stages;

    for (size_t shader_idx = 0; shader_idx < cfg.shaders.size();
         shader_idx++) {
        auto [shader_name, shader_stage_flag] = cfg.shaders[shader_idx];

        shader_modules[shader_idx] = loadShader(dev, shader_name);

        shader_stages[shader_idx] = {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            shader_stage_flag,
            shader_modules[shader_idx],
            "main",
            nullptr
        };
    }

    VkPipelineVertexInputStateCreateInfo vert_info;
    vert_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vert_info.pNext = nullptr;
    vert_info.flags = 0;
    vert_info.vertexBindingDescriptionCount =
        static_cast<uint32_t>(cfg.inputBindings.size());
    vert_info.pVertexBindingDescriptions = cfg.inputBindings.data();
    vert_info.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(cfg.inputAttrs.size());
    vert_info.pVertexAttributeDescriptions = cfg.inputAttrs.data();
    
    // Assembly
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info {};
    input_assembly_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_info.topology =
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly_info.primitiveRestartEnable = VK_FALSE;

    // Viewport
    VkRect2D scissors {
        { 0, 0 },
        { fb_cfg.width, fb_cfg.height }
    };

    VkPipelineViewportStateCreateInfo viewport_info {};
    viewport_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_info.viewportCount = 1;
    viewport_info.pViewports = nullptr;
    viewport_info.scissorCount = 1;
    viewport_info.pScissors = &scissors;

    // Multisample
    VkPipelineMultisampleStateCreateInfo multisample_info {};
    multisample_info.sType = 
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample_info.sampleShadingEnable = VK_FALSE;
    multisample_info.alphaToCoverageEnable = VK_FALSE;
    multisample_info.alphaToOneEnable = VK_FALSE;

    // Rasterization
    VkPipelineRasterizationStateCreateInfo raster_info {};
    raster_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster_info.depthClampEnable = VK_FALSE;
    raster_info.rasterizerDiscardEnable = VK_FALSE;
    raster_info.polygonMode = VK_POLYGON_MODE_FILL;
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT;
    raster_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster_info.depthBiasEnable = VK_FALSE;
    raster_info.lineWidth = 1.0f;
    
    // Depth/Stencil
    VkPipelineDepthStencilStateCreateInfo depth_info {};
    depth_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_info.depthTestEnable = VK_TRUE;
    depth_info.depthWriteEnable = VK_TRUE;
    depth_info.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depth_info.depthBoundsTestEnable = VK_FALSE;
    depth_info.stencilTestEnable = VK_FALSE;
    depth_info.back.compareOp = VK_COMPARE_OP_ALWAYS;

    // Blend
    VkPipelineColorBlendAttachmentState blend_attach {};
    blend_attach.blendEnable = VK_FALSE;
    blend_attach.colorWriteMask = 
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    array blend_attachments { blend_attach, blend_attach };

    VkPipelineColorBlendStateCreateInfo blend_info {};
    blend_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_info.logicOpEnable = VK_FALSE;
    blend_info.attachmentCount =
        static_cast<uint32_t>(blend_attachments.size());
    blend_info.pAttachments = blend_attachments.data();

    // Dynamic
    VkDynamicState dyn_viewport_enable = VK_DYNAMIC_STATE_VIEWPORT;

    VkPipelineDynamicStateCreateInfo dyn_info {};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = 1;
    dyn_info.pDynamicStates = &dyn_viewport_enable;

    // Layout configuration
    // One push constant for model matrix
    VkPushConstantRange model_const;
    model_const.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    model_const.offset = 0;
    model_const.size = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo pipeline_layout_info;
    pipeline_layout_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.pNext = nullptr;
    pipeline_layout_info.flags = 0;
    pipeline_layout_info.setLayoutCount =
        static_cast<uint32_t>(cfg.descLayouts.size());
    pipeline_layout_info.pSetLayouts = cfg.descLayouts.data();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &model_const;

    VkPipelineLayout pipeline_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &pipeline_layout_info,
                                       nullptr, &pipeline_layout));


    VkGraphicsPipelineCreateInfo pipeline_info;
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.pNext = nullptr;
    pipeline_info.flags = 0;
    pipeline_info.stageCount = static_cast<uint32_t>(shader_stages.size());
    pipeline_info.pStages = shader_stages.data();
    pipeline_info.pVertexInputState = &vert_info;
    pipeline_info.pInputAssemblyState = &input_assembly_info;
    pipeline_info.pTessellationState = nullptr;
    pipeline_info.pViewportState = &viewport_info;
    pipeline_info.pRasterizationState = &raster_info;
    pipeline_info.pMultisampleState = &multisample_info;
    pipeline_info.pDepthStencilState = &depth_info;
    pipeline_info.pColorBlendState = &blend_info;
    pipeline_info.pDynamicState = &dyn_info;
    pipeline_info.layout = pipeline_layout;
    pipeline_info.renderPass = render_pass;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;

    VkPipeline pipeline;
    REQ_VK(dev.dt.createGraphicsPipelines(dev.hdl, pipeline_cache, 1,
                                          &pipeline_info, nullptr,
                                          &pipeline));
 
    return PipelineState {
        shader_modules,
        pipeline_cache,
        pipeline_layout,
        pipeline
    };
}

QueueState::QueueState(VkQueue queue_hdl)
    : queue_hdl_(queue_hdl),
      num_users_(1),
      mutex_()
{
}

void QueueState::submit(const DeviceState &dev, uint32_t submit_count,
                        const VkSubmitInfo *pSubmits, VkFence fence) const
{
    // FIXME there is a race here if more users are added
    // while threads are already submitting
    if (num_users_ > 1) {
        mutex_.lock();
    }

    REQ_VK(dev.dt.queueSubmit(queue_hdl_, submit_count, pSubmits, fence));

    if (num_users_ > 1) {
        mutex_.unlock();
    }
}

QueueManager::QueueManager(const DeviceState &d)
    : dev(d),
      gfx_queues_(),
      cur_gfx_idx_(0),
      transfer_queues_(),
      cur_transfer_idx_(0),
      alloc_mutex_()
{}

QueueState & QueueManager::allocateQueue(uint32_t qf_idx,
                                         deque<QueueState> &queues,
                                         uint32_t &cur_queue_idx,
                                         uint32_t max_queues)
{
    scoped_lock lock(alloc_mutex_);

    if (queues.size() < max_queues) {
        queues.emplace_back(makeQueue(dev, qf_idx, queues.size()));

        return queues.back();
    }

    QueueState &cur_queue = queues[cur_queue_idx];
    cur_queue_idx = (cur_queue_idx + 1) % max_queues;

    cur_queue.incrUsers();

    return cur_queue;
}

SceneRenderState::SceneRenderState(const DeviceState &d,
                                   VkCommandBuffer render_cmd,
                                   VkDescriptorSet desc_set,
                                   const glm::u32vec2 &fb_offset,
                                   const glm::u32vec2 &render_size,
                                   MemoryAllocator &alloc)
    : dev(d),
      render_cmd_(render_cmd),
      fb_offset_(fb_offset),
      render_size_(render_size),
      desc_set_(desc_set),
      vp_ubo_(alloc.makeUniformBuffer(sizeof(PerViewUBO))),
      projection_(),
      view_(),
      transform_ssbo_()
{
    VkDescriptorBufferInfo buffer_info;
    buffer_info.buffer = vp_ubo_.buffer;
    buffer_info.offset = 0;
    buffer_info.range = sizeof(PerViewUBO);

    VkWriteDescriptorSet desc_update;
    desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_update.pNext = nullptr;
    desc_update.dstSet = desc_set_;
    desc_update.dstBinding = 0;
    desc_update.dstArrayElement = 0;
    desc_update.descriptorCount = 1;
    desc_update.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    desc_update.pImageInfo = nullptr;
    desc_update.pBufferInfo = &buffer_info;
    desc_update.pTexelBufferView = nullptr;
    dev.dt.updateDescriptorSets(dev.hdl, 1, &desc_update, 0, nullptr);
}

void SceneRenderState::setInstanceTransformBuffer(HostBuffer &&buffer)
{
    transform_ssbo_.emplace(move(buffer));
}

void SceneRenderState::setProjection(const glm::mat4 &projection)
{
    projection_ = projection;
}

glm::mat4 * SceneRenderState::getViewPtr()
{ 
    return &view_;
}

glm::mat4 * SceneRenderState::getInstanceTransformsPtr()
{ 
    return (glm::mat4 *)transform_ssbo_->ptr;
}

void SceneRenderState::record(const SceneState &scene,
                              const PipelineState &pipeline,
                              const FramebufferState &fb,
                              VkRenderPass render_pass)
{
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(render_cmd_, &begin_info));

    if (scene.textures.size() > 0) {
        dev.dt.cmdBindDescriptorSets(render_cmd_,
                                     VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     pipeline.gfxLayout, 1,
                                     1, &scene.textureSet.hdl,
                                     0, nullptr);
    }

    // FIXME this linkage is super fragile (0 is hardcoded as the
    // the vertex shader's per command stream binding for example)
    dev.dt.cmdBindDescriptorSets(render_cmd_,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline.gfxLayout, 0,
                                 1, &desc_set_,
                                 0, nullptr);

    dev.dt.cmdBindPipeline(render_cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline.gfxPipeline);
    

    array<VkClearValue, 3> clear_vals;
    clear_vals[0].color = {{ 0.f, 0.f, 0.f, 1.f }};
    clear_vals[1].color = {{ 0.f, 0.f, 0.f, 0.f }};
    clear_vals[2].depthStencil = { 1.f, 0 };

    VkRenderPassBeginInfo render_begin;
    render_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_begin.pNext = nullptr;
    render_begin.renderPass = render_pass;
    render_begin.framebuffer = fb.hdl;
    render_begin.renderArea.offset = {
        static_cast<int32_t>(fb_offset_.x),
        static_cast<int32_t>(fb_offset_.y) };
    render_begin.renderArea.extent = { render_size_.x, render_size_.y };
    render_begin.clearValueCount = static_cast<uint32_t>(clear_vals.size());
    render_begin.pClearValues = clear_vals.data();

    dev.dt.cmdBeginRenderPass(render_cmd_, &render_begin,
                              VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport;
    viewport.x = fb_offset_.x;
    viewport.y = fb_offset_.y;
    viewport.width = render_size_.x;
    viewport.height = render_size_.y;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    dev.dt.cmdSetViewport(render_cmd_, 0, 1, &viewport);

    VkDeviceSize vert_offset = 0;
    dev.dt.cmdBindVertexBuffers(render_cmd_, 0, 1, &scene.geometry.buffer,
                                &vert_offset);
    dev.dt.cmdBindIndexBuffer(render_cmd_, scene.geometry.buffer,
                              scene.indexOffset, VK_INDEX_TYPE_UINT32);

    for (const ObjectInstance &instance : scene.instances) {
        // FIXME select texture id from global array and
        // change instance system so meshes are grouped together
        const SceneMesh &mesh = scene.meshes[instance.meshIndex];

        const Material &mat = scene.materials[mesh.materialIndex];
        (void)mat;

        PushConstants consts {
            instance.modelTransform
        };

        dev.dt.cmdPushConstants(render_cmd_, pipeline.gfxLayout,
                                VK_SHADER_STAGE_VERTEX_BIT,
                                0, sizeof(PushConstants),
                                &consts);

        dev.dt.cmdDrawIndexed(render_cmd_, mesh.numIndices, 1,
                              mesh.startIndex, 0, 0);
    }

    dev.dt.cmdEndRenderPass(render_cmd_);

    REQ_VK(dev.dt.endCommandBuffer(render_cmd_));
}

void SceneRenderState::updateVP()
{
    PerViewUBO data {
        projection_ * view_
    };

    memcpy(vp_ubo_.ptr, &data, sizeof(PerViewUBO));
    vp_ubo_.flush(dev);
}

void SceneRenderState::flushInstanceTransforms()
{
    transform_ssbo_->flush(dev);
}

LoaderState::LoaderState(const DeviceState &d,
                         const PerSceneDescriptorConfig &scene_desc_cfg,
                         MemoryAllocator &alc,
                         QueueManager &queue_manager)
    : dev(d),
      gfxPool(makeCmdPool(dev, dev.gfxQF)),
      gfxQueue(queue_manager.allocateGraphicsQueue()),
      gfxCopyCommand(makeCmdBuffer(dev, gfxPool)),
      transferPool(makeCmdPool(dev, dev.transferQF)),
      transferQueue(queue_manager.allocateTransferQueue()),
      transferStageCommand(makeCmdBuffer(dev, transferPool)),
      semaphore(makeBinarySemaphore(dev)),
      fence(makeFence(dev)),
      alloc(alc),
      descriptorManager(dev, scene_desc_cfg.layout)
{}

static uint32_t getMipLevels(const Texture &texture)
{
    return static_cast<uint32_t>(
        floor(log2(max(texture.width, texture.height)))) + 1;
}

static void generateMips(const DeviceState &dev,
                         const VkCommandBuffer copy_cmd,
                         const vector<LocalImage> &gpu_textures,
                         DynArray<VkImageMemoryBarrier> &barriers)
{
    for (size_t texture_idx = 0; texture_idx < gpu_textures.size();
            texture_idx++) {
        const LocalImage &gpu_texture = gpu_textures[texture_idx];
        VkImageMemoryBarrier &barrier = barriers[texture_idx];
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        for (uint32_t mip_level = 1; mip_level < gpu_texture.mipLevels;
                mip_level++) {
            VkImageBlit blit_spec {};
            
            // Src
            blit_spec.srcSubresource =
                { VK_IMAGE_ASPECT_COLOR_BIT, mip_level - 1, 0, 1 };
            blit_spec.srcOffsets[1] =
                { static_cast<int32_t>(gpu_texture.width >> (mip_level - 1)),
                  static_cast<int32_t>(gpu_texture.height >> (mip_level - 1)),
                  1 };

            // Dst
            blit_spec.dstSubresource =
                { VK_IMAGE_ASPECT_COLOR_BIT, mip_level, 0, 1 };
            blit_spec.dstOffsets[1] =
                { static_cast<int32_t>(gpu_texture.width >> mip_level),
                  static_cast<int32_t>(gpu_texture.height >> mip_level),
                  1 };

            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.subresourceRange.baseMipLevel = mip_level;

            dev.dt.cmdPipelineBarrier(copy_cmd,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      0, 0, nullptr, 0, nullptr,
                                      1, &barrier);

            dev.dt.cmdBlitImage(copy_cmd,
                                gpu_texture.image,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                gpu_texture.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &blit_spec, VK_FILTER_LINEAR);

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

            dev.dt.cmdPipelineBarrier(copy_cmd,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      0, 0, nullptr, 0, nullptr,
                                      1, &barrier);
        }
    }
}

SceneState LoaderState::loadScene(SceneAssets &&assets)
{
    vector<HostBuffer> texture_stagings;
    vector<LocalImage> gpu_textures;
    for (const Texture &texture : assets.textures) {
        uint64_t texture_bytes = texture.width * texture.height *
            texture.num_channels * sizeof(uint8_t);

        HostBuffer texture_staging = alloc.makeStagingBuffer(texture_bytes);
        memcpy(texture_staging.ptr, texture.raw_image.data(), texture_bytes);
        texture_staging.flush(dev);

        texture_stagings.emplace_back(move(texture_staging));

        uint32_t mip_levels = getMipLevels(texture);
        gpu_textures.emplace_back(alloc.makeTexture(texture.width,
                                                    texture.height,
                                                    mip_levels));
    }

    bool textured_scene = gpu_textures.size() > 0;

    VkDeviceSize vertex_bytes = textured_scene ?
        assets.textured_vertices.size() * sizeof(TexturedVertex) :
        assets.colored_vertices.size() * sizeof(ColoredVertex);
    VkDeviceSize index_bytes = assets.indices.size() * sizeof(uint32_t);
    VkDeviceSize geometry_bytes = vertex_bytes + index_bytes;

    HostBuffer geo_staging = alloc.makeStagingBuffer(geometry_bytes);

    // Store vertex buffer immediately followed by index buffer
    if (textured_scene) {
        memcpy(geo_staging.ptr, assets.textured_vertices.data(), vertex_bytes);
    } else {
        memcpy(geo_staging.ptr, assets.colored_vertices.data(), vertex_bytes);
    }
    memcpy((uint8_t *)geo_staging.ptr + vertex_bytes, assets.indices.data(),
           index_bytes);
    geo_staging.flush(dev);

    LocalBuffer geometry = alloc.makeGeometryBuffer(geometry_bytes);

    // Start recording for transfer queue
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(transferStageCommand, &begin_info));

    // Copy vertex/index buffer onto GPU
    VkBufferCopy copy_settings {};
    copy_settings.size = geometry_bytes;
    dev.dt.cmdCopyBuffer(transferStageCommand, geo_staging.buffer,
                         geometry.buffer, 1, &copy_settings);

    // Set initial texture layouts
    DynArray<VkImageMemoryBarrier> barriers(gpu_textures.size());
    for (size_t i = 0; i < gpu_textures.size(); i++) {
        const LocalImage &gpu_texture = gpu_textures[i];
        VkImageMemoryBarrier &barrier = barriers[i];

        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = gpu_texture.image;
        barrier.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, 1, 0, 1
        };
    }

    if (textured_scene) {
        dev.dt.cmdPipelineBarrier(transferStageCommand,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 0, nullptr, 0, nullptr,
                                  barriers.size(), barriers.data());

        for (size_t i = 0; i < gpu_textures.size(); i++) {
            const HostBuffer &stage_buffer = texture_stagings[i];
            const LocalImage &gpu_texture = gpu_textures[i];
            VkBufferImageCopy copy_spec {};
            copy_spec.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy_spec.imageSubresource.mipLevel = 0;
            copy_spec.imageSubresource.baseArrayLayer = 0;
            copy_spec.imageSubresource.layerCount = 1;
            copy_spec.imageExtent =
                { gpu_texture.width, gpu_texture.height, 1 };

            dev.dt.cmdCopyBufferToImage(transferStageCommand,
                                        stage_buffer.buffer,
                                        gpu_texture.image,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        1, &copy_spec);
        }

        // Transfer queue relinquish mip level 0
        for (VkImageMemoryBarrier &barrier : barriers) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;;
            barrier.srcQueueFamilyIndex = dev.transferQF;
            barrier.dstQueueFamilyIndex = dev.gfxQF;
        }
    }

    // Transfer queue relinquish geometry (also barrier on geometry write)
    VkBufferMemoryBarrier geometry_barrier;
    geometry_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    geometry_barrier.pNext = nullptr;
    geometry_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    geometry_barrier.dstAccessMask = 0;
    geometry_barrier.srcQueueFamilyIndex = dev.transferQF;
    geometry_barrier.dstQueueFamilyIndex = dev.gfxQF;
    geometry_barrier.buffer = geometry.buffer;
    geometry_barrier.offset = 0;
    geometry_barrier.size = geometry_bytes;

    // Geometry & texture barrier execute.
    dev.dt.cmdPipelineBarrier(transferStageCommand,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr,
                              1, &geometry_barrier,
                              barriers.size(), barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(transferStageCommand));

    VkSubmitInfo copy_submit{};
    copy_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    copy_submit.commandBufferCount = 1;
    copy_submit.pCommandBuffers = &transferStageCommand;
    copy_submit.signalSemaphoreCount = 1;
    copy_submit.pSignalSemaphores = &semaphore;

    transferQueue.submit(dev, 1, &copy_submit, VK_NULL_HANDLE);

    // Start recording for graphics queue
    REQ_VK(dev.dt.beginCommandBuffer(gfxCopyCommand, &begin_info));

    // Finish moving geometry onto graphics queue family
    geometry_barrier.srcAccessMask = 0;
    geometry_barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                     VK_ACCESS_INDEX_READ_BIT;
    dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                              0, 0, nullptr,
                              1, &geometry_barrier,
                              0, nullptr);

    if (textured_scene)  {
        // Finish acquiring mip level 0 on graphics queue and transition layout
        for (VkImageMemoryBarrier &barrier : barriers) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcQueueFamilyIndex = dev.transferQF;
            barrier.dstQueueFamilyIndex = dev.gfxQF;
        }

        dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 0, nullptr, 0, nullptr,
                                  barriers.size(), barriers.data());

        generateMips(dev, gfxCopyCommand, gpu_textures, barriers);

        // Final layout transition for textures
        for (size_t texture_idx = 0; texture_idx < gpu_textures.size();
                texture_idx++) {
            const LocalImage &gpu_texture = gpu_textures[texture_idx];
            VkImageMemoryBarrier &barrier = barriers[texture_idx];

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = gpu_texture.mipLevels;
        }

        dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                  0, 0, nullptr, 0, nullptr,
                                  barriers.size(), barriers.data());

    }

    REQ_VK(dev.dt.endCommandBuffer(gfxCopyCommand));

    VkSubmitInfo gfx_submit{};
    gfx_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    gfx_submit.waitSemaphoreCount = 1;
    gfx_submit.pWaitSemaphores = &semaphore;
    VkPipelineStageFlags sema_wait_mask = 
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    gfx_submit.pWaitDstStageMask = &sema_wait_mask;
    gfx_submit.commandBufferCount = 1;
    gfx_submit.pCommandBuffers = &gfxCopyCommand;

    gfxQueue.submit(dev, 1, &gfx_submit, fence);

    waitForFenceInfinitely(dev, fence);
    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &fence));

    vector<VkImageView> texture_views;
    vector<VkDescriptorImageInfo> view_infos;
    texture_views.reserve(gpu_textures.size());
    view_infos.reserve(gpu_textures.size());
    for (const LocalImage &gpu_texture : gpu_textures) {
        VkImageViewCreateInfo view_info;
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.pNext = nullptr;
        view_info.flags = 0;
        view_info.image = gpu_texture.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = alloc.getFormats().sdrTexture;
        view_info.components = { 
            VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
            VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A
        };
        view_info.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, gpu_texture.mipLevels,
            0, 1
        };

        VkImageView view;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &view));

        texture_views.push_back(view);
        view_infos.emplace_back(VkDescriptorImageInfo {
            VK_NULL_HANDLE, // Immutable
            view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        });
    }

    assert(gpu_textures.size() <= VulkanConfig::max_textures);

    DescriptorSet texture_set = textured_scene ?
        descriptorManager.makeSet() :
        descriptorManager.emptySet();

    if (textured_scene) {
        VkWriteDescriptorSet desc_update;
        desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        desc_update.pNext = nullptr;
        desc_update.dstSet = texture_set.hdl;
        desc_update.dstBinding = 0;
        desc_update.dstArrayElement = 0;
        desc_update.descriptorCount = VulkanConfig::max_textures;
        desc_update.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        desc_update.pImageInfo = view_infos.data();
        desc_update.pBufferInfo = nullptr;
        desc_update.pTexelBufferView = nullptr;
        dev.dt.updateDescriptorSets(dev.hdl, 1, &desc_update, 0, nullptr);
    }

    return SceneState {
        move(gpu_textures),
        move(texture_views),
        move(assets.materials),
        move(texture_set),
        move(geometry),
        vertex_bytes,
        move(assets.meshes),
        move(assets.instances)
    };
}


static glm::u32vec2 computeFBPosition(uint32_t batch_idx, const FramebufferConfig &cfg, const glm::u32vec2 &render_size)
{
    return glm::u32vec2((batch_idx % cfg.numImagesWidePerBatch) * render_size.x,
                        (batch_idx / cfg.numImagesWidePerBatch) * render_size.y);
}

CommandStreamState::CommandStreamState(
        const InstanceState &i,
        const DeviceState &d,
        const PerRenderDescriptorConfig &desc_cfg,
        VkRenderPass render_pass,
        const PipelineState &textured_pipeline,
        const PipelineState &vertex_color_pipeline,
        const FramebufferConfig &fb_cfg,
        const FramebufferState &framebuffer,
        MemoryAllocator &alc,
        QueueManager &queue_manager,
        uint32_t batch_size,
        uint32_t render_width,
        uint32_t render_height,
        uint32_t stream_idx)
    : inst(i),
      dev(d),
      renderPass(render_pass),
      texturedPipeline(textured_pipeline),
      vertexColorPipeline(vertex_color_pipeline),
      fb(framebuffer),
      gfxPool(makeCmdPool(dev, dev.gfxQF)),
      gfxQueue(queue_manager.allocateGraphicsQueue()),
      copyCommand(makeCmdBuffer(dev, gfxPool)),
      alloc(alc),
      fence(makeFence(dev)),
      fb_pos_(stream_idx * batch_size * fb_cfg.numImagesWidePerBatch, 0),
      color_buffer_offset_(stream_idx * fb_cfg.totalLinearBytes),
      depth_buffer_offset_(color_buffer_offset_ + fb_cfg.colorLinearBytes),
      batch_desc_pool_(PerRenderDescriptorLayout::makePool(dev, batch_size)),
      commands_(),
      batch_state_()
{
    commands_.reserve(batch_size + 1);
    batch_state_.reserve(batch_size);

    glm::u32vec2 render_size(render_width, render_height);
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        commands_.push_back(makeCmdBuffer(dev, gfxPool));

        glm::u32vec2 elem_offset = 
            computeFBPosition(batch_idx, fb_cfg, render_size) + fb_pos_;

        VkDescriptorSet elem_desc_set = 
            makeDescriptorSet(dev, batch_desc_pool_, desc_cfg.layout);

        batch_state_.emplace_back(dev, commands_.back(), elem_desc_set,
                                  elem_offset, render_size, alloc);
    }

    commands_.push_back(makeCmdBuffer(dev, gfxPool));

    VkCommandBuffer copy_command = commands_.back();

    array<VkImageMemoryBarrier, 2> barriers {{
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            fb.color.image,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            }
        },
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            fb.linearDepth.image,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            }
        }
    }};

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(copy_command, &begin_info));
    dev.dt.cmdPipelineBarrier(copy_command,
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_DEPENDENCY_BY_REGION_BIT,
                              0, nullptr, 0, nullptr,
                              static_cast<uint32_t>(barriers.size()),
                              barriers.data());

    assert(fb_pos_.x == 0);

    VkBufferImageCopy copy_info;
    copy_info.bufferOffset = color_buffer_offset_;
    copy_info.bufferRowLength = 0;
    copy_info.bufferImageHeight = 0;
    copy_info.imageSubresource = {
        VK_IMAGE_ASPECT_COLOR_BIT,
        0, 0, 1
    };
    copy_info.imageOffset = {
        static_cast<int32_t>(fb_pos_.x),
        static_cast<int32_t>(fb_pos_.y),
        0
    };
    copy_info.imageExtent = {
        render_width * fb_cfg.numImagesWidePerBatch,
        render_height * fb_cfg.numImagesTallPerBatch,
        1
    };

    dev.dt.cmdCopyImageToBuffer(copy_command,
                                fb.color.image,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                fb.resultBuffer.buffer,
                                1,
                                &copy_info);

    copy_info.bufferOffset = depth_buffer_offset_;

    dev.dt.cmdCopyImageToBuffer(copy_command,
                                fb.linearDepth.image,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                fb.resultBuffer.buffer,
                                1,
                                &copy_info);

    REQ_VK(dev.dt.endCommandBuffer(copy_command));
}

TransformPointers CommandStreamState::setSceneRenderState(uint32_t batch_idx,
        const glm::mat4 &projection,
        const SceneState &scene)
{
    const PipelineState &scene_pipeline =
        (scene.textures.size() > 0) ? texturedPipeline : vertexColorPipeline;

    SceneRenderState &batch_elem = batch_state_[batch_idx];

    // FIXME
    batch_elem.setInstanceTransformBuffer(
        alloc.makeUniformBuffer(sizeof(glm::mat4) * scene.instances.size()));

    batch_elem.setProjection(projection);
    batch_elem.record(scene, scene_pipeline, fb, renderPass);

    return TransformPointers {
        batch_elem.getViewPtr(),
        batch_elem.getInstanceTransformsPtr()
    };
}

void CommandStreamState::render()
{
    for (SceneRenderState &batch_elem : batch_state_) {
        batch_elem.updateVP();
        batch_elem.flushInstanceTransforms();
    }

    VkSubmitInfo gfx_submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0, nullptr, nullptr,
        static_cast<uint32_t>(commands_.size()), commands_.data(),
        0, nullptr
    };

    gfxQueue.submit(dev, 1, &gfx_submit, fence);
    waitForFenceInfinitely(dev, fence);
    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &fence));
}

VulkanState::VulkanState(const RenderConfig &config)
    : cfg(config),
      inst(),
      dev(inst.makeDevice(cfg.gpuID, cfg.numStreams + cfg.numLoaders, 1, cfg.numLoaders)),
      queueMgr(dev),
      alloc(dev, inst),
      fbCfg(getFramebufferConfig(cfg)),
      streamDescCfg(getRenderDescriptorConfig(dev)),
      sceneDescCfg(getSceneDescriptorConfig(dev)),
      renderPass(makeRenderPass(dev, alloc.getFormats())),
      texturedPipeline(makePipeline(dev, fbCfg, renderPass,
                                    getTexturedPipelineConfig(
                                        streamDescCfg.layout,
                                        sceneDescCfg.layout))),
      vertexColorPipeline(makePipeline(dev, fbCfg, renderPass,
                                       getVertexColorPipelineConfig(
                                           streamDescCfg.layout))),
      fb(makeFramebuffer(dev, alloc, fbCfg, renderPass)),
      numLoaders(0),
      numStreams(0)
{}

LoaderState VulkanState::makeLoader()
{
    numLoaders++;
    assert(numLoaders <= cfg.numLoaders);

    return LoaderState(dev, sceneDescCfg,
                       alloc, queueMgr);
}

CommandStreamState VulkanState::makeStream()
{
    uint32_t stream_idx = numStreams++;
    assert(stream_idx < cfg.numStreams);

    return CommandStreamState(inst,
                              dev,
                              streamDescCfg,
                              renderPass,
                              texturedPipeline,
                              vertexColorPipeline,
                              fbCfg, fb,
                              alloc,
                              queueMgr,
                              cfg.batchSize,
                              cfg.imgWidth,
                              cfg.imgHeight,
                              stream_idx);
}

int VulkanState::getFramebufferFD() const
{
    VkMemoryGetFdInfoKHR fd_info;
    fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fd_info.pNext = nullptr;
    fd_info.memory = fb.resultMem;
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd;
    REQ_VK(dev.dt.getMemoryFdKHR(dev.hdl, &fd_info, &fd));

    return fd;
}

uint64_t VulkanState::getFramebufferBytes() const
{
    return fbCfg.totalLinearBytes;
}

}
