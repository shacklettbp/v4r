#include "vulkan_state.hpp"

#include "utils.hpp"
#include "vk_utils.hpp"
#include "vulkan_config.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <optional>
#include <vector>

using namespace std;

namespace v4r {

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

static VkFormat getDeviceDepthFormat(VkPhysicalDevice phy,
                                     const InstanceState &inst)
{
    static const array desired_formats {
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D24_UNORM_S8_UINT
    };

    const VkFormatFeatureFlags desired_features =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;

    for (auto fmt : desired_formats) {
        VkFormatProperties2 props = getFormatProperties(inst, phy, fmt);
        if ((props.formatProperties.optimalTilingFeatures &
                    desired_features) == desired_features) {
            return fmt;
        }
    }

    cerr << "Unable to find required depth format" << endl;
    fatalExit();
}

static VkFormat getDeviceColorFormat(VkPhysicalDevice phy,
                                     const InstanceState &inst)
{
    static const array desired_formats {
        VK_FORMAT_R16G16B16A16_UNORM,
        VK_FORMAT_R16G16B16A16_SNORM,
        VK_FORMAT_R16G16B16A16_SFLOAT
    };

    const VkFormatFeatureFlags desired_features =
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;

    for (auto fmt : desired_formats) {
        VkFormatProperties2 props = getFormatProperties(inst, phy, fmt);
        if ((props.formatProperties.optimalTilingFeatures &
                    desired_features) == desired_features) {
            return fmt;
        }
    }

    cerr << "Unable to find required color format" << endl;
    fatalExit();
}

static FramebufferConfig getFramebufferConfig(const DeviceState &dev,
                                              const InstanceState &inst,
                                              const RenderConfig &cfg)
{
    const uint32_t num_single_dim = sqrt(VulkanConfig::num_images_per_fb);
    static_assert(VulkanConfig::num_images_per_fb % num_single_dim == 0);

    VkPhysicalDeviceMemoryProperties2 dev_mem_props;
    dev_mem_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    dev_mem_props.pNext = nullptr;
    inst.dt.getPhysicalDeviceMemoryProperties2(dev.phy, &dev_mem_props);

    VkFormat color_fmt = getDeviceColorFormat(dev.phy, inst);
    VkImageCreateInfo color_img_info {};
    color_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    color_img_info.imageType = VK_IMAGE_TYPE_2D;
    color_img_info.format = color_fmt;
    color_img_info.extent.width = cfg.imgWidth * num_single_dim;
    color_img_info.extent.height = cfg.imgHeight * num_single_dim;
    color_img_info.extent.depth = 1;
    color_img_info.mipLevels = 1;
    color_img_info.arrayLayers = 1;
    color_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    color_img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    color_img_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    color_img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    color_img_info.queueFamilyIndexCount = 0;
    color_img_info.pQueueFamilyIndices = nullptr;
    color_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // Create an image that will have the same settings as the real
    // framebuffer images later. This is necessary to cache the
    // memory requirements of the image.
    VkImage color_test;
    REQ_VK(dev.dt.createImage(dev.hdl, &color_img_info, nullptr, &color_test));

    VkMemoryRequirements color_req;
    dev.dt.getImageMemoryRequirements(dev.hdl, color_test, &color_req);

    dev.dt.destroyImage(dev.hdl, color_test, nullptr);

    uint32_t color_type_idx = findMemoryTypeIndex(color_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    // Set depth specific settings
    VkFormat depth_fmt = getDeviceDepthFormat(dev.phy, inst);

    VkImageCreateInfo depth_img_info = color_img_info;
    depth_img_info.format = depth_fmt;
    depth_img_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    VkImage depth_test;
    REQ_VK(dev.dt.createImage(dev.hdl, &depth_img_info, nullptr, &depth_test));

    VkMemoryRequirements depth_req;
    dev.dt.getImageMemoryRequirements(dev.hdl, depth_test, &depth_req);

    dev.dt.destroyImage(dev.hdl, depth_test, nullptr);

    uint32_t depth_type_idx = findMemoryTypeIndex(depth_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            dev_mem_props);

    return FramebufferConfig {
        cfg.imgWidth,
        cfg.imgHeight,
        color_fmt,
        color_img_info,
        color_req.size,
        color_type_idx,
        depth_fmt,
        depth_img_info,
        depth_req.size,
        depth_type_idx
    };
}

static VkCommandPool makeCmdPool(uint32_t qf_idx, const DeviceState &dev)
{
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = qf_idx;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool pool;
    REQ_VK(dev.dt.createCommandPool(dev.hdl, &pool_info, nullptr, &pool));
    return pool;
}

static VkCommandBuffer makeCmdBuffer(VkCommandPool pool,
        const DeviceState &dev,
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

static VkQueue makeQueue(uint32_t qf_idx, uint32_t queue_idx,
                         const DeviceState &dev)
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

static VkRenderPass makeRenderPass(const FramebufferConfig &fb_cfg,
                                   const DeviceState &dev)
{
    array<VkAttachmentDescription, 2> attachment_descs {{
        {
            0,
            fb_cfg.colorFmt,
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
            fb_cfg.depthFmt,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        }
    }};

    array<VkAttachmentReference, 2> attachment_refs {{
        { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
        { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL }
    }};

    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount = 1;
    subpass_desc.pColorAttachments = &attachment_refs[0];
    subpass_desc.pDepthStencilAttachment = &attachment_refs[1];

    const VkPipelineStageFlags write_stages =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

    // FIXME is the incoming memory dependency necessary here
    // to prevent overwriting with a new draw call before the transfer
    // out is finished

    array<VkSubpassDependency, 2> subpass_deps {{
        {
            VK_SUBPASS_EXTERNAL,
            0,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            write_stages,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
           0
        },
        {
            0,
            VK_SUBPASS_EXTERNAL,
            write_stages,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            0
        }
    }};

    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount =
        static_cast<uint32_t>(attachment_descs.size());
    render_pass_info.pAttachments = attachment_descs.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass_desc;
    render_pass_info.dependencyCount =
        static_cast<uint32_t>(subpass_deps.size());
    render_pass_info.pDependencies = subpass_deps.data();

    VkRenderPass render_pass;
    REQ_VK(dev.dt.createRenderPass(dev.hdl, &render_pass_info,
                                   nullptr, &render_pass));

    return render_pass;
}

static FramebufferState makeFramebuffer(const FramebufferConfig &fb_cfg,
                                        const PipelineState &pipeline,
                                        const DeviceState &dev)
{
    VkImage color_img;
    REQ_VK(dev.dt.createImage(dev.hdl, &fb_cfg.colorCreationSettings,
                              nullptr, &color_img));

    VkMemoryDedicatedAllocateInfo color_dedicated;
    color_dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    color_dedicated.pNext = nullptr;
    color_dedicated.image = color_img;
    color_dedicated.buffer = VK_NULL_HANDLE;
    VkMemoryAllocateInfo color_mem_info;
    color_mem_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    color_mem_info.pNext = &color_dedicated;
    color_mem_info.allocationSize = fb_cfg.colorMemorySize;
    color_mem_info.memoryTypeIndex = fb_cfg.colorMemoryTypeIdx;

    VkDeviceMemory color_mem;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &color_mem_info,
                                 nullptr, &color_mem));
    REQ_VK(dev.dt.bindImageMemory(dev.hdl, color_img, color_mem, 0));

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = color_img;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = fb_cfg.colorFmt;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    array<VkImageView, 2> views;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info,
                                  nullptr, &views[0]));

    VkImage depth_img;
    REQ_VK(dev.dt.createImage(dev.hdl, &fb_cfg.depthCreationSettings,
                              nullptr, &depth_img));

    VkMemoryDedicatedAllocateInfo depth_dedicated;
    depth_dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    depth_dedicated.pNext = nullptr;
    depth_dedicated.image = depth_img;
    depth_dedicated.buffer = VK_NULL_HANDLE;
    VkMemoryAllocateInfo depth_mem_info;
    depth_mem_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    depth_mem_info.pNext = &depth_dedicated;
    depth_mem_info.allocationSize = fb_cfg.depthMemorySize;
    depth_mem_info.memoryTypeIndex = fb_cfg.depthMemoryTypeIdx;

    VkDeviceMemory depth_mem;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &depth_mem_info,
                                 nullptr, &depth_mem));
    REQ_VK(dev.dt.bindImageMemory(dev.hdl, depth_img, depth_mem, 0));

    view_info.image = depth_img;
    view_info.format = fb_cfg.depthFmt;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT |
                              VK_IMAGE_ASPECT_STENCIL_BIT;

    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info,
                                  nullptr, &views[1]));

    VkFramebufferCreateInfo fb_info;
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;
    fb_info.flags = 0;
    fb_info.renderPass = pipeline.renderPass;
    fb_info.attachmentCount = static_cast<uint32_t>(views.size());
    fb_info.pAttachments = views.data();
    fb_info.width = fb_cfg.width;
    fb_info.height = fb_cfg.height;
    fb_info.layers = 1;

    VkFramebuffer fb_handle;
    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fb_info, nullptr, &fb_handle));

    return FramebufferState {
        color_img,
        color_mem,
        depth_img,
        depth_mem,
        views,
        fb_handle
    };
}

static VkShaderModule loadShader(const string &base_name,
                                 const DeviceState &dev)
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

static PipelineState makePipeline(const FramebufferConfig &fb_cfg,
                                  const DeviceState &dev)
{
    // Pipeline cache (FIXME)
    VkPipelineCacheCreateInfo pcache_info {};
    pcache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkPipelineCache pipeline_cache;
    REQ_VK(dev.dt.createPipelineCache(dev.hdl, &pcache_info,
                                      nullptr, &pipeline_cache));

    // Shaders
    array shader_modules {
        loadShader("unlit.vert.spv", dev),
        loadShader("unlit.frag.spv", dev)
    };

    array<VkPipelineShaderStageCreateInfo, 2> shader_stages {{
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_VERTEX_BIT,
            shader_modules[0],
            "main",
            nullptr
        },
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            shader_modules[1],
            "main",
            nullptr
        }
    }};

    // Vertices
    array<VkVertexInputBindingDescription, 1> input_bindings {{
        {
            0,
            sizeof(Vertex),
            VK_VERTEX_INPUT_RATE_VERTEX
        }
    }};

    array<VkVertexInputAttributeDescription, 2> input_attrs {{
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 }, // Position
        { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3}
    }};

    VkPipelineVertexInputStateCreateInfo vert_info;
    vert_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vert_info.pNext = nullptr;
    vert_info.flags = 0;
    vert_info.vertexBindingDescriptionCount =
        static_cast<uint32_t>(input_bindings.size());
    vert_info.pVertexBindingDescriptions = input_bindings.data();
    vert_info.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(input_attrs.size());
    vert_info.pVertexAttributeDescriptions = input_attrs.data();
    
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
        { 1, 1 }
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
    raster_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
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
    VkPipelineColorBlendAttachmentState blend_attach_state {};
    blend_attach_state.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend_info {};
    blend_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_info.logicOpEnable = VK_FALSE;
    blend_info.attachmentCount = 1;
    blend_info.pAttachments = &blend_attach_state;

    // Dynamic
    VkDynamicState dyn_viewport_enable = VK_DYNAMIC_STATE_VIEWPORT;

    VkPipelineDynamicStateCreateInfo dyn_info {};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = 1;
    dyn_info.pDynamicStates = &dyn_viewport_enable;

    // Layout configuration
    // One push constant for MVP matrix
    VkPushConstantRange mvp_consts;
    mvp_consts.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    mvp_consts.offset = 0;
    mvp_consts.size = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo pipeline_layout_info;
    pipeline_layout_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.pNext = nullptr;
    pipeline_layout_info.flags = 0;
    pipeline_layout_info.setLayoutCount = 0;
    pipeline_layout_info.pSetLayouts = nullptr;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &mvp_consts;

    VkPipelineLayout pipeline_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &pipeline_layout_info,
                                       nullptr, &pipeline_layout));


    // Make pipeline
    VkRenderPass render_pass = makeRenderPass(fb_cfg, dev);

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
        render_pass,
        shader_modules,
        pipeline_cache,
        pipeline_layout,
        pipeline
    };
}

CommandStreamState::CommandStreamState(const InstanceState &i,
                                       const DeviceState &d,
                                       const FramebufferConfig &fb_cfg,
                                       const PipelineState &pl,
                                       MemoryAllocator &alc)
    : inst(i),
      dev(d),
      pipeline(pl),
      gfxPool(makeCmdPool(dev.gfxQF, dev)),
      gfxQueue(makeQueue(dev.gfxQF, 0, dev)),
      gfxCopyCommand(makeCmdBuffer(gfxPool, dev)),
      transferPool(makeCmdPool(dev.transferQF, dev)),
      transferQueue(makeQueue(dev.transferQF, 0, dev)),
      transferStageCommand(makeCmdBuffer(transferPool, dev)),
      copySemaphore(makeBinarySemaphore(dev)),
      copyFence(makeFence(dev)),
      alloc(alc),
      fb(makeFramebuffer(fb_cfg, pipeline, dev))
{}

static uint32_t getMipLevels(const Texture &texture)
{
    return static_cast<uint32_t>(
        floor(log2(max(texture.width, texture.height)))) + 1;
}

SceneState CommandStreamState::loadScene(SceneAssets &&assets)
{
    vector<StageBuffer> texture_stagings;
    vector<LocalTexture> gpu_textures;
    for (const Texture &texture : assets.textures) {
        uint64_t texture_bytes = texture.width * texture.height *
            texture.num_channels * sizeof(uint8_t);

        StageBuffer texture_staging = alloc.makeStagingBuffer(texture_bytes);
        memcpy(texture_staging.ptr, texture.raw_image.data(), texture_bytes);

        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

        {
            VkFormatProperties2 format_props =
                getFormatProperties(inst, dev.phy, format);
            
            if ((format_props.formatProperties.optimalTilingFeatures &
                    VK_FORMAT_FEATURE_BLIT_SRC_BIT) == 0 || 
                (format_props.formatProperties.optimalTilingFeatures &
                    VK_FORMAT_FEATURE_BLIT_DST_BIT) == 0 ||
                (format_props.formatProperties.optimalTilingFeatures &
                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0) {
                cerr << "Device does not support features for mipmap gen" <<
                    endl;

                fatalExit();
            }
        }

        uint32_t mip_levels = getMipLevels(texture);

        VkImageCreateInfo img_info;
        img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_info.pNext = nullptr;
        img_info.flags = 0;
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.format = format;
        img_info.extent = { texture.width, texture.height, 1 };
        img_info.mipLevels = mip_levels;
        img_info.arrayLayers = 1;
        img_info.samples = VK_SAMPLE_COUNT_1_BIT;
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT;
        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.queueFamilyIndexCount = 0;
        img_info.pQueueFamilyIndices = nullptr;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        LocalTexture gpu_texture = alloc.makeTexture(img_info);

        texture_stagings.emplace_back(move(texture_staging));
        gpu_textures.emplace_back(alloc.makeTexture(img_info));
    }

    VkDeviceSize vertex_bytes = assets.vertices.size() * sizeof(Vertex);
    VkDeviceSize index_bytes = assets.indices.size() * sizeof(uint32_t);
    VkDeviceSize geometry_bytes = vertex_bytes + index_bytes;

    StageBuffer geo_staging = alloc.makeStagingBuffer(geometry_bytes);

    // Store vertex buffer immediately followed by index buffer
    memcpy(geo_staging.ptr, assets.vertices.data(), vertex_bytes);
    memcpy((uint8_t *)geo_staging.ptr + vertex_bytes, assets.indices.data(),
           index_bytes);
    LocalBuffer geometry = alloc.makeGeometryBuffer(geometry_bytes);

    // Start recording for transfer queue
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(transferStageCommand, &begin_info));

    // Copy textures and generate mipmaps
    DynArray<VkImageMemoryBarrier> barriers(gpu_textures.size());
    for (size_t i = 0; i < gpu_textures.size(); i++) {
        const LocalTexture &gpu_texture = gpu_textures[i];
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
    dev.dt.cmdPipelineBarrier(transferStageCommand,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr, 0, nullptr,
                              barriers.size(), barriers.data());

    for (size_t i = 0; i < gpu_textures.size(); i++) {
        const StageBuffer &stage_buffer = texture_stagings[i];
        const LocalTexture &gpu_texture = gpu_textures[i];
        VkBufferImageCopy copy_spec {};
        copy_spec.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_spec.imageSubresource.mipLevel = 0;
        copy_spec.imageSubresource.baseArrayLayer = 0;
        copy_spec.imageSubresource.layerCount = 1;
        copy_spec.imageExtent = { gpu_texture.width, gpu_texture.height, 1 };

        dev.dt.cmdCopyBufferToImage(transferStageCommand, stage_buffer.buffer,
                                    gpu_texture.image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    1, &copy_spec);
    }

    // Prepare mip level 0 to have ownership transferred
    for (VkImageMemoryBarrier &barrier : barriers) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;;
        barrier.srcQueueFamilyIndex = dev.transferQF;
        barrier.dstQueueFamilyIndex = dev.gfxQF;
    }

    dev.dt.cmdPipelineBarrier(transferStageCommand,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr, 0, nullptr,
                              barriers.size(), barriers.data());

    // Copy vertex/index buffer onto GPU
    VkBufferCopy copy_settings {};
    copy_settings.size = geometry_bytes;
    dev.dt.cmdCopyBuffer(transferStageCommand, geo_staging.buffer,
                         geometry.buffer, 1, &copy_settings);

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

    dev.dt.cmdPipelineBarrier(transferStageCommand,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr,
                              1, &geometry_barrier,
                              0, nullptr);

    REQ_VK(dev.dt.endCommandBuffer(transferStageCommand));

    VkSubmitInfo copy_submit{};
    copy_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    copy_submit.commandBufferCount = 1;
    copy_submit.pCommandBuffers = &transferStageCommand;
    copy_submit.signalSemaphoreCount = 1;
    copy_submit.pSignalSemaphores = &copySemaphore;

    REQ_VK(dev.dt.queueSubmit(transferQueue, 1, &copy_submit,
                              VK_NULL_HANDLE));

    // Start recording for graphics queue
    REQ_VK(dev.dt.beginCommandBuffer(gfxCopyCommand, &begin_info));

    // Finish moving geometry onto graphics queue family
    // FIXME any advantage for separate barriers with different offsets here for
    // index vs vertices?
    geometry_barrier.srcAccessMask = 0;
    geometry_barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                     VK_ACCESS_INDEX_READ_BIT;
    dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                              0, 0, nullptr,
                              1, &geometry_barrier,
                              0, nullptr);

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

    for (size_t texture_idx = 0; texture_idx < gpu_textures.size();
            texture_idx++) {
        const LocalTexture &gpu_texture = gpu_textures[texture_idx];
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

            dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      0, 0, nullptr, 0, nullptr,
                                      1, &barrier);

            dev.dt.cmdBlitImage(gfxCopyCommand,
                                gpu_texture.image,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                gpu_texture.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1, &blit_spec, VK_FILTER_LINEAR);

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

            dev.dt.cmdPipelineBarrier(gfxCopyCommand,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      0, 0, nullptr, 0, nullptr,
                                      1, &barrier);
        }
    }

    for (size_t texture_idx = 0; texture_idx < gpu_textures.size();
            texture_idx++) {
        const LocalTexture &gpu_texture = gpu_textures[texture_idx];
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

    REQ_VK(dev.dt.endCommandBuffer(gfxCopyCommand));

    VkSubmitInfo gfx_submit{};
    gfx_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    gfx_submit.waitSemaphoreCount = 1;
    gfx_submit.pWaitSemaphores = &copySemaphore;
    // FIXME is this right?
    VkPipelineStageFlags sema_wait_mask = 
        VK_PIPELINE_STAGE_TRANSFER_BIT |
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    gfx_submit.pWaitDstStageMask = &sema_wait_mask;
    gfx_submit.commandBufferCount = 1;
    gfx_submit.pCommandBuffers = &gfxCopyCommand;

    REQ_VK(dev.dt.queueSubmit(gfxQueue, 1, &gfx_submit,
                              copyFence));

    waitForFenceInfinitely(dev, copyFence);
    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &copyFence));

    return SceneState {
        move(geometry),
        vertex_bytes,
        move(assets.meshes),
        move(gpu_textures)
    };
}

VulkanState::VulkanState(const RenderConfig &config)
    : cfg(config),
      inst(),
      dev(inst.makeDevice(cfg.gpuID)),
      alloc(dev, inst),
      fbCfg(getFramebufferConfig(dev, inst, cfg)),
      pipeline(makePipeline(fbCfg, dev))
{}

}
