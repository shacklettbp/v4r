#include "vulkan_state.hpp"

#include "utils.hpp"
#include "vk_utils.hpp"
#include "vulkan_config.hpp"

#include <iostream>
#include <fstream>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

using namespace std;

extern "C" {
    VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
            const VkInstanceCreateInfo *, const VkAllocationCallbacks *,
            VkInstance *);
}

namespace v4r {

static const char *instance_extensions[] = {
    //VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME
};

static constexpr uint32_t num_instance_extensions =
    sizeof(instance_extensions) / sizeof(const char *);

static VkInstance createInstance()
{
    VkApplicationInfo app_info {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "v4r";
    app_info.pEngineName = "v4r";
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo inst_info {};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo = &app_info;

    if (num_instance_extensions > 0) {
        inst_info.enabledExtensionCount = num_instance_extensions;
        inst_info.ppEnabledExtensionNames = instance_extensions;
    }

    VkInstance inst;
    REQ_VK(vkCreateInstance(&inst_info, nullptr, &inst));

    return inst;
}

InstanceState::InstanceState()
    : hdl(createInstance()),
      dt(hdl)
{}

static void fillQueueInfo(VkDeviceQueueCreateInfo &info, uint32_t idx,
                          const vector<float> &priorities)
{
    info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    info.queueFamilyIndex = idx;
    info.queueCount = priorities.size();
    info.pQueuePriorities = priorities.data();
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
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16B16A16_UNORM,
        VK_FORMAT_R16G16B16A16_SNORM
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

static uint32_t findMemoryTypeIndex(VkPhysicalDevice phy,
                                    uint32_t allowed_type_bits,
                                    VkMemoryPropertyFlags required_props,
                                    const InstanceState &inst)
{
    VkPhysicalDeviceMemoryProperties2 mem_props;
    mem_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    mem_props.pNext = nullptr;
    inst.dt.getPhysicalDeviceMemoryProperties2(phy, &mem_props);

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

static FramebufferConfig getFramebufferConfig(const DeviceState &dev,
                                              const InstanceState &inst,
                                              const RenderConfig &cfg)
{
    VkFormat depth_fmt = getDeviceDepthFormat(dev.phy, inst);
    VkFormat color_fmt = getDeviceColorFormat(dev.phy, inst);

    VkImageCreateInfo color_img_info {};
    color_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    color_img_info.imageType = VK_IMAGE_TYPE_2D;
    color_img_info.format = color_fmt;
    color_img_info.extent.width = cfg.imgWidth;
    color_img_info.extent.height = cfg.imgHeight;
    color_img_info.extent.depth = 1;
    color_img_info.mipLevels = 1;
    color_img_info.arrayLayers = 1;
    color_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    color_img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    color_img_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    color_img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    color_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // Create an image that will have the same settings as the real
    // framebuffer images later. This is necessary to cache the
    // memory requirements of the image.
    VkImage color_test;
    REQ_VK(dev.dt.createImage(dev.hdl, &color_img_info, nullptr, &color_test));

    VkMemoryRequirements color_req;
    dev.dt.getImageMemoryRequirements(dev.hdl, color_test, &color_req);

    dev.dt.destroyImage(dev.hdl, color_test, nullptr);

    VkMemoryAllocateInfo color_alloc_info;
    color_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    color_alloc_info.pNext = nullptr;
    color_alloc_info.allocationSize = color_req.size;
    color_alloc_info.memoryTypeIndex =
        findMemoryTypeIndex(dev.phy, color_req.memoryTypeBits,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, inst);

    // Set depth specific settings
    VkImageCreateInfo depth_img_info = color_img_info;
    depth_img_info.format = depth_fmt;
    depth_img_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    VkImage depth_test;
    REQ_VK(dev.dt.createImage(dev.hdl, &depth_img_info, nullptr, &depth_test));

    VkMemoryRequirements depth_req;
    dev.dt.getImageMemoryRequirements(dev.hdl, depth_test, &depth_req);

    dev.dt.destroyImage(dev.hdl, depth_test, nullptr);

    VkMemoryAllocateInfo depth_alloc_info;
    depth_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    depth_alloc_info.pNext = nullptr;
    depth_alloc_info.allocationSize = depth_req.size;
    depth_alloc_info.memoryTypeIndex =
        findMemoryTypeIndex(dev.phy, depth_req.memoryTypeBits,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, inst);

    return FramebufferConfig {
        cfg.imgWidth,
        cfg.imgHeight,
        depth_fmt,
        color_fmt,
        color_img_info,
        color_alloc_info,
        depth_img_info,
        depth_alloc_info
    };
}

DeviceState InstanceState::makeDevice(const uint32_t gpu_id) const
{
    uint32_t num_gpus;
    REQ_VK(dt.enumeratePhysicalDevices(hdl, &num_gpus, nullptr));

    DynArray<VkPhysicalDevice> phys(num_gpus);
    REQ_VK(dt.enumeratePhysicalDevices(hdl, &num_gpus, phys.data()));

    if (num_gpus <= gpu_id) {
        cerr << "Not enough GPUs found by vulkan" << endl;
        fatalExit();
    }

    VkPhysicalDevice phy = phys[gpu_id];

    VkPhysicalDeviceProperties2 props;
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = nullptr;
    dt.getPhysicalDeviceProperties2(phy, &props);

    VkPhysicalDeviceFeatures2 feats;
    feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats.pNext = nullptr;
    dt.getPhysicalDeviceFeatures2(phy, &feats);

    uint32_t num_queue_families;
    dt.getPhysicalDeviceQueueFamilyProperties2(phy, &num_queue_families,
                                                  nullptr);

    if (num_queue_families == 0) {
        cerr << "GPU doesn't have any queue families" << endl;
        fatalExit();
    }

    DynArray<VkQueueFamilyProperties2> queue_family_props(num_queue_families);
    for (auto &qf : queue_family_props) {
        qf.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        qf.pNext = nullptr;
    }

    dt.getPhysicalDeviceQueueFamilyProperties2(phy, &num_queue_families,
                                               queue_family_props.data());

    // Currently only finds dedicated transfer, compute, and gfx queues
    // FIXME implement more flexiblity in queue choices
    optional<uint32_t> compute_queue_family;
    optional<uint32_t> gfx_queue_family;
    optional<uint32_t> transfer_queue_family;
    for (uint32_t i = 0; i < num_queue_families; i++) {
        const auto &qf = queue_family_props[i];
        auto &qf_prop = qf.queueFamilyProperties;

        if (!transfer_queue_family &&
            (qf_prop.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(qf_prop.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {

            transfer_queue_family = i;
        } else if (!compute_queue_family &&
                   (qf_prop.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                   !(qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {

            compute_queue_family = i;;
        } else if (!gfx_queue_family &&
                   (qf_prop.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {

            gfx_queue_family = i;
        }

        if (transfer_queue_family && compute_queue_family &&
            gfx_queue_family) {
            break;
        }
    }

    if (!compute_queue_family || !gfx_queue_family || !transfer_queue_family) {
        cerr << "GPU does not support required separate queues" << endl;
        fatalExit();
    }

    const uint32_t num_gfx_queues =
        min(VulkanConfig::num_desired_gfx_queues,
            queue_family_props[*gfx_queue_family].
                queueFamilyProperties.queueCount);
    const uint32_t num_compute_queues =
        min(VulkanConfig::num_desired_compute_queues,
            queue_family_props[*compute_queue_family].
                queueFamilyProperties.queueCount);
    const uint32_t num_transfer_queues =
        min(VulkanConfig::num_desired_transfer_queues,
        queue_family_props[*transfer_queue_family].
            queueFamilyProperties.queueCount);

    array<VkDeviceQueueCreateInfo, 3> queue_infos {};
    vector<float> gfx_pris(num_gfx_queues, VulkanConfig::gfx_priority);
    vector<float> compute_pris(num_compute_queues,
                               VulkanConfig::compute_priority);
    vector<float> transfer_pris(num_transfer_queues,
                                VulkanConfig::transfer_priority);
    fillQueueInfo(queue_infos[0], *gfx_queue_family, gfx_pris);
    fillQueueInfo(queue_infos[1], *compute_queue_family, compute_pris);
    fillQueueInfo(queue_infos[2], *transfer_queue_family, transfer_pris);

    VkDeviceCreateInfo dev_create_info {};
    dev_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_create_info.queueCreateInfoCount = 3;
    dev_create_info.pQueueCreateInfos = queue_infos.data();

    // Currently ask for no features
    dev_create_info.pEnabledFeatures = nullptr;
    VkPhysicalDeviceFeatures2 requested_features {};
    requested_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    requested_features.pNext = nullptr;
    dev_create_info.pNext = &requested_features;

    VkDevice dev;
    REQ_VK(dt.createDevice(phy, &dev_create_info, nullptr, &dev));

    return DeviceState {
        *gfx_queue_family,
        *compute_queue_family,
        *transfer_queue_family,
        phy,
        dev,
        DeviceDispatch(dev)
    };
}

static VkCommandPool createCmdPool(const DeviceState &dev, uint32_t qf_idx)
{
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = qf_idx;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool pool;
    REQ_VK(dev.dt.createCommandPool(dev.hdl, &pool_info, nullptr, &pool));
    return pool;
}

static VkQueue createQueue(const DeviceState &dev, uint32_t qf_idx,
                           uint32_t queue_idx)
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

    VkDeviceMemory color_mem;
    // FIXME probably want to allocate this somewhere else and use part here
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &fb_cfg.colorMemorySettings,
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

    VkDeviceMemory depth_mem;
    REQ_VK(dev.dt.allocateMemory(dev.hdl, &fb_cfg.depthMemorySettings,
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

CommandStreamState::CommandStreamState(const DeviceState &d,
                                       const FramebufferConfig &fb_cfg,
                                       const PipelineState &pl)
    : dev(d),
      pipeline(pl),
      gfxPool(createCmdPool(dev, dev.gfxQF)),
      gfxQueue(createQueue(dev, dev.gfxQF, 0)),
      fb(makeFramebuffer(fb_cfg, pipeline, dev))
{}

VulkanState::VulkanState(const RenderConfig &config)
    : cfg(config),
      inst(),
      dev(inst.makeDevice(cfg.gpuID)),
      fbCfg(getFramebufferConfig(dev, inst, cfg)),
      pipeline(makePipeline(fbCfg, dev))
{}

}
