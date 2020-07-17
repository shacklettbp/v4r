#include "vulkan_state.hpp"

#include "utils.hpp"
#include "vk_utils.hpp"
#include "vulkan_config.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <optional>
#include <tuple>
#include <vector>

#include <glm/gtx/string_cast.hpp>

using namespace std;

namespace v4r {

static PerSceneDescriptorConfig getSceneDescriptorConfig(
        const RenderFeatures &features,
        const DeviceState &dev)
{
    if (features.colorSrc != RenderFeatures::MeshColor::Texture) {
        return {
            VK_NULL_HANDLE,
            VK_NULL_HANDLE
        };
    }

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

    return {
        sampler,
        PerSceneDescriptorLayout::makeSetLayout(dev, nullptr, &sampler)
    };
}

static PerRenderDescriptorConfig getRenderDescriptorConfig(
        const RenderFeatures &features, 
        uint32_t batch_size,
        const MemoryAllocator &alloc,
        const DeviceState &dev)
{
    PerRenderDescriptorConfig cfg {};
    cfg.bytesPerTransform = sizeof(glm::mat4);
    cfg.totalTransformBytes = cfg.bytesPerTransform *
        VulkanConfig::max_instances;

    cfg.viewOffset = alloc.alignUniformBufferOffset(cfg.totalTransformBytes);
    cfg.totalViewBytes = sizeof(ViewInfo) * batch_size;

    bool need_material = false;
    bool need_lighting = false;

    switch (features.pipeline) {
        case RenderFeatures::Pipeline::Unlit: {
            if (features.colorSrc == RenderFeatures::MeshColor::Texture) {
                need_material = true;
                cfg.layout = UnlitMaterialPerRenderLayout::makeSetLayout(
                        dev, nullptr, nullptr, nullptr);
                cfg.makePool = UnlitMaterialPerRenderLayout::makePool;
            } else {
                cfg.layout = UnlitNoMaterialPerRenderLayout::makeSetLayout(
                        dev, nullptr, nullptr);
                cfg.makePool = UnlitNoMaterialPerRenderLayout::makePool;
            }
            break;
        }
        case RenderFeatures::Pipeline::Lit: {
            need_material = true;
            need_lighting = true;

            cfg.layout = LitPerRenderLayout::makeSetLayout(
                    dev, nullptr, nullptr, nullptr, nullptr);
            cfg.makePool = LitPerRenderLayout::makePool;
            break;
        }
        case RenderFeatures::Pipeline::Shadowed: {
            cerr << "Shadowed pipeline unimplemented" << endl;
            fatalExit();
            
            break;
        }
    }

    VkDeviceSize cur_offset = cfg.viewOffset + cfg.totalViewBytes;

    if (need_material) {
        cfg.materialIndicesOffset =
            alloc.alignStorageBufferOffset(cur_offset);

        cfg.totalMaterialIndexBytes = sizeof(uint32_t) *
            VulkanConfig::max_instances;

        cur_offset = cfg.materialIndicesOffset + cfg.totalMaterialIndexBytes;
    }

    if (need_lighting) {
        cfg.lightsOffset = alloc.alignUniformBufferOffset(cur_offset);
        cfg.totalLightParamBytes = sizeof(LightProperties) *
            VulkanConfig::max_lights + sizeof(uint32_t);

        cur_offset = cfg.lightsOffset + cfg.totalLightParamBytes;
    }

    cfg.totalParamBytes = cur_offset;

    return cfg;
}

static FramebufferConfig getFramebufferConfig(const RenderConfig &cfg)
{
    uint32_t num_frames_per_stream =
        cfg.features.options & RenderFeatures::Options::DoubleBuffered ?
            2 : 1;

    uint32_t batch_fb_images_wide = ceil(sqrt(cfg.batchSize));
    while (cfg.batchSize % batch_fb_images_wide != 0) {
        batch_fb_images_wide++;
    }

    uint32_t batch_fb_images_tall = (cfg.batchSize / batch_fb_images_wide);
    assert(batch_fb_images_wide * batch_fb_images_tall == cfg.batchSize);

    uint32_t frame_fb_width = cfg.imgWidth * batch_fb_images_wide;
    uint32_t frame_fb_height = cfg.imgHeight * batch_fb_images_tall;

    uint32_t total_fb_width = frame_fb_width * cfg.numStreams *
        num_frames_per_stream;
    uint32_t total_fb_height = frame_fb_height;
    
    vector<VkClearValue> clear_vals;

    uint64_t frame_color_bytes = 0;
    if (cfg.features.outputs & RenderFeatures::Outputs::Color) {
        frame_color_bytes = 
            4 * sizeof(uint8_t) * frame_fb_width * frame_fb_height;

        VkClearValue clear_val;
        clear_val.color = {{ 0.f, 0.f, 0.f, 1.f }};

        clear_vals.push_back(clear_val);
    }

    uint64_t frame_depth_bytes = 0;
    if (cfg.features.outputs & RenderFeatures::Outputs::Depth) {
        frame_depth_bytes =
            sizeof(float) * frame_fb_width * frame_fb_height;

        VkClearValue clear_val;
        clear_val.color = {{ 0.f, 0.f, 0.f, 0.f }};

        clear_vals.push_back(clear_val);
    }

    VkClearValue depth_clear_value;
    depth_clear_value.depthStencil = { 1.f, 0 };

    clear_vals.push_back(depth_clear_value);

    uint32_t frame_linear_bytes = frame_color_bytes + frame_depth_bytes;

    assert(frame_linear_bytes > 0);

    return FramebufferConfig {
        batch_fb_images_wide,
        batch_fb_images_tall,
        frame_fb_width,
        frame_fb_height,
        total_fb_width,
        total_fb_height,
        frame_color_bytes,
        frame_depth_bytes,
        frame_linear_bytes,
        frame_linear_bytes * cfg.numStreams * num_frames_per_stream,
        move(clear_vals)
    };
}

static VkRenderPass makeRenderPass(const FramebufferConfig &fb_cfg,
                                   const DeviceState &dev,
                                   const ResourceFormats &fmts)
{
    vector<VkAttachmentDescription> attachment_descs;
    vector<VkAttachmentReference> attachment_refs;

    if (fb_cfg.colorLinearBytesPerFrame > 0) {
        attachment_descs.push_back({
            0,
            fmts.colorAttachment,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        });

        attachment_refs.push_back({
            0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        });
    }

    if (fb_cfg.depthLinearBytesPerFrame > 0) {
        attachment_descs.push_back({
            0,
            fmts.linearDepthAttachment,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        });

        attachment_refs.push_back({
            static_cast<uint32_t>(attachment_refs.size()),
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        });
    }

    attachment_descs.push_back({
        0,
        fmts.depthAttachment,
        VK_SAMPLE_COUNT_1_BIT,
        VK_ATTACHMENT_LOAD_OP_CLEAR,
        VK_ATTACHMENT_STORE_OP_DONT_CARE,
        VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        VK_ATTACHMENT_STORE_OP_DONT_CARE,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    });

    attachment_refs.push_back({
        static_cast<uint32_t>(attachment_refs.size()),
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    });

    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount =
        static_cast<uint32_t>(attachment_refs.size() - 1);
    subpass_desc.pColorAttachments = &attachment_refs[0];
    subpass_desc.pDepthStencilAttachment = &attachment_refs.back();

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
    vector<LocalImage> attachments;
    vector<VkImageView> attachment_views;

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    if (fb_cfg.colorLinearBytesPerFrame > 0) {
        attachments.emplace_back(
                alloc.makeColorAttachment(fb_cfg.totalWidth,
                                          fb_cfg.totalHeight));

        VkImageView color_view;
        view_info.image = attachments.back().image;
        view_info.format = alloc.getFormats().colorAttachment;

        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info,
                                      nullptr, &color_view));

        attachment_views.push_back(color_view);
    }

    if (fb_cfg.depthLinearBytesPerFrame > 0) {
        attachments.emplace_back(
            alloc.makeLinearDepthAttachment(fb_cfg.totalWidth,
                                            fb_cfg.totalHeight));

        VkImageView linear_depth_view;
        view_info.image = attachments.back().image;
        view_info.format = alloc.getFormats().linearDepthAttachment;

        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info,
                                      nullptr, &linear_depth_view));

        attachment_views.push_back(linear_depth_view);
    }

    attachments.emplace_back(
            alloc.makeDepthAttachment(fb_cfg.totalWidth, fb_cfg.totalHeight));

    view_info.image = attachments.back().image;
    view_info.format = alloc.getFormats().depthAttachment;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkImageView depth_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info,
                                  nullptr, &depth_view));

    attachment_views.push_back(depth_view);

    VkFramebufferCreateInfo fb_info;
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;
    fb_info.flags = 0;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = static_cast<uint32_t>(attachment_views.size());
    fb_info.pAttachments = attachment_views.data();
    fb_info.width = fb_cfg.totalWidth;
    fb_info.height = fb_cfg.totalHeight;
    fb_info.layers = 1;

    VkFramebuffer fb_handle;
    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fb_info, nullptr, &fb_handle));

    auto [result_buffer, result_mem] =
        alloc.makeDedicatedBuffer(fb_cfg.totalLinearBytes);

    return FramebufferState {
        move(attachments),
        attachment_views,
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

static pair<vector<VkVertexInputAttributeDescription>, uint32_t>
getVertexFormat(const RenderConfig &cfg)
{
    vector<VkVertexInputAttributeDescription> input_desc;

    // All vertex types have position first (offset 0)
    input_desc.push_back({
        0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0
    });

    switch (cfg.features.pipeline) {
        case RenderFeatures::Pipeline::Unlit:
            switch (cfg.features.colorSrc) {
                case RenderFeatures::MeshColor::None:
                    return {
                        move(input_desc),
                        sizeof(UnlitRendererInputs::NoColorVertex)
                    };
                case RenderFeatures::MeshColor::Vertex:
                    input_desc.push_back({
                        1, 0, VK_FORMAT_R8G8B8_UNORM,
                        offsetof(UnlitRendererInputs::ColoredVertex, color)
                    });
                    return {
                        move(input_desc),
                        sizeof(UnlitRendererInputs::ColoredVertex)
                    };
                case RenderFeatures::MeshColor::Texture:
                    input_desc.push_back({
                        1, 0, VK_FORMAT_R32G32_SFLOAT,
                        offsetof(UnlitRendererInputs::TexturedVertex, uv)
                    });
                    return {
                        move(input_desc),
                        sizeof(UnlitRendererInputs::TexturedVertex)
                    };
                default:
                    return {{}, 0};
            };
        case RenderFeatures::Pipeline::Lit:
        case RenderFeatures::Pipeline::Shadowed:
            switch (cfg.features.colorSrc) {
                case RenderFeatures::MeshColor::None:
                    input_desc.push_back({
                        1, 0, VK_FORMAT_R32G32B32_SFLOAT,
                        offsetof(LitRendererInputs::NoColorVertex, normal)
                    });
                    return {
                        move(input_desc),
                        sizeof(LitRendererInputs::NoColorVertex)
                    };
                case RenderFeatures::MeshColor::Texture:
                    input_desc.push_back({
                        1, 0, VK_FORMAT_R32G32B32_SFLOAT,
                        offsetof(LitRendererInputs::TexturedVertex, normal)
                    });
                    input_desc.push_back({
                        2, 0, VK_FORMAT_R32G32_SFLOAT,
                        offsetof(LitRendererInputs::TexturedVertex, uv)
                    });
                    return {
                        move(input_desc),
                        sizeof(LitRendererInputs::TexturedVertex)
                    };
                default:
                    return {{}, 0};
            }
    }

    unreachable();
}

static pair<const string, const string> getShaderNames(const RenderConfig &cfg)
{
    string pipeline;

    switch (cfg.features.pipeline) {
        case RenderFeatures::Pipeline::Unlit:
            pipeline = "unlit";
            break;
        case RenderFeatures::Pipeline::Lit:
            pipeline = "lit";
            break;
        case RenderFeatures::Pipeline::Shadowed:
            pipeline = "shadowed";
            break;
    };

    string color;
    switch (cfg.features.colorSrc) {
        case RenderFeatures::MeshColor::None:
            color = "none";
            break;
        case RenderFeatures::MeshColor::Vertex:
            color = "vertex";
            break;
        case RenderFeatures::MeshColor::Texture: 
            color = "texture";
            break;
    };

    string outputs = "";
    if (cfg.features.outputs & RenderFeatures::Outputs::Color) {
        outputs += "color";
    }

    if (cfg.features.outputs & RenderFeatures::Outputs::Depth) {
        outputs += "depth";
    }

    string name = pipeline + "_" + color + "_" + outputs;

    return {
        name + ".vert.spv",
        name + ".frag.spv"
    };
}

static auto getPipelineConfig(const RenderConfig &cfg,
                              const VkDescriptorSetLayout &per_render_layout,
                              const VkDescriptorSetLayout &per_scene_layout)
{
    PipelineConfig pipeline_cfg;

    auto [input_attrs, vertex_size] = getVertexFormat(cfg);
    assert(vertex_size != 0);

    // Vertex input assembly
    pipeline_cfg.inputBindings = {{
        0, vertex_size, VK_VERTEX_INPUT_RATE_VERTEX
    }};

    pipeline_cfg.inputAttrs = move(input_attrs);

    auto [vert_name, frag_name] = getShaderNames(cfg);

    pipeline_cfg.shaders = decltype(pipeline_cfg.shaders) {
        {vert_name, VK_SHADER_STAGE_VERTEX_BIT},
        {frag_name, VK_SHADER_STAGE_FRAGMENT_BIT}
    };

    pipeline_cfg.descLayouts.push_back(per_render_layout);

    if (per_scene_layout != VK_NULL_HANDLE) {
        pipeline_cfg.descLayouts.push_back(per_scene_layout);
    }

    return pipeline_cfg;
}

static PipelineState makePipeline(const DeviceState &dev,
                                  const FramebufferConfig &fb_cfg,
                                  VkRenderPass render_pass,
                                  const PipelineConfig &cfg)
                                  
{
    // Pipeline cache (FIXME)
    VkPipelineCacheCreateInfo pcache_info {};
    pcache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkPipelineCache pipeline_cache;
    REQ_VK(dev.dt.createPipelineCache(dev.hdl, &pcache_info,
                                      nullptr, &pipeline_cache));

    vector<VkShaderModule> shader_modules;
    shader_modules.reserve(cfg.shaders.size());

    vector<VkPipelineShaderStageCreateInfo> shader_stages;
    shader_stages.reserve(shader_modules.size());

    for (size_t shader_idx = 0; shader_idx < cfg.shaders.size();
         shader_idx++) {
        auto [shader_name, shader_stage_flag] = cfg.shaders[shader_idx];

        shader_modules.push_back(loadShader(dev, shader_name));

        shader_stages.push_back({
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            shader_stage_flag,
            shader_modules[shader_idx],
            "main",
            nullptr
        });
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
        { fb_cfg.totalWidth, fb_cfg.totalHeight }
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

    vector<VkPipelineColorBlendAttachmentState> blend_attachments;
    if (fb_cfg.colorLinearBytesPerFrame > 0) {
        blend_attachments.push_back(blend_attach);
    }

    if (fb_cfg.depthLinearBytesPerFrame > 0) {
        blend_attachments.push_back(blend_attach);
    }

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

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_VERTEX_BIT |
            VK_SHADER_STAGE_FRAGMENT_BIT, // FIXME this isn't necessary for all pipelines
        0,
        sizeof(RenderPushConstant)
    };

    // Layout configuration
    VkPipelineLayoutCreateInfo pipeline_layout_info;
    pipeline_layout_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.pNext = nullptr;
    pipeline_layout_info.flags = 0;
    pipeline_layout_info.setLayoutCount =
        static_cast<uint32_t>(cfg.descLayouts.size());
    pipeline_layout_info.pSetLayouts = cfg.descLayouts.data();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_const;

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

static glm::u32vec2 computeFBPosition(uint32_t batch_idx,
                                      const FramebufferConfig &cfg,
                                      const glm::u32vec2 &render_size)
{
    return glm::u32vec2((batch_idx % cfg.numImagesWidePerBatch) *
                         render_size.x,
                        (batch_idx / cfg.numImagesWidePerBatch) *
                         render_size.y);
}

static PerFrameState makeFrameState(const DeviceState &dev,
                                    const FramebufferConfig &fb_cfg,
                                    VkCommandPool gfxPool,
                                    bool cpu_sync,
                                    const glm::u32vec2 &render_size,
                                    uint32_t batch_size,
                                    uint32_t frame_idx,
                                    uint32_t num_frames_per_stream,
                                    uint32_t stream_idx)
{
    VkCommandBuffer render_command = makeCmdBuffer(dev, gfxPool);
    VkCommandBuffer copy_command = makeCmdBuffer(dev, gfxPool);

    uint32_t global_frame_idx = stream_idx * num_frames_per_stream + frame_idx;

    glm::u32vec2 base_fb_offset(
            global_frame_idx * fb_cfg.numImagesWidePerBatch * render_size.x,
            0);

    DynArray<glm::u32vec2> batch_fb_offsets(batch_size);
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        batch_fb_offsets[batch_idx] =
            computeFBPosition(batch_idx, fb_cfg, render_size) + base_fb_offset;
    }


    VkDeviceSize color_buffer_offset =
        global_frame_idx * fb_cfg.linearBytesPerFrame;

    VkDeviceSize depth_buffer_offset =
        color_buffer_offset + fb_cfg.colorLinearBytesPerFrame;

    return PerFrameState {
        makeBinaryExternalSemaphore(dev),
        cpu_sync ? makeFence(dev) : VK_NULL_HANDLE,
        { render_command, copy_command },
        base_fb_offset,
        move(batch_fb_offsets),
        color_buffer_offset,
        depth_buffer_offset
    };
}

static void recordFBToLinearCopy(const PerFrameState &state,
                                 RenderFeatures::Outputs outputs,
                                 const DeviceState &dev,
                                 const FramebufferState &fb,
                                 const glm::u32vec2 &render_size)
{
    // FIXME move this to FramebufferState
    vector<VkImageMemoryBarrier> fb_barriers;

    if (outputs & RenderFeatures::Outputs::Color) {
        fb_barriers.emplace_back(VkImageMemoryBarrier {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            fb.attachments[0].image,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            }
        });
    }

    if (outputs & RenderFeatures::Outputs::Depth) {
        fb_barriers.emplace_back(VkImageMemoryBarrier {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            fb.attachments[fb.attachments.size() - 2].image,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            }
        });
    }

    VkCommandBuffer copy_cmd = state.commands[1];

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(copy_cmd, &begin_info));
    dev.dt.cmdPipelineBarrier(copy_cmd,
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_DEPENDENCY_BY_REGION_BIT,
                              0, nullptr, 0, nullptr,
                              static_cast<uint32_t>(fb_barriers.size()),
                              fb_barriers.data());

    uint32_t batch_size = state.batchFBOffsets.size();

    DynArray<VkBufferImageCopy> copy_regions(batch_size);

    auto make_copy_cmd = [&](VkDeviceSize base_offset, uint32_t texel_bytes,
                             VkImage src_image) {

        uint32_t cur_offset = base_offset;

        for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            glm::u32vec2 cur_fb_pos = state.batchFBOffsets[batch_idx];

            VkBufferImageCopy &region = copy_regions[batch_idx];
            region.bufferOffset = cur_offset;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource = {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 0, 1
            };
            region.imageOffset = {
                static_cast<int32_t>(cur_fb_pos.x),
                static_cast<int32_t>(cur_fb_pos.y),
                0
            };
            region.imageExtent = {
                render_size.x,
                render_size.y,
                1
            };

            cur_offset += render_size.x * render_size.y * texel_bytes;
        }

        dev.dt.cmdCopyImageToBuffer(copy_cmd,
                                    src_image,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    fb.resultBuffer.buffer,
                                    batch_size,
                                    copy_regions.data());
    };

    if (outputs & RenderFeatures::Outputs::Color) {
        make_copy_cmd(state.colorBufferOffset, sizeof(uint8_t) * 4,
                      fb.attachments[0].image);
    }

    if (outputs & RenderFeatures::Outputs::Depth) {
        make_copy_cmd(state.depthBufferOffset, sizeof(float),
                      fb.attachments[fb.attachments.size() - 2].image);
    }

    REQ_VK(dev.dt.endCommandBuffer(copy_cmd));
}

CommandStreamState::CommandStreamState(
        const RenderFeatures &features,
        const InstanceState &i,
        const DeviceState &d,
        const PerRenderDescriptorConfig &per_render_cfg,
        VkDescriptorSet per_render_descriptor,
        VkRenderPass render_pass,
        const PipelineState &pl,
        const FramebufferConfig &fb_cfg,
        const FramebufferState &framebuffer,
        MemoryAllocator &alc,
        QueueManager &queue_manager,
        uint32_t batch_size,
        uint32_t render_width,
        uint32_t render_height,
        uint32_t stream_idx,
        uint32_t num_frames_inflight)
    : inst(i),
      dev(d),
      pipeline(pl),
      gfxPool(makeCmdPool(dev, dev.gfxQF)),
      gfxQueue(queue_manager.allocateGraphicsQueue()),
      alloc(alc),
      fb_cfg_(fb_cfg),
      fb_(framebuffer),
      render_pass_(render_pass),
      per_render_descriptor_(per_render_descriptor),
      per_render_buffer_(
              alloc.makeShaderBuffer(per_render_cfg.totalParamBytes)),
      transform_ptr_(reinterpret_cast<glm::mat4 *>(
          per_render_buffer_.ptr)),
      bytes_per_txfm_(per_render_cfg.bytesPerTransform),
      view_ptr_(reinterpret_cast<ViewInfo *>(
          reinterpret_cast<uint8_t *>(per_render_buffer_.ptr) +
              per_render_cfg.viewOffset)),
      material_ptr_(per_render_cfg.totalMaterialIndexBytes == 0 ? nullptr :
          reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(
              per_render_buffer_.ptr) + per_render_cfg.materialIndicesOffset)),
      light_ptr_(per_render_cfg.totalLightParamBytes == 0 ? nullptr :
          reinterpret_cast<LightProperties *>(reinterpret_cast<uint8_t *>(
              per_render_buffer_.ptr) + per_render_cfg.lightsOffset)),
      num_lights_ptr_(light_ptr_ == nullptr ? nullptr :
          reinterpret_cast<uint32_t *>(
              light_ptr_ + VulkanConfig::max_lights)),
      render_size_(render_width, render_height),
      render_extent_(render_width * fb_cfg.numImagesWidePerBatch,
                     render_height * fb_cfg.numImagesTallPerBatch),
      frame_states_(),
      cur_frame_(0)
{
    vector<VkWriteDescriptorSet> render_desc_update;

    auto makeBufferInfo = [this](
            VkDeviceSize offset, VkDeviceSize num_bytes) {
        VkDescriptorBufferInfo buffer_info;
        buffer_info.buffer = per_render_buffer_.buffer;
        buffer_info.offset = offset;
        buffer_info.range = num_bytes;

        return buffer_info;
    };

    VkDescriptorBufferInfo transform_info =
        makeBufferInfo(0, per_render_cfg.totalTransformBytes);
    VkDescriptorBufferInfo view_info =
        makeBufferInfo(per_render_cfg.viewOffset,
                       per_render_cfg.totalViewBytes);

    // Define these outside if blocks so pointers remain valid if used
    VkDescriptorBufferInfo material_info;
    VkDescriptorBufferInfo light_info;

    uint32_t cur_binding = 0;
    VkWriteDescriptorSet binding_update;
    binding_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    binding_update.pNext = nullptr;
    binding_update.dstSet = per_render_descriptor_;
    binding_update.dstBinding = cur_binding++;
    binding_update.dstArrayElement = 0;
    binding_update.descriptorCount = 1;
    binding_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding_update.pImageInfo = nullptr;
    binding_update.pBufferInfo = &transform_info;
    binding_update.pTexelBufferView = nullptr;
    render_desc_update.push_back(binding_update);

    binding_update.dstBinding = cur_binding++;
    binding_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding_update.pBufferInfo = &view_info;
    render_desc_update.push_back(binding_update);

    if (per_render_cfg.totalMaterialIndexBytes > 0) {
        material_info = makeBufferInfo(
                per_render_cfg.materialIndicesOffset,
                per_render_cfg.totalMaterialIndexBytes);

        binding_update.dstBinding = cur_binding++;
        binding_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding_update.pBufferInfo = &material_info;
        render_desc_update.push_back(binding_update);
    }

    if (per_render_cfg.totalLightParamBytes > 0) {
        light_info = makeBufferInfo(
                per_render_cfg.lightsOffset,
                per_render_cfg.totalLightParamBytes);

        binding_update.dstBinding = cur_binding++;
        binding_update.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding_update.pBufferInfo = &light_info;
        render_desc_update.push_back(binding_update);
    }

    dev.dt.updateDescriptorSets(dev.hdl,
            static_cast<uint32_t>(render_desc_update.size()),
            render_desc_update.data(), 0, nullptr);

    bool cpu_sync = features.options &
        RenderFeatures::Options::CpuSynchronization;

    frame_states_.reserve(num_frames_inflight);
    for (uint32_t frame_idx = 0; frame_idx < num_frames_inflight;
         frame_idx++) {
        frame_states_.emplace_back(makeFrameState(dev, fb_cfg, gfxPool,
                                                  cpu_sync, render_size_,
                                                  batch_size,
                                                  frame_idx,
                                                  num_frames_inflight,
                                                  stream_idx));

        recordFBToLinearCopy(frame_states_.back(), features.outputs, dev, fb_,
                             render_size_);
    }
}

uint32_t CommandStreamState::render(const vector<Environment> &envs)
{
    return render(envs, [this](uint32_t num_commands,
                               const VkCommandBuffer *commands,
                               VkSemaphore semaphore,
                               VkFence fence) {

        VkSubmitInfo gfx_submit {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0, nullptr, nullptr,
            num_commands, commands,
            1, &semaphore
        };

        gfxQueue.submit(dev, 1, &gfx_submit, fence);
    });
}

int CommandStreamState::getSemaphoreFD(uint32_t frame_idx) const
{
    VkSemaphoreGetFdInfoKHR fd_info;
    fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    fd_info.pNext = nullptr;
    fd_info.semaphore = frame_states_[frame_idx].semaphore;
    fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd;
    REQ_VK(dev.dt.getSemaphoreFdKHR(dev.hdl, &fd_info, &fd));

    return fd;
}

VulkanState::VulkanState(const RenderConfig &config, const DeviceUUID &uuid)
    : VulkanState(config, [&config, &uuid]() {
        InstanceState inst_state(false, {});
        DeviceState dev_state(
                inst_state.makeDevice(uuid,
                                      config.numStreams + config.numLoaders,
                                      1,
                                      config.numLoaders, nullptr));
        return CoreVulkanHandles { move(inst_state), move(dev_state) };
    }())
{}

VulkanState::VulkanState(const RenderConfig &config,
                         CoreVulkanHandles &&handles)
    : cfg(config),
      inst(move(handles.inst)),
      dev(move(handles.dev)),
      queueMgr(dev),
      alloc(dev, inst),
      fbCfg(getFramebufferConfig(cfg)),
      streamDescCfg(getRenderDescriptorConfig(cfg.features, cfg.batchSize,
                                              alloc, dev)),
      sceneDescCfg(getSceneDescriptorConfig(cfg.features, dev)),
      renderDescriptorPool(streamDescCfg.makePool(dev, cfg.numStreams)),
      renderPass(makeRenderPass(fbCfg, dev, alloc.getFormats())),
      pipeline(makePipeline(dev, fbCfg, renderPass,
                            getPipelineConfig(config,
                                              streamDescCfg.layout,
                                              sceneDescCfg.layout))),
      fb(makeFramebuffer(dev, alloc, fbCfg, renderPass)),
      numLoaders(0),
      numStreams(0)
{}

LoaderState VulkanState::makeLoader()
{
    numLoaders++;
    assert(numLoaders <= cfg.numLoaders);

    return LoaderState(dev, cfg.features, sceneDescCfg,
                       alloc, queueMgr,
                       cfg.coordinateTransform);
}

CommandStreamState VulkanState::makeStream()
{
    uint32_t stream_idx = numStreams++;
    assert(stream_idx < cfg.numStreams);

    uint32_t num_frames_inflight =
        (cfg.features.options & RenderFeatures::Options::DoubleBuffered) ?
            2 : 1;

    return CommandStreamState(cfg.features,
                              inst,
                              dev,
                              streamDescCfg,
                              makeDescriptorSet(dev, renderDescriptorPool,
                                                streamDescCfg.layout),
                              renderPass,
                              pipeline,
                              fbCfg, fb,
                              alloc,
                              queueMgr,
                              cfg.batchSize,
                              cfg.imgWidth,
                              cfg.imgHeight,
                              stream_idx,
                              num_frames_inflight);
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
