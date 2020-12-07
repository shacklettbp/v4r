#include "vulkan_state.hpp"
#include "render_definitions.inl"
#include "profiler.hpp"
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

template <typename PipelineType>
FramebufferConfig PipelineImpl<PipelineType>::getFramebufferConfig(
    uint32_t batch_size, uint32_t img_width, uint32_t img_height,
    uint32_t num_streams, const RenderOptions &opts)
{
    using Props = PipelineProps<PipelineType>;
    constexpr bool need_color_output = Props::needColorOutput;
    constexpr bool need_depth_output = Props::needDepthOutput;
    const bool is_double_buffered =
        opts & RenderOptions::DoubleBuffered;

    uint32_t num_frames_per_stream =
        is_double_buffered ?
            2 : 1;

    uint32_t minibatch_size = max(batch_size / VulkanConfig::minibatch_divisor,
                                  batch_size);
    assert(batch_size % minibatch_size == 0);

    uint32_t batch_fb_images_wide = ceil(sqrt(batch_size));
    while (batch_size % batch_fb_images_wide != 0) {
        batch_fb_images_wide++;
    }

    uint32_t minibatch_fb_images_wide;
    uint32_t minibatch_fb_images_tall;
    if (batch_fb_images_wide >= minibatch_size) {
        assert(batch_fb_images_wide % minibatch_size == 0);
        minibatch_fb_images_wide = minibatch_size;
        minibatch_fb_images_tall = 1;
    } else {
        minibatch_fb_images_wide = batch_fb_images_wide;
        minibatch_fb_images_tall = minibatch_size / batch_fb_images_wide;
    }

    assert(minibatch_fb_images_wide * minibatch_fb_images_tall ==
           minibatch_size);

    uint32_t batch_fb_images_tall = (batch_size / batch_fb_images_wide);
    assert(batch_fb_images_wide * batch_fb_images_tall == batch_size);

    uint32_t frame_fb_width = img_width * batch_fb_images_wide;
    uint32_t frame_fb_height = img_height * batch_fb_images_tall;

    uint32_t total_fb_width = frame_fb_width * num_streams *
        num_frames_per_stream;
    uint32_t total_fb_height = frame_fb_height;
    
    vector<VkClearValue> clear_vals;

    uint64_t frame_color_bytes = 0;
    if constexpr (need_color_output) {
        frame_color_bytes = 
            4 * sizeof(uint8_t) * frame_fb_width * frame_fb_height;

        VkClearValue clear_val;
        clear_val.color = {{ 0.f, 0.f, 0.f, 1.f }};

        clear_vals.push_back(clear_val);
    }

    uint64_t frame_depth_bytes = 0;
    if constexpr (need_depth_output) {
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
        img_width,
        img_height,
        minibatch_size,
        minibatch_fb_images_wide,
        minibatch_fb_images_tall,
        batch_fb_images_wide,
        batch_fb_images_tall,
        frame_fb_width,
        frame_fb_height,
        total_fb_width,
        total_fb_height,
        frame_color_bytes,
        frame_depth_bytes,
        frame_linear_bytes,
        frame_linear_bytes * num_streams * num_frames_per_stream,
        need_color_output,
        need_depth_output,
        move(clear_vals)
    };
}

static VkRenderPass makeRenderPass(const DeviceState &dev,
                                   const ResourceFormats &fmts,
                                   bool color_output,
                                   bool depth_output)
{
    vector<VkAttachmentDescription> attachment_descs;
    vector<VkAttachmentReference> attachment_refs;

    if (color_output) {
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

    if (depth_output) {
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

static VkSampler makeImmutableSampler(const DeviceState &dev)
{
    VkSampler sampler;

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

    REQ_VK(dev.dt.createSampler(dev.hdl, &sampler_info,
                                nullptr, &sampler));

    return sampler;
}

static ParamBufferConfig computeParamBufferConfig(
        bool need_materials, bool need_lighting,
        uint32_t batch_size, const MemoryAllocator &alloc)
{
    ParamBufferConfig cfg {};

    cfg.totalTransformBytes = sizeof(glm::mat4x3) * 
        VulkanConfig::max_instances;

    VkDeviceSize cur_offset = cfg.totalTransformBytes;

    if (need_materials) {
        cfg.materialIndicesOffset = cur_offset;

        cfg.totalMaterialIndexBytes = sizeof(uint32_t) *
            VulkanConfig::max_instances;

        cur_offset = cfg.materialIndicesOffset + cfg.totalMaterialIndexBytes;
    }


    cfg.viewOffset = alloc.alignUniformBufferOffset(cur_offset);
    cfg.totalViewBytes = sizeof(ViewInfo) * batch_size;

    cur_offset = cfg.viewOffset + cfg.totalViewBytes;

    if (need_lighting) {
        cfg.lightsOffset = alloc.alignUniformBufferOffset(cur_offset);
        cfg.totalLightParamBytes = sizeof(LightProperties) *
            VulkanConfig::max_lights + sizeof(uint32_t);

        cur_offset = cfg.lightsOffset + cfg.totalLightParamBytes;
    }

    cfg.cullInputOffset = alloc.alignStorageBufferOffset(cur_offset);
    cfg.totalCullInputBytes = sizeof(DrawInput) * VulkanConfig::max_instances;
    cur_offset = cfg.cullInputOffset + cfg.totalCullInputBytes;

    // Ensure that full block is aligned to maximum requirement
    cfg.totalParamBytes = alloc.alignStorageBufferOffset(
        alloc.alignUniformBufferOffset(cur_offset));

    cfg.countIndirectOffset = 0;
    cfg.totalCountIndirectBytes = sizeof(uint32_t) * batch_size;

    cfg.drawIndirectOffset = alloc.alignStorageBufferOffset(
        alloc.alignUniformBufferOffset(cfg.totalCountIndirectBytes));
    cfg.totalDrawIndirectBytes =
        sizeof(VkDrawIndexedIndirectCommand) *
        VulkanConfig::max_instances;

    cfg.totalIndirectBytes = alloc.alignStorageBufferOffset(
        alloc.alignUniformBufferOffset(cfg.drawIndirectOffset +
                                       cfg.totalDrawIndirectBytes));

    return  cfg;
}

template <typename PipelineType>
RenderState PipelineImpl<PipelineType>::makeRenderState(
        const DeviceState &dev, uint32_t batch_size,
        uint32_t num_streams, const RenderOptions &opts,
        MemoryAllocator &alloc)
{
    using Props = PipelineProps<PipelineType>;

    ParamBufferConfig param_positions = computeParamBufferConfig(
            Props::needMaterial, Props::needLighting, batch_size, alloc);

    using FrameLayout = typename Props::PerFrameLayout;
    array<VkSampler *, FrameLayout::NumBindings> frame_layout_args;
    frame_layout_args.fill(nullptr);

    VkDescriptorSetLayout frame_descriptor_layout = apply([&](auto ...args) {
        return FrameLayout::makeSetLayout(dev, args...);
    }, frame_layout_args);

    const uint32_t frames_per_stream =
        (opts & RenderOptions::DoubleBuffered) ? 2 : 1;

    VkDescriptorPool frame_descriptor_pool = FrameLayout::makePool(
            dev, num_streams * frames_per_stream);

    // Descriptor sets for mesh culling
    using MeshCullLayout = DescriptorLayout<
        BindingConfig<0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT>,
        BindingConfig<1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT>,
        BindingConfig<2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT>,
        BindingConfig<3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT>,
        BindingConfig<4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT>
    >;

    VkDescriptorSetLayout mesh_cull_descriptor_layout =
        MeshCullLayout::makeSetLayout(dev, nullptr, nullptr, nullptr,
                                      nullptr, nullptr);

    VkDescriptorPool mesh_cull_descriptor_pool = MeshCullLayout::makePool(
            dev, num_streams * frames_per_stream);

    using MeshCullSceneLayout = DescriptorLayout<
        BindingConfig<0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT>>;

    VkDescriptorSetLayout mesh_cull_scene_descriptor_layout =
        MeshCullSceneLayout::makeSetLayout(dev, nullptr);

    DescriptorManager::MakePoolType make_mesh_cull_scene_pool =
        MeshCullSceneLayout::makePool;

    // Conditional scene set
    VkSampler texture_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout scene_descriptor_layout = VK_NULL_HANDLE;
    DescriptorManager::MakePoolType make_scene_pool = nullptr;

    if constexpr (Props::needMaterial) {
        using SceneLayout = typename Props::PerSceneLayout;


        array<VkSampler *, SceneLayout::NumBindings> layout_args;
        layout_args.fill(nullptr);

        if constexpr (Props::needTextures) {
            texture_sampler = makeImmutableSampler(dev);
            layout_args[0] = &texture_sampler;
        }

        scene_descriptor_layout = apply([&](auto ...args) {
            return SceneLayout::makeSetLayout(dev, args...);
        }, layout_args);

        make_scene_pool = SceneLayout::makePool;
    }

    VkDescriptorSetLayout rt_layout = VK_NULL_HANDLE;
    DescriptorManager::MakePoolType make_rt_pool = nullptr;
    VkDescriptorSetLayout rt_image_layout = VK_NULL_HANDLE;
    VkDescriptorPool rt_image_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout rt_scene_layout = VK_NULL_HANDLE;
    DescriptorManager::MakePoolType make_rt_scene_pool = nullptr;
    if (opts & RenderOptions::RayTracePrimary) {
        using RTDescriptorLayout = DescriptorLayout<
            BindingConfig<0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                          1, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                             VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>>;

        rt_layout = RTDescriptorLayout::makeSetLayout(dev, nullptr);
        make_rt_pool = RTDescriptorLayout::makePool;

        using RTImageDescriptorLayout = DescriptorLayout<
            BindingConfig<0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                          1, VK_SHADER_STAGE_RAYGEN_BIT_KHR>,
            BindingConfig<1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                          1, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                             VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>,
            BindingConfig<2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                          1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>,
            BindingConfig<3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                          1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>>;

        rt_image_layout =
            RTImageDescriptorLayout::makeSetLayout(dev, nullptr, nullptr, nullptr, nullptr);
        rt_image_pool = RTImageDescriptorLayout::makePool(dev,
            num_streams * frames_per_stream);

        // FIXME needs to be moved into Props
#if 0
        using RTSceneLayout = DescriptorLayout<
            BindingConfig<0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                          VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>,
            BindingConfig<1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                          VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>,
            BindingConfig<2, VK_DESCRIPTOR_TYPE_SAMPLER, 1,
                          VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>,
            BindingConfig<3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                          VulkanConfig::max_materials,
                          VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                          VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT>

        >;

        rt_scene_layout = RTSceneLayout::makeSetLayout(dev, nullptr, nullptr,
            &texture_sampler, nullptr);
#endif
        using RTSceneLayout = DescriptorLayout<
            BindingConfig<0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                          VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>,
            BindingConfig<1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                          VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR>
        >;

        rt_scene_layout = RTSceneLayout::makeSetLayout(dev, nullptr, nullptr);
        make_rt_scene_pool = RTSceneLayout::makePool;
    }

    return {
        param_positions,
        mesh_cull_descriptor_layout,
        mesh_cull_descriptor_pool,
        mesh_cull_scene_descriptor_layout,
        make_mesh_cull_scene_pool,
        frame_descriptor_layout,
        frame_descriptor_pool,
        scene_descriptor_layout,
        make_scene_pool,
        rt_layout,
        make_rt_pool,
        rt_image_layout,
        rt_image_pool,
        rt_scene_layout,
        make_rt_scene_pool,
        texture_sampler,
        makeRenderPass(dev, alloc.getFormats(),
                       Props::needColorOutput,
                       Props::needDepthOutput)
    };
}

static FramebufferState makeFramebuffer(const DeviceState &dev,
                                        MemoryAllocator &alloc,
                                        const FramebufferConfig &fb_cfg,
                                        VkRenderPass render_pass,
                                        bool use_rt)
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

    if (fb_cfg.colorOutput) {
        if (use_rt) {
            attachments.emplace_back(
                alloc.makeRTStorageImage(fb_cfg.totalWidth,
                                         fb_cfg.totalHeight));
        } else {
            attachments.emplace_back(
                    alloc.makeColorAttachment(fb_cfg.totalWidth,
                                              fb_cfg.totalHeight));
        }

        VkImageView color_view;
        view_info.image = attachments.back().image;
        view_info.format = alloc.getFormats().colorAttachment;

        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info,
                                      nullptr, &color_view));

        attachment_views.push_back(color_view);
    }

    if (fb_cfg.depthOutput) {
        if (use_rt) {
            attachments.emplace_back(
                alloc.makeRTStorageImage(fb_cfg.totalWidth,
                                         fb_cfg.totalHeight));
        } else {
            attachments.emplace_back(
                alloc.makeLinearDepthAttachment(fb_cfg.totalWidth,
                                            fb_cfg.totalHeight));
        }

        VkImageView linear_depth_view;
        view_info.image = attachments.back().image;
        view_info.format = alloc.getFormats().linearDepthAttachment;

        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info,
                                      nullptr, &linear_depth_view));

        attachment_views.push_back(linear_depth_view);
    }

    VkFramebuffer fb_handle = VK_NULL_HANDLE;
    if (!use_rt) {
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

        REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fb_info, nullptr, &fb_handle));
    }

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

template <typename PipelineType>
PipelineState PipelineImpl<PipelineType>::makePipeline(
        const DeviceState &dev,
        const FramebufferConfig &fb_cfg,
        const RenderState &render_state,
        MemoryAllocator &alloc)
{
    using Props = PipelineProps<PipelineType>;
    using VertexType = typename PipelineType::Vertex;

    (void)fb_cfg;
    array<VkDescriptorSetLayout, 3> gfx_desc_layouts {
        render_state.rtImageDescriptorLayout,
        render_state.rtDescriptorLayout,
        render_state.rtSceneDescriptorLayout,
    };

    // Pipeline cache (FIXME)
    VkPipelineCacheCreateInfo pcache_info {};
    pcache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkPipelineCache pipeline_cache;
    REQ_VK(dev.dt.createPipelineCache(dev.hdl, &pcache_info,
                                      nullptr, &pipeline_cache));

    const array<pair<const char *, VkShaderStageFlagBits>, 3> shader_cfg {{
        {"primary.rgen.spv", VK_SHADER_STAGE_RAYGEN_BIT_KHR},
        {"primary.rmiss.spv", VK_SHADER_STAGE_MISS_BIT_KHR},
        {"primary_unlit_color_texture.rchit.spv",
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
    }};

    constexpr size_t num_shaders = shader_cfg.size();

    vector<VkShaderModule> shader_modules(num_shaders);
    array<VkPipelineShaderStageCreateInfo, num_shaders> shader_stages;

    for (size_t shader_idx = 0; shader_idx < shader_cfg.size();
         shader_idx++) {
        auto [shader_name, shader_stage_flag] = shader_cfg[shader_idx];

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

    array<VkRayTracingShaderGroupCreateInfoKHR, 3> shader_groups;

    shader_groups[0].sType =
        VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    shader_groups[0].pNext = nullptr;
    shader_groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    shader_groups[0].generalShader = 0;
    shader_groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    shader_groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    shader_groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_groups[0].pShaderGroupCaptureReplayHandle = nullptr;

    shader_groups[1].sType =
        VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    shader_groups[1].pNext = nullptr;
    shader_groups[1].type =
        VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    shader_groups[1].generalShader = 1;
    shader_groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    shader_groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    shader_groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_groups[1].pShaderGroupCaptureReplayHandle = nullptr;

    shader_groups[2].sType =
        VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    shader_groups[2].pNext = nullptr;
    shader_groups[2].type =
        VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    shader_groups[2].generalShader = VK_SHADER_UNUSED_KHR;
    shader_groups[2].closestHitShader = 2;
    shader_groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    shader_groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;
    shader_groups[2].pShaderGroupCaptureReplayHandle = nullptr;


    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_RAYGEN_BIT_KHR |
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        0,
        sizeof(Shader::RTRenderPushConstant)
    };

    // Layout configuration
    VkPipelineLayoutCreateInfo gfx_layout_info;
    gfx_layout_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfx_layout_info.pNext = nullptr;
    gfx_layout_info.flags = 0;
    gfx_layout_info.setLayoutCount =
        static_cast<uint32_t>(gfx_desc_layouts.size());
    gfx_layout_info.pSetLayouts = gfx_desc_layouts.data();
    gfx_layout_info.pushConstantRangeCount = 1;
    gfx_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout gfx_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &gfx_layout_info,
                                       nullptr, &gfx_layout));

    VkRayTracingPipelineCreateInfoKHR rt_pipeline_info;
    rt_pipeline_info.sType =
        VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rt_pipeline_info.pNext = nullptr;
    rt_pipeline_info.flags = 0;
    rt_pipeline_info.stageCount = shader_stages.size();
    rt_pipeline_info.pStages = shader_stages.data();
    rt_pipeline_info.groupCount = shader_groups.size();
    rt_pipeline_info.pGroups = shader_groups.data();
    rt_pipeline_info.maxPipelineRayRecursionDepth = 1;
    rt_pipeline_info.pLibraryInfo = nullptr;
    rt_pipeline_info.pLibraryInterface = nullptr;
    rt_pipeline_info.pDynamicState = nullptr;
    rt_pipeline_info.layout = gfx_layout;
    rt_pipeline_info.basePipelineHandle = 0;
    rt_pipeline_info.basePipelineIndex = 0;

    VkPipeline gfx_pipeline;
    REQ_VK(dev.dt.createRayTracingPipelinesKHR(dev.hdl, VK_NULL_HANDLE,
                                               VK_NULL_HANDLE, 1,
                                               &rt_pipeline_info,
                                               nullptr, &gfx_pipeline));

    VkDeviceSize group_size = alignOffset(dev.rtShaderGroupHandleSize,
                                          dev.rtShaderGroupBaseAlignment);

    // FIXME device local shader binding table?
    const VkDeviceSize sbt_size =
        group_size * shader_groups.size();

    HostBuffer shader_binding_table = alloc.makeSBTBuffer(sbt_size);

    DynArray<uint8_t> sbt_scratch(sbt_size);
    REQ_VK(dev.dt.getRayTracingShaderGroupHandlesKHR(dev.hdl, gfx_pipeline, 0,
                                                     shader_groups.size(),
                                                     sbt_size,
                                                     sbt_scratch.data()));

    uint8_t *sbt_ptr = (uint8_t *)shader_binding_table.ptr;
    for (uint32_t i = 0; i < shader_groups.size(); i++) {
        memcpy(sbt_ptr, sbt_scratch.data() + i * dev.rtShaderGroupHandleSize,
               dev.rtShaderGroupHandleSize);

        sbt_ptr += dev.rtShaderGroupBaseAlignment;
    }

    shader_binding_table.flush(dev);

    VkBufferDeviceAddressInfoKHR sbt_addr_info;
    sbt_addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    sbt_addr_info.pNext = nullptr;
    sbt_addr_info.buffer = shader_binding_table.buffer;
    VkDeviceAddress sbt_base_addr =
        dev.dt.getBufferDeviceAddressKHR(dev.hdl, &sbt_addr_info);

    VkStridedDeviceAddressRegionKHR raygen_entry {
        sbt_base_addr,
        group_size,
        group_size,
    };

    VkStridedDeviceAddressRegionKHR miss_entry {
        sbt_base_addr + group_size,
        group_size,
        group_size,
    };

    VkStridedDeviceAddressRegionKHR hit_entry {
        sbt_base_addr + group_size * 2,
        group_size,
        group_size,
    };

    VkStridedDeviceAddressRegionKHR callable_entry {};

    return PipelineState {
        shader_modules,
        pipeline_cache,
        gfx_layout,
        gfx_pipeline,
        move(shader_binding_table),
        raygen_entry,
        miss_entry,
        hit_entry,
        callable_entry,
    };
}

static glm::u32vec2 computeFBPosition(uint32_t batch_idx,
                                      const FramebufferConfig &cfg)
{
    return glm::u32vec2((batch_idx % cfg.numImagesWidePerBatch) *
                        cfg.imgWidth,
                        (batch_idx / cfg.numImagesWidePerBatch) *
                        cfg.imgHeight);
}

static PerFrameState makeFrameState(const DeviceState &dev,
                                    const FramebufferConfig &fb_cfg,
                                    const FramebufferState &fb,
                                    const ParamBufferConfig &param_config,
                                    const HostBuffer &param_buffer,
                                    const LocalBuffer &indirect_buffer,
                                    VkCommandPool gfx_pool,
                                    VkDescriptorPool frame_set_pool,
                                    VkDescriptorSetLayout frame_set_layout,
                                    VkDescriptorPool mesh_cull_set_pool,
                                    VkDescriptorSetLayout mesh_cull_set_layout,
                                    VkDescriptorPool rt_set_pool,
                                    VkDescriptorSetLayout rt_set_layout,
                                    bool cpu_sync,
                                    uint32_t batch_size,
                                    uint32_t frame_idx,
                                    uint32_t num_frames_per_stream,
                                    uint32_t stream_idx)
{
    VkCommandBuffer render_command = makeCmdBuffer(dev, gfx_pool);
    VkCommandBuffer copy_command = makeCmdBuffer(dev, gfx_pool);

    uint32_t global_frame_idx = stream_idx * num_frames_per_stream + frame_idx;

    glm::u32vec2 base_fb_offset(
            global_frame_idx * fb_cfg.numImagesWidePerBatch * fb_cfg.imgWidth,
            0);

    DynArray<glm::u32vec2> batch_fb_offsets(batch_size);
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        batch_fb_offsets[batch_idx] =
            computeFBPosition(batch_idx, fb_cfg) + base_fb_offset;
    }

    VkDeviceSize color_buffer_offset =
        global_frame_idx * fb_cfg.linearBytesPerFrame;

    VkDeviceSize depth_buffer_offset =
        color_buffer_offset + fb_cfg.colorLinearBytesPerFrame;

    const bool use_materials = param_config.totalMaterialIndexBytes > 0;
    const bool use_lights = param_config.totalLightParamBytes > 0;

    VkDescriptorSet frame_set = makeDescriptorSet(dev, frame_set_pool,
                                                  frame_set_layout);
    vector<VkWriteDescriptorSet> desc_set_updates;
    desc_set_updates.reserve(2 + 5); // Frame set + cull set

    const size_t num_vertex_inputs = use_materials ? 3 : 2;

    DynArray<VkBuffer> vertex_buffers(num_vertex_inputs);
    DynArray<VkDeviceSize> vertex_offsets(num_vertex_inputs);

    // First vertex buffer is the actual vertex buffer, starts at 0, handle
    // is set at render time
    vertex_buffers[0] = VK_NULL_HANDLE;
    vertex_offsets[0] = 0;

    VkDeviceSize base_offset = frame_idx * param_config.totalParamBytes;

    vertex_buffers[1] = param_buffer.buffer;
    vertex_offsets[1] = base_offset;

    uint8_t *base_ptr = reinterpret_cast<uint8_t *>(param_buffer.ptr) +
        base_offset;

    glm::mat4x3 *transform_ptr = reinterpret_cast<glm::mat4x3 *>(base_ptr);

    ViewInfo *view_ptr = reinterpret_cast<ViewInfo *>(
            base_ptr + param_config.viewOffset);

    VkDescriptorBufferInfo view_buffer_info = {
        param_buffer.buffer,
        base_offset + param_config.viewOffset,
        param_config.totalViewBytes
    };

    VkWriteDescriptorSet binding_update;
    binding_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    binding_update.pNext = nullptr;
    binding_update.dstSet = frame_set;
    binding_update.dstBinding = 0;
    binding_update.dstArrayElement = 0;
    binding_update.descriptorCount = 1;
    binding_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding_update.pImageInfo = nullptr;
    binding_update.pBufferInfo = &view_buffer_info;
    binding_update.pTexelBufferView = nullptr;
    desc_set_updates.push_back(binding_update);

    uint32_t *material_ptr = nullptr;
    LightProperties *light_ptr = nullptr;
    uint32_t *num_lights_ptr = nullptr;

    if (use_materials) {
        material_ptr = reinterpret_cast<uint32_t *>(
                base_ptr + param_config.materialIndicesOffset);

        vertex_buffers[2] = param_buffer.buffer;
        vertex_offsets[2] = base_offset + param_config.materialIndicesOffset;
    }

    VkDescriptorBufferInfo light_info;
    if (use_lights) {
        light_ptr = reinterpret_cast<LightProperties *>(
                base_ptr + param_config.lightsOffset);

        num_lights_ptr = reinterpret_cast<uint32_t *>(
                light_ptr + VulkanConfig::max_lights);

        light_info = {
            param_buffer.buffer,
            base_offset + param_config.lightsOffset,
            param_config.totalLightParamBytes
        };

        binding_update.dstBinding = 1;
        binding_update.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding_update.pBufferInfo = &light_info;
        desc_set_updates.push_back(binding_update);
    }

    // Indirect draw stuff
    
    DrawInput *draw_ptr = reinterpret_cast<DrawInput *>(
            base_ptr + param_config.cullInputOffset);

    VkDeviceSize base_indirect_offset =
        param_config.totalIndirectBytes * frame_idx;

    VkDeviceSize count_indirect_offset = 
        base_indirect_offset + param_config.countIndirectOffset;

    VkDeviceSize draw_indirect_offset = 
        base_indirect_offset + param_config.drawIndirectOffset;

    VkDescriptorSet cull_set = makeDescriptorSet(dev, mesh_cull_set_pool,
                                                 mesh_cull_set_layout);

    VkDescriptorBufferInfo transform_info {
        param_buffer.buffer,
        base_offset,
        param_config.totalTransformBytes
    };
    
    binding_update.dstSet = cull_set;
    binding_update.dstBinding = 0;
    binding_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding_update.pBufferInfo = &transform_info;
    desc_set_updates.push_back(binding_update);

    binding_update.dstBinding = 1;
    binding_update.pBufferInfo = &view_buffer_info;
    desc_set_updates.push_back(binding_update);

    VkDescriptorBufferInfo indirect_input_buffer_info {
        param_buffer.buffer,
        param_config.cullInputOffset,
        param_config.totalCullInputBytes
    };

    binding_update.dstBinding = 2;
    binding_update.pBufferInfo = &indirect_input_buffer_info;
    desc_set_updates.push_back(binding_update);

    VkDescriptorBufferInfo indirect_output_buffer_info {
        indirect_buffer.buffer,
        draw_indirect_offset,
        param_config.totalDrawIndirectBytes
    };

    binding_update.dstBinding = 3;
    binding_update.pBufferInfo = &indirect_output_buffer_info;
    desc_set_updates.push_back(binding_update);

    VkDescriptorBufferInfo indirect_count_buffer_info {
        indirect_buffer.buffer,
        count_indirect_offset,
        param_config.totalCountIndirectBytes
    };

    binding_update.dstBinding = 4;
    binding_update.pBufferInfo = &indirect_count_buffer_info;
    desc_set_updates.push_back(binding_update);

    VkDescriptorSet rt_set = makeDescriptorSet(dev, rt_set_pool,
                                               rt_set_layout);

    VkDescriptorBufferInfo material_buffer_info = {
        param_buffer.buffer,
        base_offset + param_config.materialIndicesOffset,
        param_config.totalMaterialIndexBytes,
    };

    VkDescriptorImageInfo storage_image_info = {
        VK_NULL_HANDLE,                      
        fb.attachmentViews[0],
        VK_IMAGE_LAYOUT_GENERAL,
    };

    if (rt_set != VK_NULL_HANDLE) {
        binding_update.dstSet = rt_set;
        binding_update.dstBinding = 0;
        binding_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binding_update.pBufferInfo = nullptr;
        binding_update.pImageInfo = &storage_image_info;
        desc_set_updates.push_back(binding_update);

        binding_update.dstBinding = 1;
        binding_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding_update.pBufferInfo = &view_buffer_info;
        binding_update.pImageInfo = nullptr;
        desc_set_updates.push_back(binding_update);

        binding_update.dstBinding = 2;
        binding_update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding_update.pBufferInfo = &material_buffer_info;
        binding_update.pImageInfo = nullptr;
        desc_set_updates.push_back(binding_update);

        binding_update.dstBinding = 3;
        binding_update.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding_update.pBufferInfo = &light_info;
        binding_update.pImageInfo = nullptr;
        desc_set_updates.push_back(binding_update);
    }

    dev.dt.updateDescriptorSets(dev.hdl,
            static_cast<uint32_t>(desc_set_updates.size()),
            desc_set_updates.data(), 0, nullptr);

    return PerFrameState {
        cpu_sync ? makeFence(dev) : VK_NULL_HANDLE,
        { render_command, copy_command },
        count_indirect_offset,
        sizeof(uint32_t) * batch_size,
        draw_indirect_offset,
        DynArray<uint32_t>(batch_size),
        DynArray<uint32_t>(batch_size),
        base_fb_offset,
        move(batch_fb_offsets),
        color_buffer_offset,
        depth_buffer_offset,
        cull_set,
        frame_set,
        rt_set,
        move(vertex_buffers),
        move(vertex_offsets),
        transform_ptr,
        view_ptr,
        material_ptr,
        light_ptr,
        num_lights_ptr,
        draw_ptr
    };
}

static void recordFBToLinearCopy(const DeviceState &dev,
                                 const PerFrameState &state,
                                 const FramebufferConfig &fb_cfg,
                                 const FramebufferState &fb,
                                 bool rt_primary)
{
    // FIXME move this to FramebufferState
    vector<VkImageMemoryBarrier> fb_barriers;

    if (fb_cfg.colorOutput) {
        fb_barriers.emplace_back(VkImageMemoryBarrier {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            rt_primary ? VK_ACCESS_SHADER_WRITE_BIT :
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            rt_primary ? VK_IMAGE_LAYOUT_GENERAL :
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

    if (fb_cfg.depthOutput) {
        fb_barriers.emplace_back(VkImageMemoryBarrier {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            rt_primary ? VK_ACCESS_SHADER_WRITE_BIT :
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            rt_primary ? VK_IMAGE_LAYOUT_GENERAL : 
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
                              rt_primary ?
                                   VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
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
                fb_cfg.imgWidth,
                fb_cfg.imgHeight,
                1
            };

            cur_offset += fb_cfg.imgWidth * fb_cfg.imgHeight * texel_bytes;
        }

        dev.dt.cmdCopyImageToBuffer(copy_cmd,
                                    src_image,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    fb.resultBuffer.buffer,
                                    batch_size,
                                    copy_regions.data());
    };

    if (fb_cfg.colorOutput) {
        make_copy_cmd(state.colorBufferOffset, sizeof(uint8_t) * 4,
                      fb.attachments[0].image);
    }

    if (fb_cfg.depthOutput) {
        make_copy_cmd(state.depthBufferOffset, sizeof(float),
                      fb.attachments[fb.attachments.size() - 2].image);
    }

    if (rt_primary) {
        for (auto &barrier : fb_barriers) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        }

        dev.dt.cmdPipelineBarrier(copy_cmd,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                  VK_DEPENDENCY_BY_REGION_BIT,
                                  0, nullptr, 0, nullptr,
                                  static_cast<uint32_t>(fb_barriers.size()),
                                  fb_barriers.data());
    }

    REQ_VK(dev.dt.endCommandBuffer(copy_cmd));
}

CommandStreamState::CommandStreamState(
        const InstanceState &i,
        const DeviceState &d,
        const FramebufferConfig &fb_cfg,
        const RenderState &render_state,
        const PipelineState &pl,
        const FramebufferState &framebuffer,
        MemoryAllocator &alc,
        QueueState &graphics_queue,
        uint32_t batch_size,
        uint32_t stream_idx,
        uint32_t num_frames_inflight,
        bool cpu_sync,
        bool enable_rt)
    : inst(i),
      dev(d),
      pipeline(pl),
      gfxPool(makeCmdPool(dev, dev.gfxQF)),
      gfxQueue(graphics_queue),
      alloc(alc),
      fb_cfg_(fb_cfg),
      fb_(framebuffer),
      render_pass_(render_state.renderPass),
      per_render_buffer_(alloc.makeParamBuffer(
          render_state.paramPositions.totalParamBytes *
              num_frames_inflight)),
      indirect_draw_buffer_([&]() {
          auto opt_buffer = alloc.makeIndirectBuffer( 
            render_state.paramPositions.totalIndirectBytes *
                num_frames_inflight);
          if (!opt_buffer.has_value()) {
              cerr << "Out of memory, cannot allocate indirect draw buffer"
                   << endl;
              fatalExit();
          }
          return move(opt_buffer.value());
      }()),
      rt_desc_mgr_(dev, render_state.rtDescriptorLayout,
                   render_state.makeRTDescriptorPool),
      mini_batch_size_(fb_cfg.miniBatchSize),
      num_mini_batches_(batch_size / mini_batch_size_),
      per_elem_render_size_(fb_cfg.imgWidth, fb_cfg.imgHeight),
      per_minibatch_render_size_(
          per_elem_render_size_.x * fb_cfg.numImagesWidePerMiniBatch,
          per_elem_render_size_.y * fb_cfg.numImagesTallPerMiniBatch),
      per_batch_render_size_(
          per_elem_render_size_.x * fb_cfg.numImagesWidePerBatch,
          per_elem_render_size_.y * fb_cfg.numImagesTallPerBatch),
      frame_states_(),
      cur_frame_(0),
      enable_rt_(enable_rt),
      tlas_build_cmd_(enable_rt_ ? makeCmdBuffer(dev, gfxPool) :
                      VK_NULL_HANDLE)

{
    assert(num_mini_batches_ * mini_batch_size_ == batch_size);

    frame_states_.reserve(num_frames_inflight);
    for (uint32_t frame_idx = 0; frame_idx < num_frames_inflight;
         frame_idx++) {
        frame_states_.emplace_back(makeFrameState(dev,
                fb_cfg,
                fb_,
                render_state.paramPositions,
                per_render_buffer_,
                indirect_draw_buffer_,
                gfxPool,
                render_state.frameDescriptorPool,
                render_state.frameDescriptorLayout,
                render_state.meshCullDescriptorPool,
                render_state.meshCullDescriptorLayout,
                render_state.rtImageDescriptorPool,
                render_state.rtImageDescriptorLayout,
                cpu_sync, batch_size,
                frame_idx, num_frames_inflight, stream_idx));

        recordFBToLinearCopy(dev, frame_states_.back(), fb_cfg_, fb_,
                             enable_rt);
    }

    if (enable_rt) {
        // FIXME, depth output
        VkCommandBuffer fb_init_command = frame_states_[0].commands[0];

        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        REQ_VK(dev.dt.beginCommandBuffer(fb_init_command, &begin_info));

        VkImageMemoryBarrier storage_barrier {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            fb_.attachments[0].image,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            }
        };

        dev.dt.cmdPipelineBarrier(fb_init_command,
                                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                  VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                  0,
                                  0, nullptr, 0, nullptr,
                                  1, &storage_barrier);

        REQ_VK(dev.dt.endCommandBuffer(fb_init_command));

        VkSubmitInfo gfx_submit {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0, nullptr, nullptr,
            1, &fb_init_command,
            0, nullptr
        };

        gfxQueue.submit(dev, 1, &gfx_submit, frame_states_[0].fence);
        waitForFenceInfinitely(dev, frame_states_[0].fence);
        resetFence(dev, frame_states_[0].fence);
    }
}

uint32_t CommandStreamState::render(const vector<Environment> &envs)
{
    return render(envs, [this](uint32_t frame_id,
                               uint32_t num_commands,
                               const VkCommandBuffer *commands,
                               VkFence fence) {
        auto p = Profiler::start(ProfileType::RenderSubmit);
        (void)frame_id;

        VkSubmitInfo gfx_submit {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0, nullptr, nullptr,
            num_commands, commands,
            0, nullptr
        };

        gfxQueue.submit(dev, 1, &gfx_submit, fence);
    });
}

EnvironmentState CommandStreamState::makeEnvironment(
    const std::shared_ptr<Scene> &scene, 
    const glm::mat4 &perspective)
{
    return EnvironmentState(dev, alloc, scene, perspective, enable_rt_,
        rt_desc_mgr_.makeSet(), tlas_build_cmd_, gfxQueue);
}

template <typename PipelineType>
VulkanState::VulkanState(const RenderConfig &config,
                         const RenderFeatures<PipelineType> &features,
                         const DeviceUUID &uuid)
    : VulkanState(config, features, [&config, &uuid, &features]() {
        uint32_t num_compute_queues =
            features.options & RenderOptions::RayTracePrimary ?
                config.numLoaders : 1;

        InstanceState inst_state(false, {});
        DeviceState dev_state(inst_state.makeDevice(uuid,
            features.options & RenderOptions::RayTracePrimary,
            config.numStreams + 1,
            num_compute_queues,
            config.numLoaders, nullptr));
        return CoreVulkanHandles { move(inst_state), move(dev_state) };
    }())
{}

template <typename PipelineType>
VulkanState::VulkanState(const RenderConfig &cfg,
                         const RenderFeatures<PipelineType> &features,
                         CoreVulkanHandles &&handles)
    : inst(move(handles.inst)),
      dev(move(handles.dev)),
      alloc(dev, inst, features.options & RenderOptions::RayTracePrimary),
      fbCfg(PipelineImpl<PipelineType>::getFramebufferConfig(
              cfg.batchSize, cfg.imgWidth,
              cfg.imgHeight, cfg.numStreams,
              features.options)),
      renderState(PipelineImpl<PipelineType>::makeRenderState(
              dev, cfg.batchSize, cfg.numStreams,
              features.options, alloc)),
      pipeline(PipelineImpl<PipelineType>::makePipeline(
              dev, fbCfg, renderState, alloc)),
      fb(makeFramebuffer(dev, alloc, fbCfg, renderState.renderPass,
                         features.options & RenderOptions::RayTracePrimary)),
      globalTransform(cfg.coordinateTransform),
      loader_impl_(
              LoaderImpl::create<typename PipelineType::Vertex,
                                 typename PipelineType::MaterialParams>()),
      num_loaders_(0),
      num_streams_(0),
      max_num_loaders_(cfg.numLoaders),
      max_num_streams_(cfg.numStreams),
      transfer_queues_(dev.numTransferQueues),
      graphics_queues_(dev.numGraphicsQueues),
      compute_queues_(features.options & RenderOptions::RayTracePrimary ? 
                      dev.numComputeQueues : 0),
      batch_size_(cfg.batchSize),
      double_buffered_(features.options & RenderOptions::DoubleBuffered),
      cpu_sync_(features.options & RenderOptions::CpuSynchronization),
      enable_rt_(features.options & RenderOptions::RayTracePrimary)
{
    bool transfer_shared = cfg.numLoaders > dev.numTransferQueues;

    for (uint32_t i = 0; i < transfer_queues_.size(); i++) {
        new (&transfer_queues_[i]) QueueState(makeQueue(dev, dev.transferQF, i),
                                            transfer_shared);
    }

    bool graphics_shared = cfg.numStreams + 1 > dev.numGraphicsQueues;

    for (uint32_t i = 0; i < graphics_queues_.size(); i++) {
        new (&graphics_queues_[i]) QueueState(makeQueue(dev, dev.gfxQF, i),
            (i == graphics_queues_.size() - 1) ? true : graphics_shared);
    }

    bool compute_shared = cfg.numLoaders > dev.numComputeQueues;

    for (uint32_t i = 0; i < compute_queues_.size(); i++) {
        new (&compute_queues_[i]) QueueState(makeQueue(dev, dev.computeQF, i),
                                             compute_shared);
    }
}

LoaderState VulkanState::makeLoader()
{
    num_loaders_++;
    assert(num_loaders_ <= max_num_loaders_);

    return LoaderState(dev, loader_impl_,
                       renderState.sceneDescriptorLayout,
                       renderState.makeScenePool,
                       renderState.meshCullSceneDescriptorLayout,
                       renderState.makeMeshCullScenePool,
                       renderState.rtSceneDescriptorLayout,
                       renderState.makeRTScenePool,
                       alloc,
                       transfer_queues_[num_loaders_ % transfer_queues_.size()],
                       graphics_queues_.back(),
                       globalTransform,
                       enable_rt_ ?
                           &compute_queues_[num_loaders_ %
                               compute_queues_.size()] :
                           nullptr,
                       enable_rt_);
}

CommandStreamState VulkanState::makeStream()
{
    uint32_t stream_idx = num_streams_++;
    assert(stream_idx < max_num_streams_);

    uint32_t num_frames_inflight = double_buffered_ ? 2 : 1;

    return CommandStreamState(inst,
                              dev,
                              fbCfg,
                              renderState,
                              pipeline,
                              fb,
                              alloc,
                              graphics_queues_[stream_idx % graphics_queues_.size()],
                              batch_size_,
                              stream_idx,
                              num_frames_inflight,
                              cpu_sync_,
                              enable_rt_);
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

#include "render_instantiations.inl"
