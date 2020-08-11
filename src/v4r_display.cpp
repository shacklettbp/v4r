#include <v4r/display.hpp>

#include "cuda_state.hpp"
#include "vulkan_state.hpp"

#include <cstring>

using namespace std;

namespace v4r {

struct PresentationSync {
    VkSemaphore swapchainReady;
    VkSemaphore renderReady;
};

struct PresentationState {
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkExtent2D swapchainSize;
    DynArray<VkImage> images;
    DynArray<PresentationSync> syncs;
};

template struct HandleDeleter<PresentationState>;

vector<const char *> getGLFWPresentationExtensions()
{
    uint32_t count;
    const char **names = glfwGetRequiredInstanceExtensions(&count);

    vector<const char *> exts(count);
    memcpy(exts.data(), names, count * sizeof(const char *));

    return exts;
}

static VkBool32 presentationSupportWrapper(VkInstance inst,
                                           VkPhysicalDevice phy,
                                           uint32_t idx)
{
    auto glfw_ret = glfwGetPhysicalDevicePresentationSupport(inst, phy, idx);
    return glfw_ret == GLFW_TRUE ? VK_TRUE : VK_FALSE;
}

VkSurfaceKHR getWindowSurface(const InstanceState &inst, GLFWwindow *window)
{
    VkSurfaceKHR surface;
    REQ_VK(glfwCreateWindowSurface(inst.hdl, window, nullptr, &surface));

    return surface;
}

VkSurfaceFormatKHR selectSwapchainFormat(const InstanceState &inst,
                                         VkPhysicalDevice phy,
                                         VkSurfaceKHR surface)
{
    uint32_t num_formats;
    REQ_VK(inst.dt.getPhysicalDeviceSurfaceFormatsKHR(
            phy, surface, &num_formats, nullptr));

    DynArray<VkSurfaceFormatKHR> formats(num_formats);
    REQ_VK(inst.dt.getPhysicalDeviceSurfaceFormatsKHR(
            phy, surface, &num_formats, formats.data()));

    if (num_formats == 0) {
        cerr  << "Zero swapchain formats" << endl;
        fatalExit();
    }

    // FIXME
    for (VkSurfaceFormatKHR format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    return formats[0];
}

VkPresentModeKHR selectSwapchainMode(const InstanceState &inst,
                                     VkPhysicalDevice phy,
                                     VkSurfaceKHR surface)
{
    uint32_t num_modes;
    REQ_VK(inst.dt.getPhysicalDeviceSurfacePresentModesKHR(
            phy, surface, &num_modes, nullptr));

    DynArray<VkPresentModeKHR> modes(num_modes);
    REQ_VK(inst.dt.getPhysicalDeviceSurfacePresentModesKHR(
            phy, surface, &num_modes, modes.data()));

    for (VkPresentModeKHR mode : modes) {
        if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            return mode;
        }
    }
    
    cerr << "Could not find immediate swapchain" << endl;
    fatalExit();
}

PresentationState makePresentationState(const InstanceState &inst,
                                        const DeviceState &dev,
                                        GLFWwindow *window,
                                        uint32_t num_frames_inflight)
{
    VkSurfaceKHR surface = getWindowSurface(inst, window);

    // Need to include this call despite the platform specific check
    // earlier (pre surface creation), or validation layers complain
    VkBool32 surface_supported;
    REQ_VK(inst.dt.getPhysicalDeviceSurfaceSupportKHR(
            dev.phy, dev.gfxQF, surface, &surface_supported));

    if (surface_supported == VK_FALSE) {
        cerr << "GLFW surface doesn't support presentation" << endl;
        fatalExit();
    }

    VkSurfaceFormatKHR format = selectSwapchainFormat(inst, dev.phy, surface);
    VkPresentModeKHR mode = selectSwapchainMode(inst, dev.phy, surface);

    VkSurfaceCapabilitiesKHR caps;
    REQ_VK(inst.dt.getPhysicalDeviceSurfaceCapabilitiesKHR(
            dev.phy, surface, &caps));

    VkExtent2D swapchain_size = caps.currentExtent;
    if (swapchain_size.width == UINT32_MAX &&
        swapchain_size.height == UINT32_MAX) {
        glfwGetWindowSize(window, (int *)&swapchain_size.width,
                          (int *)&swapchain_size.height);

        swapchain_size.width = max(caps.minImageExtent.width,
                                   min(caps.maxImageExtent.width,
                                       swapchain_size.width));

        swapchain_size.height = max(caps.minImageExtent.height,
                                    min(caps.maxImageExtent.height,
                                        swapchain_size.height));
    }

    uint32_t num_requested_images = caps.minImageCount + 1;
    if (caps.maxImageCount != 0 && num_requested_images > caps.maxImageCount) {
        num_requested_images = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchain_info;
    swapchain_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_info.pNext = nullptr;
    swapchain_info.flags = 0;
    swapchain_info.surface = surface;
    swapchain_info.minImageCount = num_requested_images;
    swapchain_info.imageFormat = format.format;
    swapchain_info.imageColorSpace = format.colorSpace;
    swapchain_info.imageExtent = swapchain_size;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_info.queueFamilyIndexCount = 0;
    swapchain_info.pQueueFamilyIndices = nullptr;
    swapchain_info.preTransform = caps.currentTransform;
    swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_info.presentMode = mode;
    swapchain_info.clipped = VK_TRUE;
    swapchain_info.oldSwapchain = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain;
    REQ_VK(dev.dt.createSwapchainKHR(dev.hdl, &swapchain_info, nullptr,
                                     &swapchain));

    uint32_t num_images;
    REQ_VK(dev.dt.getSwapchainImagesKHR(dev.hdl, swapchain, &num_images,
                                        nullptr));

    DynArray<VkImage> swapchain_images(num_images);
    REQ_VK(dev.dt.getSwapchainImagesKHR(dev.hdl, swapchain, &num_images,
                                        swapchain_images.data()));

    DynArray<PresentationSync> ready_semaphores(num_frames_inflight);

    for (uint32_t sema_idx = 0; sema_idx < num_frames_inflight; sema_idx++) {
        ready_semaphores[sema_idx] = {
            makeBinarySemaphore(dev),
            makeBinarySemaphore(dev)
        };
    }

    return PresentationState {
        surface,
        swapchain,
        swapchain_size,
        move(swapchain_images),
        move(ready_semaphores)
    };
}

PresentCommandStream::PresentCommandStream(CommandStream &&base,
                                           GLFWwindow *window,
                                           bool benchmark_mode)
    : CommandStream(move(base)),
      presentation_state_(Handle<PresentationState>(
            new PresentationState(
                    makePresentationState(state_->inst, state_->dev, window,
                                          state_->getNumFrames())))),
      benchmark_mode_(benchmark_mode)
{
    if (benchmark_mode) {
        const DeviceState &dev = state_->dev;
        VkFence fence = state_->getFence(0);

        VkCommandBuffer transition_cmd = makeCmdBuffer(dev,
                                                       state_->gfxPool);

        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        REQ_VK(dev.dt.beginCommandBuffer(transition_cmd, &begin_info));

        vector<VkImageMemoryBarrier> barriers;
        for (VkImage image : presentation_state_->images) {
            barriers.push_back({
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                0,
                0,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                }
            });
        }

        dev.dt.cmdPipelineBarrier(transition_cmd, 
                                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                  0,
                                  0, nullptr, 0, nullptr,
                                  static_cast<uint32_t>(barriers.size()),
                                  barriers.data());

        REQ_VK(dev.dt.endCommandBuffer(transition_cmd));

        VkSubmitInfo transition_submit {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0, nullptr, nullptr,
            1u, &transition_cmd,
            0, nullptr
        };

        state_->gfxQueue.submit(dev, 1, &transition_submit, fence);
        waitForFenceInfinitely(dev, fence);
        resetFence(dev, fence);

        dev.dt.freeCommandBuffers(dev.hdl, state_->gfxPool,
                                  1u, &transition_cmd);
    }
}

uint32_t PresentCommandStream::render(const vector<Environment> &elems)
{
    uint32_t frame_idx = state_->getCurrentFrame();

    auto present_sync = presentation_state_->syncs[frame_idx];
    VkSemaphore swapchain_ready = present_sync.swapchainReady;
    VkSemaphore render_ready = present_sync.renderReady;

    uint32_t swapchain_idx;
    REQ_VK(state_->dev.dt.acquireNextImageKHR(state_->dev.hdl,
                                              presentation_state_->swapchain,
                                              0, swapchain_ready,
                                              VK_NULL_HANDLE,
                                              &swapchain_idx));

    VkImage swapchain_img = presentation_state_->images[swapchain_idx];

    state_->render(elems, [&, dev=&state_->dev](
                              uint32_t frame_id,
                              uint32_t num_commands,
                              const VkCommandBuffer *commands,
                              VkFence fence) {
        (void)frame_id;

        array render_signals { render_ready };

        if (benchmark_mode_) {
            VkSubmitInfo gfx_submit {
                VK_STRUCTURE_TYPE_SUBMIT_INFO,
                nullptr,
                0, nullptr, nullptr,
                num_commands, commands,
                static_cast<uint32_t>(render_signals.size()), 
                render_signals.data()
            };

            state_->gfxQueue.submit(*dev, 1, &gfx_submit, fence);
            return;
        }

        VkCommandBuffer blit_cmd = makeCmdBuffer(*dev, state_->gfxPool);
        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        REQ_VK(dev->dt.beginCommandBuffer(blit_cmd, &begin_info));

        VkImageMemoryBarrier barrier {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            swapchain_img,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            }
        };

        dev->dt.cmdPipelineBarrier(
                blit_cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr, 0, nullptr,
                1, &barrier);

        glm::u32vec2 fb_offset = state_->getFBOffset(frame_idx);
        glm::u32vec2 render_extent = state_->getFrameExtent();
        VkImage render_img = state_->getColorImage(frame_idx);

        VkImageBlit blit_region {
            { VK_IMAGE_ASPECT_COLOR_BIT,
              0, 0, 1 },
            {
                { 
                    static_cast<int32_t>(
                        fb_offset.x),
                    static_cast<int32_t>(
                        fb_offset.y),
                    0,
                },
                {
                    static_cast<int32_t>(
                        fb_offset.x + render_extent.x),
                    static_cast<int32_t>(
                        fb_offset.y + render_extent.y),
                    1
                }
            },
            { VK_IMAGE_ASPECT_COLOR_BIT,
              0, 0, 1 },
            {
                {
                    0,
                    0,
                    0
                },
                {
                    static_cast<int32_t>(
                            presentation_state_->swapchainSize.width),
                    static_cast<int32_t>(
                        presentation_state_->swapchainSize.height),
                    1
                }
            }
        };

        dev->dt.cmdBlitImage(blit_cmd, render_img,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             swapchain_img,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             1,
                             &blit_region,
                             VK_FILTER_NEAREST);

        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        dev->dt.cmdPipelineBarrier(blit_cmd,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                   0,
                                   0, nullptr, 0, nullptr,
                                   1, &barrier);

        REQ_VK(dev->dt.endCommandBuffer(blit_cmd));

        VkPipelineStageFlags wait_flags =
            VK_PIPELINE_STAGE_TRANSFER_BIT;

        DynArray<VkCommandBuffer> extended_commands(num_commands + 1);
        memcpy(extended_commands.data(), commands,
               num_commands * sizeof(VkCommandBuffer));

        extended_commands[num_commands] = blit_cmd;

        VkSubmitInfo gfx_submit {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            1, &swapchain_ready, &wait_flags,
            static_cast<uint32_t>(extended_commands.size()),
            extended_commands.data(),
            static_cast<uint32_t>(render_signals.size()), 
            render_signals.data()
        };

        state_->gfxQueue.submit(*dev, 1, &gfx_submit, fence);
    });

    array present_waits { render_ready, swapchain_ready };

    VkPresentInfoKHR present_info;
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.pNext = nullptr;
    // Hack to skip waiting on swapchain_ready if not in benchmark_mode
    // The render submit will already have waited on it
    present_info.waitSemaphoreCount =
        static_cast<uint32_t>(
                benchmark_mode_ ? present_waits.size() : 1);
    present_info.pWaitSemaphores = present_waits.data();
    present_info.swapchainCount = 1u;
    present_info.pSwapchains = &presentation_state_->swapchain;
    present_info.pImageIndices = &swapchain_idx;
    present_info.pResults = nullptr;

    state_->gfxQueue.presentSubmit(state_->dev, &present_info);

    return frame_idx;
}

CoreVulkanHandles makeCoreHandles(const RenderConfig &config,
                                  const DeviceUUID &dev_id) {
    InstanceState inst_state(true, getGLFWPresentationExtensions());

    DeviceState dev_state = inst_state.makeDevice(
        dev_id, config.numStreams + config.numLoaders,
        1, config.numLoaders, presentationSupportWrapper);

    return CoreVulkanHandles {
        move(inst_state),
        move(dev_state)
    };
}

template <typename PipelineType>
BatchPresentRenderer::BatchPresentRenderer(
        const RenderConfig &cfg,
        const RenderFeatures<PipelineType> &features,
        bool benchmark_mode)
    : BatchRenderer(make_handle<VulkanState>(cfg, features, makeCoreHandles(
            cfg, getUUIDFromCudaID(cfg.gpuID)))),
      benchmark_mode_(benchmark_mode)
{
    assert(benchmark_mode || state_->fbCfg.colorOutput);
}

PresentCommandStream BatchPresentRenderer::makeCommandStream(
        GLFWwindow *window)
{
    CommandStream base = BatchRenderer::makeCommandStream();
    return PresentCommandStream(move(base), window, benchmark_mode_);
}

glm::u32vec2 BatchPresentRenderer::getFrameDimensions() const
{
    return glm::u32vec2(state_->fbCfg.frameWidth,
                        state_->fbCfg.frameHeight);
}

}

#include "v4r_display_instantiations.inl"
