#include <v4r/cuda.hpp>
#include <unistd.h>

#include "vulkan_state.hpp"
#include "cuda_state.hpp"
#include "vk_utils.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

using namespace std;

namespace v4r {

struct SyncState {
    SyncState(const DeviceState &d)
        : dev(d),
          extSemaphore(makeBinaryExternalSemaphore(dev)),
          fd(exportBinarySemaphore(dev, extSemaphore))
    {}

    ~SyncState() 
    {
        close(fd);
        dev.dt.destroySemaphore(dev.hdl, extSemaphore, nullptr);
    }

    const DeviceState &dev;
    VkSemaphore extSemaphore;
    int fd;
};

template struct HandleDeleter<CudaState>;
template struct HandleDeleter<CudaStreamState[]>;
template struct HandleDeleter<SyncState[]>;

static SyncState *makeExportableSemaphores(
        const DeviceState &dev,
        bool double_buffered)
{
    if (double_buffered) {
        return new SyncState[2] {
            SyncState(dev),
            SyncState(dev)
        };
    } else {
        return new SyncState[1] {
            SyncState(dev)
        };
    }
}

static CudaStreamState * makeCudaStreamStates(
        const CommandStreamState &cmd_stream,
        const SyncState *syncs,
        const CudaState &cuda_global,
        bool double_buffered)
{
    cuda_global.setActiveDevice();

    if (double_buffered) {
        return new CudaStreamState[2] {
            CudaStreamState {
                (uint8_t *)cuda_global.getPointer(
                        cmd_stream.getColorOffset(0)),
                (float *)cuda_global.getPointer(
                        cmd_stream.getDepthOffset(0)),
                syncs[0].fd
            },
            CudaStreamState {
                (uint8_t *)cuda_global.getPointer(
                        cmd_stream.getColorOffset(1)),
                (float *)cuda_global.getPointer(
                        cmd_stream.getDepthOffset(1)),
                syncs[1].fd
            }
        };
    } else {
        return new CudaStreamState[1] {
            CudaStreamState {
                (uint8_t *)cuda_global.getPointer(
                        cmd_stream.getColorOffset(0)),
                (float *)cuda_global.getPointer(
                        cmd_stream.getDepthOffset(0)),
                syncs[0].fd
            }
        };
    }
}

static GLFWwindow * makeWindow()
{
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    return glfwCreateWindow(1, 1,
                            "V4R", NULL, NULL);
}

struct PresentationSync {
    VkSemaphore swapchainReady;
    VkSemaphore renderReady;
};

struct PresentationState {
    GLFWwindow *window;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkExtent2D swapchainSize;
    DynArray<VkImage> images;
    DynArray<PresentationSync> syncs;
};

template struct HandleDeleter<PresentationState>;

static vector<const char *> getGLFWPresentationExtensions()
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

static VkSurfaceKHR getWindowSurface(const InstanceState &inst, GLFWwindow *window)
{
    VkSurfaceKHR surface;
    REQ_VK(glfwCreateWindowSurface(inst.hdl, window, nullptr, &surface));

    return surface;
}

static VkSurfaceFormatKHR selectSwapchainFormat(const InstanceState &inst,
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

static VkPresentModeKHR selectSwapchainMode(const InstanceState &inst,
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

static PresentationState makePresentationState(const InstanceState &inst,
                                               const DeviceState &dev,
                                               uint32_t num_frames_inflight)
{
    GLFWwindow *window = makeWindow();

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
        window,
        surface,
        swapchain,
        swapchain_size,
        move(swapchain_images),
        move(ready_semaphores)
    };
}
CommandStreamCUDA::CommandStreamCUDA(CommandStream &&base,
                                     const CudaState &cuda_global,
                                     bool double_buffered)
    : CommandStream(move(base)),
      syncs_(makeExportableSemaphores(state_->dev, double_buffered)),
      cuda_(makeCudaStreamStates(*state_, syncs_.get(), cuda_global,
                                 double_buffered)),
      presentation_state_(Handle<PresentationState>(
            new PresentationState(
                    makePresentationState(state_->inst, state_->dev,
                                          state_->getNumFrames()))))
{
    const DeviceState &dev = state_->dev;
    VkFence fence = makeFence(dev);

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

    dev.dt.destroyFence(dev.hdl, fence, nullptr);
}

uint32_t CommandStreamCUDA::render(const vector<Environment> &envs)
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

    state_->render(envs, [&](
                    uint32_t frame_id,
                    uint32_t num_commands,
                    const VkCommandBuffer *commands,
                    VkFence fence) {

        array render_signals { render_ready, syncs_[frame_id].extSemaphore };

        VkSubmitInfo gfx_submit {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0, nullptr, nullptr,
            num_commands, commands,
            static_cast<uint32_t>(render_signals.size()), 
            render_signals.data()
        };

        state_->gfxQueue.submit(state_->dev, 1, &gfx_submit, fence);
    });

    array present_waits { render_ready, swapchain_ready };

    VkPresentInfoKHR present_info;
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.pNext = nullptr;
    present_info.waitSemaphoreCount =
        static_cast<uint32_t>(present_waits.size());
    present_info.pWaitSemaphores = present_waits.data();
    present_info.swapchainCount = 1u;
    present_info.pSwapchains = &presentation_state_->swapchain;
    present_info.pImageIndices = &swapchain_idx;
    present_info.pResults = nullptr;

    state_->gfxQueue.presentSubmit(state_->dev, &present_info);

    return frame_idx;
}

uint8_t * CommandStreamCUDA::getColorDevicePtr(uint32_t frame_id) const
{
    return cuda_[frame_id].getColor();
}

float * CommandStreamCUDA::getDepthDevicePtr(uint32_t frame_id) const
{
    return cuda_[frame_id].getDepth();
}

cudaExternalSemaphore_t CommandStreamCUDA::getCudaSemaphore(
        uint32_t frame_id) const
{
    return cuda_[frame_id].getSemaphore();
}

void CommandStreamCUDA::streamWaitForFrame(cudaStream_t strm,
                                           uint32_t frame_id) const
{
    return cudaGPUWait(getCudaSemaphore(frame_id), strm);
}

static CoreVulkanHandles makeCoreHandles(const RenderConfig &config,
                                  const DeviceUUID &dev_id)
{
    static atomic_bool glfw_init { false };
    if (!glfw_init.exchange(true)) {
        if (!glfwInit()) {
            cerr << "GLFW failed to initialize" << endl;
            exit(EXIT_FAILURE);
        }
    }

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
BatchRendererCUDA::BatchRendererCUDA(const RenderConfig &cfg,
                        const RenderFeatures<PipelineType> &features)
    : BatchRenderer(make_handle<VulkanState>(cfg, features, makeCoreHandles(
            cfg, getUUIDFromCudaID(cfg.gpuID)))),
      cuda_(make_handle<CudaState>(cfg.gpuID,
                                   state_->getFramebufferFD(),
                                   state_->getFramebufferBytes()))
{
}

CommandStreamCUDA BatchRendererCUDA::makeCommandStream()
{
    return CommandStreamCUDA(BatchRenderer::makeCommandStream(),
                             *cuda_, state_->isDoubleBuffered());
}

}

#include "v4r_cuda_instantiations.inl"
