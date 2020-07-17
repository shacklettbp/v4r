#include <v4r/display.hpp>

#include "cuda_state.hpp"
#include "v4r_utils.hpp"
#include "vulkan_state.hpp"

#include <cstring>

using namespace std;

namespace v4r {

struct PresentationState {
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    DynArray<VkImage> images;
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
                                        GLFWwindow *window)
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

    return PresentationState {
        surface,
        swapchain,
        move(swapchain_images)
    };
}

PresentCommandStream::PresentCommandStream(CommandStream &&base,
                                           GLFWwindow *window)
    : CommandStream(move(base)),
      presentation_state_(Handle<PresentationState>(
            new PresentationState(
                    makePresentationState(state_->inst, state_->dev, window))))
{}

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

BatchPresentRenderer::BatchPresentRenderer(const RenderConfig &cfg)
    : BatchRenderer(make_handle<VulkanState>(cfg, makeCoreHandles(
            cfg, getUUIDFromCudaID(cfg.gpuID))))
{}

PresentCommandStream BatchPresentRenderer::makeCommandStream(
        GLFWwindow *window)
{
    CommandStream base = BatchRenderer::makeCommandStream();
    return PresentCommandStream(move(base), window);
}

glm::u32vec2 BatchPresentRenderer::getFrameDimensions() const
{
    return glm::u32vec2(state_->fbCfg.frameWidth,
                        state_->fbCfg.frameHeight);
}

}
