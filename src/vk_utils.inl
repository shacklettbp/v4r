#ifndef VK_UTILS_INL_INCLUDED
#define VK_UTILS_INL_INCLUDED

#include "vk_utils.hpp"

namespace v4r {

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

QueueState & QueueManager::allocateGraphicsQueue()

{
    return allocateQueue(dev.gfxQF, gfx_queues_,
                         cur_gfx_idx_, dev.numGraphicsQueues);
}

QueueState & QueueManager::allocateTransferQueue()
{ 
    return allocateQueue(dev.transferQF, transfer_queues_,
                         cur_transfer_idx_, dev.numTransferQueues);
}

VkCommandPool makeCmdPool(const DeviceState &dev, uint32_t qf_idx)
{
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = qf_idx;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool pool;
    REQ_VK(dev.dt.createCommandPool(dev.hdl, &pool_info, nullptr, &pool));
    return pool;
}

VkCommandBuffer makeCmdBuffer(const DeviceState &dev,
                              VkCommandPool pool,
                              VkCommandBufferLevel level)
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

VkQueue makeQueue(const DeviceState &dev,
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

VkSemaphore makeBinarySemaphore(const DeviceState &dev)
{
    VkSemaphoreCreateInfo sema_info;
    sema_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sema_info.pNext = nullptr;
    sema_info.flags = 0;

    VkSemaphore sema;
    REQ_VK(dev.dt.createSemaphore(dev.hdl, &sema_info, nullptr, &sema));

    return sema;
}

VkSemaphore makeBinaryExternalSemaphore(const DeviceState &dev)
{
    VkExportSemaphoreCreateInfo export_info;
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    export_info.pNext = nullptr;
    export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
 
    VkSemaphoreCreateInfo sema_info;
    sema_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sema_info.pNext = &export_info;
    sema_info.flags = 0;

    VkSemaphore sema;
    REQ_VK(dev.dt.createSemaphore(dev.hdl, &sema_info, nullptr, &sema));

    return sema;
}

VkFence makeFence(const DeviceState &dev, bool pre_signal)
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

void waitForFenceInfinitely(const DeviceState &dev, VkFence fence)
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

void resetFence(const DeviceState &dev, VkFence fence)
{
    dev.dt.resetFences(dev.hdl, 1, &fence);
}

}

#endif
