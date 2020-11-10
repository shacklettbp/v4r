#include <v4r/cuda.hpp>
#include <unistd.h>

#include "profiler.hpp"
#include "vulkan_state.hpp"
#include "cuda_state.hpp"
#include "vk_utils.hpp"


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

CommandStreamCUDA::CommandStreamCUDA(CommandStream &&base,
                                     const CudaState &cuda_global,
                                     bool double_buffered)
    : CommandStream(move(base)),
      syncs_(makeExportableSemaphores(state_->dev, double_buffered)),
      cuda_(makeCudaStreamStates(*state_, syncs_.get(), cuda_global,
                                 double_buffered))
{}

uint32_t CommandStreamCUDA::render(const vector<Environment> &envs)
{
    return state_->render(envs, [&](
                    uint32_t frame_id,
                    uint32_t num_commands,
                    const VkCommandBuffer *commands,
                    VkFence fence) {
        auto p = Profiler::start(ProfileType::RenderSubmit);

        VkSubmitInfo gfx_submit {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0, nullptr, nullptr,
            num_commands, commands,
            1, &syncs_[frame_id].extSemaphore
        };

        state_->gfxQueue.submit(state_->dev, 1, &gfx_submit, fence);
    });
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

BatchRendererCUDA::BatchRendererCUDA(BatchRenderer &&base,
                                     int gpu_id)
    : BatchRenderer(move(base)),
      cuda_(make_handle<CudaState>(gpu_id,
                                   state_->getFramebufferFD(),
                                   state_->getFramebufferBytes()))
{}

CommandStreamCUDA BatchRendererCUDA::makeCommandStream()
{
    return CommandStreamCUDA(BatchRenderer::makeCommandStream(),
                             *cuda_, state_->isDoubleBuffered());
}

}
