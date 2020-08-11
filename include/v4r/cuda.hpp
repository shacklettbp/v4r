#ifndef V4R_CUDA_HPP_INCLUDED
#define V4R_CUDA_HPP_INCLUDED

#include <v4r.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

namespace v4r {

struct CudaStreamState;
struct CudaState;
struct SyncState;

class BatchRendererCUDA;

class CommandStreamCUDA : public CommandStream {
public:
    uint32_t render(const std::vector<Environment> &envs);

    uint8_t *getColorDevicePtr(uint32_t frame_id = 0) const;
    float *getDepthDevicePtr(uint32_t frame_id = 0) const;

    cudaExternalSemaphore_t getCudaSemaphore(uint32_t frame_id = 0) const;

    void streamWaitForFrame(cudaStream_t strm, uint32_t frame_id = 0) const;

private:
    CommandStreamCUDA(CommandStream &&base,
                      const CudaState &cuda_global,
                      bool double_buffered);

    Handle<SyncState[]> syncs_;
    Handle<CudaStreamState[]> cuda_;

friend class BatchRendererCUDA;
};

class BatchRendererCUDA : public BatchRenderer {
public:
    template <typename PipelineType>
    BatchRendererCUDA(const RenderConfig &cfg,
                      const RenderFeatures<PipelineType> &features)
        : BatchRendererCUDA(BatchRenderer(cfg, features),
                            cfg.gpuID)
    {}

    CommandStreamCUDA makeCommandStream();

private:
    BatchRendererCUDA(BatchRenderer &&base, int gpu_id);

    Handle<CudaState> cuda_;
};

}

#endif
