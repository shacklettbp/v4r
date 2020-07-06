#ifndef CUDA_STATE_INL_INCLUDED
#define CUDA_STATE_INL_INCLUDED

#include "cuda_state.hpp"
#include "utils.hpp"

#include <iostream>

namespace v4r {

void cudaGPUWait(cudaExternalSemaphore_t sem, cudaStream_t strm)
{
    cudaExternalSemaphoreWaitParams params {};
    cudaError_t res =
        cudaWaitExternalSemaphoresAsync(&sem, &params, 1, strm);
    if (res != cudaSuccess) {
        std::cerr << "CUDA failed to wait on vulkan semaphore" << std::endl;
        fatalExit();
    }
}

void cudaCPUWait(cudaExternalSemaphore_t sem)
{
    cudaGPUWait(sem, 0);
    cudaStreamSynchronize(0);
}

}

#endif
