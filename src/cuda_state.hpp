#ifndef CUDA_STATE_HPP_INCLUDED
#define CUDA_STATE_HPP_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>

#include "vulkan_handles.hpp"

namespace v4r {

class CudaStreamState {
public:
    CudaStreamState(uint8_t *color_ptr, float *depth_ptr,
                    int sem_fd);

    uint8_t * getColor() const { return color_ptr_; }
    float * getDepth() const { return depth_ptr_; }

    cudaExternalSemaphore_t getSemaphore() const { return ext_sem_; }

private:
    uint8_t *color_ptr_;
    float *depth_ptr_;
    cudaExternalSemaphore_t ext_sem_;
};

class CudaState {
public:
    CudaState(int cuda_id, int buf_fd, uint64_t num_bytes);

    void setActiveDevice() const;

    void *getPointer(uint64_t offset) const;

private:
    int cuda_id_;
    cudaExternalMemory_t ext_mem_;
    void *dev_ptr_;
};

DeviceUUID getUUIDFromCudaID(int cuda_id);

inline void cudaGPUWait(cudaExternalSemaphore_t sem, cudaStream_t strm);
inline void cudaCPUWait(cudaExternalSemaphore_t sem);

}

#ifndef CUDA_STATE_INL_INCLUDED
#include "cuda_state.inl"
#endif

#endif
