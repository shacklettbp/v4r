#ifndef CUDA_STATE_HPP_INCLUDED
#define CUDA_STATE_HPP_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>

namespace v4r {

class CudaStreamState {
public:
    CudaStreamState(uint8_t *color_ptr, float *depth_ptr, int sem_fd);

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
    CudaState(int buf_fd, uint64_t num_bytes);

    void *getPointer(uint64_t offset) const;

private:
    cudaExternalMemory_t ext_mem_;
    void *dev_ptr_;
};

void cudaGPUWait(cudaExternalSemaphore_t sem, cudaStream_t strm);
void cudaCPUWait(cudaExternalSemaphore_t sem);

}

#endif
