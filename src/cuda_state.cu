#include "cuda_state.hpp"

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include "utils.hpp"

using namespace std;

namespace v4r {

cudaExternalMemory_t importBuffer(int buf_fd, uint64_t num_bytes)
{
    cudaExternalMemoryHandleDesc cuda_ext_info {};
    cuda_ext_info.type = cudaExternalMemoryHandleTypeOpaqueFd;
    cuda_ext_info.handle.fd = buf_fd;
    cuda_ext_info.size = num_bytes;
    cuda_ext_info.flags = cudaExternalMemoryDedicated;

    cudaExternalMemory_t ext_mem;
    cudaError_t res = cudaImportExternalMemory(&ext_mem, &cuda_ext_info);

    if (res != cudaSuccess) {
        cerr << "CUDA failed to import vulkan buffer" << endl;
        fatalExit();
    }

    return ext_mem;
}

void *mapExternal(cudaExternalMemory_t ext_mem, uint64_t num_bytes)
{
    void *dev_ptr;
    cudaExternalMemoryBufferDesc ext_info;
    ext_info.offset = 0;
    ext_info.size = num_bytes;
    ext_info.flags = 0;

    cudaError_t res = cudaExternalMemoryGetMappedBuffer(&dev_ptr, ext_mem, &ext_info);
    if (res != cudaSuccess) {
        cerr << "CUDA failed to map vulkan buffer" << endl;
        fatalExit();
    }

    return dev_ptr;
}

CudaState::CudaState(int buf_fd, uint64_t num_bytes)
    : cuda_ext_mem_(importBuffer(buf_fd, num_bytes)),
      dev_ptr_(mapExternal((cudaExternalMemory_t)cuda_ext_mem_, num_bytes))
{}

}
