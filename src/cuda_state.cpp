#include "cuda_state.hpp"

#include "utils.hpp"

#include <iostream>
#include <cstring>

using namespace std;

namespace v4r {

static cudaExternalSemaphore_t importSemaphore(int sem_fd)
{
    cudaExternalSemaphoreHandleDesc cuda_ext_info {};
    cuda_ext_info.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    cuda_ext_info.handle.fd = sem_fd;
    cuda_ext_info.flags = 0;

    cudaExternalSemaphore_t ext_sem;
    cudaError_t res = cudaImportExternalSemaphore(&ext_sem, &cuda_ext_info);
    if (res != cudaSuccess) {
        cerr << "CUDA failed to import vulkan semaphore" << endl;
        fatalExit();
    }

    return ext_sem;
}

CudaStreamState::CudaStreamState(uint8_t *color_ptr, float *depth_ptr,
                                 int sem_fd)
    : color_ptr_(color_ptr), depth_ptr_(depth_ptr),
      ext_sem_()
{
    ext_sem_ = importSemaphore(sem_fd);
}

static cudaExternalMemory_t importBuffer(int buf_fd, uint64_t num_bytes)
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

static void *mapExternal(cudaExternalMemory_t ext_mem, uint64_t num_bytes)
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

CudaState::CudaState(int cuda_id, int fd, uint64_t num_bytes)
    : cuda_id_(cuda_id),
      ext_mem_(),
      dev_ptr_()
{
    setActiveDevice();
    ext_mem_ = importBuffer(fd, num_bytes);
    dev_ptr_ = mapExternal(ext_mem_, num_bytes);
}

void CudaState::setActiveDevice() const
{
    cudaError_t res = cudaSetDevice(cuda_id_);

    if (res != cudaSuccess) {
        cerr << "CUDA failed to set device" << endl;
        fatalExit();
    }
}

void * CudaState::getPointer(uint64_t offset) const
{
        return (uint8_t *)dev_ptr_ + offset;
}

DeviceUUID getUUIDFromCudaID(int cuda_id)
{
    DeviceUUID uuid;

    int device_count;
    cudaError_t res = cudaGetDeviceCount(&device_count);
    if (res != cudaSuccess) {
        cerr << "CUDA failed to enumerate devices" << endl;
        fatalExit();
    }

    if (device_count <= cuda_id) {
        cerr << cuda_id << " is not a valid CUDA ID given " << device_count <<
             " supported devices" << endl;
        fatalExit();
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, cuda_id);

    if (props.computeMode == cudaComputeModeProhibited) {
        cerr << cuda_id << " corresponds to a prohibited device" << endl;
        fatalExit();
    }

    memcpy(uuid.data(), &props.uuid,
           sizeof(DeviceUUID::value_type) * uuid.size());

    return uuid;
}

}
