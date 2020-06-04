#ifndef CUDA_STATE_HPP_INCLUDED
#define CUDA_STATE_HPP_INCLUDED

#include <cstdint>

namespace v4r {

class CudaState {
public:
    CudaState(int buf_fd, uint64_t num_bytes);

    void *getPointer(uint64_t offset) const
    {
        return (uint8_t *)dev_ptr_ + offset;
    }

private:
    void *cuda_ext_mem_;
    void *dev_ptr_;
};

}

#endif
