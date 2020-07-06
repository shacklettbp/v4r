#include <v4r/debug.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector> 

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

using namespace std;

namespace v4r {

template<typename T>
static vector<T> copyToHost(const T *dev_ptr, uint32_t width,
                            uint32_t height, uint32_t num_channels)
{
    uint64_t num_pixels = width * height * num_channels;

    vector<T> buffer(num_pixels);

    cudaMemcpy(buffer.data(), dev_ptr, sizeof(T) * num_pixels,
               cudaMemcpyDeviceToHost);

    return buffer;
}

void saveFrame(const char *fname, const float *dev_ptr,
               uint32_t width, uint32_t height, uint32_t num_channels)
{
    auto buffer = copyToHost(dev_ptr, width, height, num_channels);

    vector<uint8_t> sdr_buffer(buffer.size());
    for (unsigned i = 0; i < buffer.size(); i++) {
        float v = buffer[i];
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        sdr_buffer[i] = v * 255;
    }

    stbi_write_bmp(fname, width, height, num_channels, sdr_buffer.data());
}

void saveFrame(const char *fname, const uint8_t *dev_ptr,
               uint32_t width, uint32_t height, uint32_t num_channels)
{
    auto buffer = copyToHost(dev_ptr, width, height, num_channels);

    stbi_write_bmp(fname, width, height, num_channels, buffer.data());
}

}
