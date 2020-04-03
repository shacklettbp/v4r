#include <v4r/debug.hpp>
#include <vector> 


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

using namespace std;

namespace v4r {

void saveFrame(const char *fname, void *dev_ptr,
               uint32_t width, uint32_t height, uint32_t num_channels)
{
    uint64_t num_pixels = width * height * num_channels;

    vector<float> buffer(num_pixels);

    cudaMemcpy(buffer.data(), dev_ptr, sizeof(float) * num_pixels,
               cudaMemcpyDeviceToHost);

    vector<uint8_t> sdr_buffer(num_pixels);
    for (unsigned i = 0; i < buffer.size(); i++) {
        sdr_buffer[i] = buffer[i] * 255;
    }

    stbi_write_bmp(fname, width, height, num_channels, sdr_buffer.data());
}

}
