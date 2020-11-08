#ifndef V4R_DEBUG_HPP_INCLUDED
#define V4R_DEBUG_HPP_INCLUDED

#include <cstdint>

namespace v4r {

void saveFrame(const char *frame_path, const uint8_t *dev_ptr, uint32_t width,
               uint32_t height, uint32_t num_channels);

void saveFrame(const char *frame_path, const float *dev_ptr, uint32_t width,
               uint32_t height, uint32_t num_channels);

class RenderDoc {
public:
    RenderDoc();

    inline void startFrame() const
    {
        if (rdoc_impl_) startImpl();
    }

    inline void endFrame() const
    {
        if (rdoc_impl_) endImpl();
    }

    bool loaded() const { return !!rdoc_impl_; }

private:
    void startImpl() const;
    void endImpl() const;

    void *rdoc_impl_;
};

}

#endif
