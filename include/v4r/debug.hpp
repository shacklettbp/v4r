#ifndef DEBUG_HPP_INCLUDED
#define DEBUG_HPP_INCLUDED

#include <cstdint>

namespace v4r {

void saveFrame(const char *frame_path, void *dev_ptr, uint32_t width,
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

private:
    void startImpl() const;
    void endImpl() const;

    void *rdoc_impl_;
};

}

#endif
