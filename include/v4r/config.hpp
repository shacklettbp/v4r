#ifndef V4R_CONFIG_HPP_INCLUDED
#define V4R_CONFIG_HPP_INCLUDED

#include <glm/glm.hpp>

namespace v4r {

struct RenderFeatures {
    enum class MeshColor : uint32_t {
        Texture,
        Vertex,
        None
    };

    enum Pipeline : uint32_t {
        Unlit,
        Lit,
        Shadowed
    };

    enum class Outputs : uint32_t {
        Color = 1 << 0,
        Depth = 1 << 1
    };

    enum class Options : uint32_t {
        CpuSynchronization = 1 << 0,
        DoubleBuffered = 1 << 1
    };

    MeshColor colorSrc;
    Pipeline pipeline;
    Outputs outputs;
    Options options;
};

struct RenderConfig {
    int gpuID;
    uint32_t numLoaders;
    uint32_t numStreams;
    uint32_t batchSize;
    uint32_t imgWidth;
    uint32_t imgHeight;
    glm::mat4 coordinateTransform;
    RenderFeatures features;
};

inline RenderFeatures::Outputs & operator|=(RenderFeatures::Outputs &a,
                                            RenderFeatures::Outputs b)
{
    return a = RenderFeatures::Outputs {
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b)
    };
}

inline RenderFeatures::Outputs operator|(RenderFeatures::Outputs a,
                                         RenderFeatures::Outputs b)
{
    return a |= b;
}

inline bool operator&(RenderFeatures::Outputs flags,
                      RenderFeatures::Outputs mask)
{
    uint32_t mask_int = static_cast<uint32_t>(mask);
    return (static_cast<uint32_t>(flags) & mask_int) == mask_int;
}

inline RenderFeatures::Options & operator|=(RenderFeatures::Options &a,
                                            RenderFeatures::Options b)
{
    return a = RenderFeatures::Options {
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b)
    };
}

inline RenderFeatures::Options operator|(RenderFeatures::Options a,
                                         RenderFeatures::Options b)
{
    return a |= b;
}

inline bool operator&(RenderFeatures::Options flags,
                      RenderFeatures::Options mask)
{
    uint32_t mask_int = static_cast<uint32_t>(mask);
    return (static_cast<uint32_t>(flags) & mask_int) == mask_int;
}

}

#endif
