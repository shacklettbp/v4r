#ifndef V4R_CONFIG_HPP_INCLUDED
#define V4R_CONFIG_HPP_INCLUDED

#include <v4r/fwd.hpp>

#include <memory>
#include <glm/glm.hpp>

namespace v4r {

enum class DataSource : uint32_t {
    None,
    Vertex,
    Uniform,
    Texture
};

enum class RenderOutputs : uint32_t {
    Color = 1 << 0,
    Depth = 1 << 1
};

enum class RenderOptions : uint32_t {
    CpuSynchronization = 1 << 0,
    DoubleBuffered = 1 << 1,
    VerticalSync = 1 << 2
};

struct NoMaterial {
private:
    NoMaterial();
};

template <typename PipelineType>
struct RenderFeatures {
    RenderOptions options;

    using VertexType = typename PipelineType::VertexType;
    using MaterialParamsType = typename PipelineType::MaterialParamsType;
};

struct RenderConfig {
    int gpuID;
    uint32_t numLoaders;
    uint32_t numStreams;
    uint32_t batchSize;
    uint32_t imgWidth;
    uint32_t imgHeight;
    glm::mat4 coordinateTransform;
};

inline constexpr RenderOutputs & operator|=(RenderOutputs &a,
                                            RenderOutputs b)
{
    return a = static_cast<RenderOutputs>(
                static_cast<uint32_t>(a) | static_cast<uint32_t>(b)
    );
}

inline constexpr RenderOutputs operator|(RenderOutputs a,
                                         RenderOutputs b)
{
    return a |= b;
}

inline constexpr bool operator&(RenderOutputs flags,
                                RenderOutputs mask)
{
    uint32_t mask_int = static_cast<uint32_t>(mask);
    return (static_cast<uint32_t>(flags) & mask_int) == mask_int;
}

inline constexpr RenderOptions & operator|=(RenderOptions &a,
                                            RenderOptions b)
{
    return a = static_cast<RenderOptions>(
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b)
    );
}

inline constexpr RenderOptions operator|(RenderOptions a,
                                         RenderOptions b)
{
    return a |= b;
}

inline bool operator&(RenderOptions flags,
                      RenderOptions mask)
{
    uint32_t mask_int = static_cast<uint32_t>(mask);
    return (static_cast<uint32_t>(flags) & mask_int) == mask_int;
}

}

#include "pipelines/config.inl"

#endif
