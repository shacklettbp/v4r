#ifndef V4R_CONFIG_HPP_INCLUDED
#define V4R_CONFIG_HPP_INCLUDED

#include <v4r/fwd.hpp>

#include <memory>
#include <glm/glm.hpp>

namespace v4r {

enum class ColorSource : uint32_t {
    None,
    Vertex,
    Constant,
    Texture
};

enum class RenderOutputs : uint32_t {
    Color = 1 << 0,
    Depth = 1 << 1
};

enum class RenderOptions : uint32_t {
    CpuSynchronization = 1 << 0,
    DoubleBuffered = 1 << 1,
    VerticalSync = 1 << 2,
    NonuniformScale = 1 << 4
};

struct NoMaterial {
private:
    NoMaterial();
};

template <ColorSource color>
struct UnlitPipeline {
};

template <ColorSource diffuse_color,
          ColorSource specular_color>
struct BlinnPhongPipeline {
};

template <typename PipelineT,
          RenderOutputs out,
          RenderOptions opt>
struct FeatureComponents {
    using PipelineType = PipelineT;
    static constexpr RenderOutputs outputs = out; 
    static constexpr RenderOptions options = opt; 
};

template <typename PipelineType,
          RenderOutputs outputs,
          RenderOptions options>
struct RenderFeatures; 

template <typename FeaturesType>
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

template <>
struct UnlitPipeline<ColorSource::None> {
    struct Vertex {
        glm::vec3 position;
    };
};

template <RenderOptions options>
struct RenderFeatures<UnlitPipeline<ColorSource::Vertex>,
                      RenderOutputs::Color | RenderOutputs::Depth,
                      options> {
    struct Vertex {
        glm::vec3 position;
        glm::u8vec3 color;
    };
};

template <>
struct UnlitPipeline<ColorSource::Constant> {
    struct Vertex {
        glm::vec3 position;
    };

    struct MaterialParams {
        glm::vec3 color;
    };
};

template <>
struct UnlitPipeline<ColorSource::Texture> {
    struct Vertex {
        glm::vec3 position;
        glm::vec2 uv;
    };
    
    struct MaterialParams {
        std::shared_ptr<Texture> color;
    };
};

struct BlinnPhongVertex {
    glm::vec3 position;
    glm::vec3 normal;
};

struct BlinnPhongTexturedVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

}

#endif
