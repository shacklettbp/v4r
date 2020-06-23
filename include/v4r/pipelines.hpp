#ifndef V4R_PIPELINES_HPP_INCLUDED
#define V4R_PIPELINES_HPP_INCLUDED

#include <v4r/assets.hpp>

namespace v4r {

struct UnlitPipeline {
    using VertexType = UnlitVertex;
    using MaterialDescType = UnlitMaterialDescription;
};

struct VertexColorPipeline {
    using VertexType = ColoredVertex;
    using MaterialDescType = UnlitMaterialDescription;
};

}

#endif
