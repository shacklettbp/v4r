#ifndef V4R_PREPROCESS_HPP_INCLUDED
#define V4R_PREPROCESS_HPP_INCLUDED

#include <v4r/utils.hpp>

#include <string_view>

namespace v4r {

struct SceneData;

class ScenePreprocessor {
public:
    ScenePreprocessor(std::string_view gltf_path);

    void dump(std::string_view out_path);

private:
    Handle<SceneData> scene_data_;
};

}

#endif
