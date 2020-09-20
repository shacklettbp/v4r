#include <v4r/preprocess.hpp>

#include "utils.hpp"

using namespace std;

namespace v4r {

struct SceneData {
};

static SceneData parseSceneData(string_view gltf_path)
{
    (void)gltf_path;
    return SceneData {
    };
}

ScenePreprocessor::ScenePreprocessor(string_view gltf_path)
    : scene_data_(new SceneData(parseSceneData(gltf_path)))
{
}

void ScenePreprocessor::dump(string_view out_path)
{
    (void)out_path;
}

template struct HandleDeleter<SceneData>;

}
