#include <v4r/cuda.hpp>
#include <v4r/debug.hpp>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace v4r;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << "scene batch_size" << endl;
        exit(EXIT_FAILURE);
    }

    uint32_t batch_size = stoul(argv[2]);

    using Pipeline = Unlit<RenderOutputs::Color | RenderOutputs::Depth,
                           DataSource::Texture>;

    BatchRendererCUDA renderer({0, 1, 1, batch_size, 256, 256,
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0, 0, 0, 0, 1) },
        RenderFeatures<Pipeline> {
            RenderOptions::CpuSynchronization |
                RenderOptions::RayTracePrimary,
        }
    );

    RenderDoc rdoc {};

    rdoc.startFrame();
    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);
    auto cmd_stream = renderer.makeCommandStream();
    vector<Environment> envs;

    glm::mat4 base = glm::inverse(
            glm::mat4(-1.19209e-07, 0, 1, 0,
                      0, 1, 0, 0,
                      -1, 0, -1.19209e-07, 0,
                      -3.38921, 1.62114, -3.34509, 1));
    
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        envs.emplace_back(cmd_stream.makeEnvironment(scene, 90, 0.01, 1000));

        envs.back().setCameraView(
                glm::rotate(base, glm::radians(10.f * batch_idx),
                            glm::vec3(0.f, 0.f, 1.f)));
    }

    cmd_stream.render(envs);
    cmd_stream.waitForFrame();
    rdoc.endFrame();

    uint8_t *base_color_ptr = cmd_stream.getColorDevicePtr();
    float *base_depth_ptr = cmd_stream.getDepthDevicePtr();

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        saveFrame(("/tmp/out_color_" + to_string(batch_idx) + ".bmp").c_str(),
                  base_color_ptr + batch_idx * 256 * 256 * 4,
                  256, 256, 4);
        saveFrame(("/tmp/out_depth_" + to_string(batch_idx) + ".bmp").c_str(),
                  base_depth_ptr + batch_idx * 256 * 256,
                  256, 256, 1);
    }
}
