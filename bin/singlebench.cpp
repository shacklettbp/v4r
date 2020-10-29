#include <v4r.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace v4r;

constexpr uint32_t num_frames = 1000000;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << " scene batch_size" << endl;
        exit(EXIT_FAILURE);
    }

    uint32_t batch_size = stoul(argv[2]);

    using Pipeline = Unlit<RenderOutputs::Color | RenderOutputs::Depth,
                           DataSource::Texture>;

    BatchRenderer renderer({0, 1, 1, batch_size, 256, 256, 4ul << 30,
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0,
            0, 0, 0, 1
        )},
        RenderFeatures<Pipeline> { RenderOptions::CpuSynchronization }
    );

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    auto cmd_stream = renderer.makeCommandStream();
    vector<Environment> envs;

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        envs.emplace_back(cmd_stream.makeEnvironment(scene, 90)); 

        envs[batch_idx].setCameraView(
            glm::inverse(glm::mat4(
                -1.19209e-07, 0, 1, 0,
                0, 1, 0, 0,
                -1, 0, -1.19209e-07, 0,
                -3.38921, 1.62114, -3.34509, 1)));
    }

    auto start = chrono::steady_clock::now();

    uint32_t num_iters = num_frames / batch_size;

    for (uint32_t i = 0; i < num_iters; i++) {
        cmd_stream.render(envs);
        cmd_stream.waitForFrame();
    }

    auto end = chrono::steady_clock::now();

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "FPS: " << ((double)num_iters * (double)batch_size /
            (double)diff.count()) * 1000.0 << endl;
}
