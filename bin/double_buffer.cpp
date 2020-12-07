#include <v4r.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace std;
using namespace v4r;

constexpr uint32_t num_frames = 1000000;

BatchRenderer makeRenderer(const RenderConfig &cfg,
                           RenderOutputs desired_outputs,
                           RenderOptions opts)
{
    if ((desired_outputs & RenderOutputs::Color) &&
        (desired_outputs & RenderOutputs::Depth)) {
        return BatchRenderer(cfg,
            RenderFeatures<Unlit<RenderOutputs::Color | RenderOutputs::Depth,
                                 DataSource::Texture>> { opts });
    } else if (desired_outputs & RenderOutputs::Color) {
        return BatchRenderer(cfg,
            RenderFeatures<Unlit<RenderOutputs::Color,
                                 DataSource::Texture>> { opts });
    } else {
        return BatchRenderer(cfg,
            RenderFeatures<Unlit<RenderOutputs::Depth,
                                 DataSource::None>> { opts });
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << " scene batch_size [output]" << endl;
        exit(EXIT_FAILURE);
    }

    RenderOutputs outputs = RenderOutputs::Color | RenderOutputs::Depth;

    if (argc > 3) {
        if (!strcmp(argv[3], "color")) {
            outputs = RenderOutputs::Color;
        } else if (!strcmp(argv[3], "depth")) {
            outputs = RenderOutputs::Depth;
        }
    }

    uint32_t batch_size = stoul(argv[2]);

    BatchRenderer renderer = makeRenderer({0, 1, 1, batch_size, 256, 256,
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0,
            0, 0, 0, 1
        )},
        outputs, RenderOptions::CpuSynchronization | RenderOptions::DoubleBuffered);

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    CommandStream cmd_stream = renderer.makeCommandStream();
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

    uint32_t prev_frame = cmd_stream.render(envs);

    for (uint32_t i = 1; i < num_iters; i++) {
        uint32_t new_frame = cmd_stream.render(envs);
        cmd_stream.waitForFrame(prev_frame);
        prev_frame = new_frame;
    }

    auto end = chrono::steady_clock::now();

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "FPS: " << ((double)num_iters * (double)batch_size /
            (double)diff.count()) * 1000.0 << endl;
}
