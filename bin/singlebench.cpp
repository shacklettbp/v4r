#include <v4r.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>

using namespace std;
using namespace v4r;

constexpr uint32_t num_frames = 1000000;

constexpr uint32_t max_load_frames = 10000;

vector<glm::mat4> readViews(const char *dump_path)
{
    ifstream dump_file(dump_path, ios::binary);

    vector<glm::mat4> views;

    for (size_t i = 0; i < max_load_frames; i++) {
        float raw[16];
        dump_file.read((char *)raw, sizeof(float)*16);
        views.emplace_back(glm::inverse(
                glm::mat4(raw[0], raw[1], raw[2], raw[3],
                          raw[4], raw[5], raw[6], raw[7],
                          raw[8], raw[9], raw[10], raw[11],
                          raw[12], raw[13], raw[14], raw[15])));
    }

    return views;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << argv[0] << " scene batch_size res views" << endl;
        exit(EXIT_FAILURE);
    }

    uint32_t batch_size = stoul(argv[2]);
    uint32_t res = stoul(argv[3]);
    vector<glm::mat4> init_views;
    if (argc > 4) {
        init_views = readViews(argv[4]);
    } else {
        init_views = {glm::mat4(
            -1.19209e-07, 0, 1, 0,
            0, 1, 0, 0,
            -1, 0, -1.19209e-07, 0,
            -3.38921, 1.62114, -3.34509, 1)
        };
    }

    using Pipeline = BlinnPhong<RenderOutputs::Color,
                           DataSource::Uniform, DataSource::Uniform, DataSource::Uniform>;

    BatchRenderer renderer({0, 1, 1, batch_size, res, res,
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0,
            0, 0, 0, 1
        )},
        RenderFeatures<Pipeline> { RenderOptions::CpuSynchronization | RenderOptions::RayTracePrimary }
    );

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    auto cmd_stream = renderer.makeCommandStream();
    vector<Environment> envs;

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        envs.emplace_back(cmd_stream.makeEnvironment(scene, 90)); 
        envs.back().addLight(glm::vec3(5, 6, 0), glm::vec3(1, 2, 2));
        envs.back().addLight(glm::vec3(4, 6, -4), glm::vec3(2, 2, 1));

        envs[batch_idx].setCameraView(
            glm::inverse(glm::mat4(
                -1.19209e-07, 0, 1, 0,
                0, 1, 0, 0,
                -1, 0, -1.19209e-07, 0,
                -3.38921, 1.62114, -3.34509, 1)));
    }

    auto start = chrono::steady_clock::now();

    uint32_t num_iters = num_frames / batch_size;

    uint32_t cur_view = 0;

    for (uint32_t i = 0; i < num_iters; i++) {
        for (uint32_t j = 0; j < batch_size; j++) {
            envs[j].setCameraView(init_views[(cur_view++) % init_views.size()]);
        }
        cmd_stream.render(envs);
        cmd_stream.waitForFrame();
    }

    auto end = chrono::steady_clock::now();

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Batch size " << batch_size << ", Resolution " << res << ", FPS: " << ((double)num_iters * (double)batch_size /
            (double)diff.count()) * 1000.0 << endl;
}
