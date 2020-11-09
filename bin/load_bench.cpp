#include <v4r/cuda.hpp>
#include <v4r/debug.hpp>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <atomic>

using namespace std;
using namespace v4r;

int globCB(const char *, int)
{
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << " num_threads scenes ..." << endl;
        exit(EXIT_FAILURE);
    }

    uint32_t num_threads = stoul(argv[1]);

    using Pipeline = Unlit<RenderOutputs::Color | RenderOutputs::Depth,
                           DataSource::Texture>;

    BatchRendererCUDA renderer({0, num_threads, 1, 1, 256, 256,
        4ul << 30,
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0,
            0, 0, 0, 1
        ) },
        RenderFeatures<Pipeline> { RenderOptions::CpuSynchronization }
    );
    auto cmd_stream = renderer.makeCommandStream();
    (void)cmd_stream;

    vector<AssetLoader> loaders;

    for (uint32_t i = 0; i < num_threads; i++) {
        auto loader = renderer.makeLoader();
        loaders.emplace_back(move(loader));
    }

    vector<thread> threads;
    atomic_uint32_t scene_idx = 0;
    vector<string> scene_names;

    for (int i = 2; i < argc; i++) {
        cout << argv[i] << "\n";
        scene_names.push_back(argv[i]);
    }
    cout.flush();

    for (uint32_t i = 0; i < num_threads; i++) {
        threads.emplace_back(
            [&loaders, &scene_names, &scene_idx](uint32_t thread_idx) {
            auto &loader = loaders[thread_idx];

            for (uint32_t j = 0; j < 100; j++) {
                uint32_t cur_idx = scene_idx++;
                cur_idx = cur_idx % scene_names.size();
                auto scene = loader.loadScene(scene_names[cur_idx]);
                (void)scene;
            }
        }, i);
    }

    for (thread &t : threads) {
        t.join();
    }

}
