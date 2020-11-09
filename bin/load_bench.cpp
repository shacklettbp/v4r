#include <v4r/cuda.hpp>
#include <v4r/debug.hpp>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <atomic>
#include <unistd.h>

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

    BatchRendererCUDA renderer({0, num_threads, 1, 1024, 64, 64,
        16ul << 30,
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0,
            0, 0, 0, 1
        ) },
        RenderFeatures<Pipeline> { RenderOptions::CpuSynchronization }
    );
    auto cmd_stream = renderer.makeCommandStream();

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

    scene_idx++;

    auto base_scene = loaders[0].loadScene(scene_names[0]);
    vector<Environment> envs;
    
    for (uint32_t batch_idx = 0; batch_idx < 1024; batch_idx++) {
        envs.emplace_back(cmd_stream.makeEnvironment(base_scene, 90)); 

        envs[batch_idx].setCameraView(
            glm::inverse(glm::mat4(
                -1.19209e-07, 0, 1, 0,
                0, 1, 0, 0,
                -1, 0, -1.19209e-07, 0,
                -3.38921, 1.62114, -3.34509, 1)));
    }

    atomic_uint32_t launched = 0;
    atomic_bool exit = false;

    vector<shared_ptr<Scene>> scenes;
    for (uint32_t i = 0; i < num_threads; i++) {
        scenes.push_back(base_scene);
    }

    atomic_thread_fence(std::memory_order_seq_cst);

    for (uint32_t i = 0; i < num_threads; i++) {
        threads.emplace_back(
            [&loaders, &scene_names, &scene_idx, &scenes, &launched, &exit](uint32_t thread_idx) {
            auto &loader = loaders[thread_idx];
            launched++;

            while (!exit.load()) {
                uint32_t cur_idx = scene_idx++;
                cur_idx = cur_idx % scene_names.size();
                cout << "Loading " << scene_names[cur_idx] << endl;
                auto scene = loader.loadScene(scene_names[cur_idx]);
                cout << "Loaded " << scene_names[cur_idx] << " " << &*scene << endl;

                atomic_store(&scenes[thread_idx], scene);
            }
        }, i);
    }

    while (launched != threads.size()) {
    }


    uint32_t num_iters = 10000;

    double diff = 0;
    double launch_diff = 0;
    for (uint32_t i = 0; i < num_iters; i++) {

        for (uint32_t j = 0; j < num_threads; j++) {
            auto cur_scene = atomic_load(&scenes[j]);
            envs[(i + j) % 1024] = cmd_stream.makeEnvironment(cur_scene, 90);
            envs[(i + j) % 1024].setCameraView(
                glm::inverse(glm::mat4(
                    -1.19209e-07, 0, 1, 0,
                    0, 1, 0, 0,
                    -1, 0, -1.19209e-07, 0,
                    -3.38921, 1.62114, -3.34509, 1)));
        }

        auto start = chrono::steady_clock::now();
        cmd_stream.render(envs);
        auto mid = chrono::steady_clock::now();
        cmd_stream.waitForFrame();
        auto end = chrono::steady_clock::now();
        diff += chrono::duration_cast<chrono::milliseconds>(end - start).count();
        launch_diff += chrono::duration_cast<chrono::milliseconds>(mid - start).count();
    }

    exit = true;

    
    cout << "FPS: " << ((double)num_iters * (double)1024 /
            (double)diff) * 1000.0 << endl;
    cout << scene_idx << endl;
    cout << diff / ((double)num_iters * (double)1024) << endl;

    for (auto &t : threads) {
        t.join();
    }

}
