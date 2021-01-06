#include <v4r.hpp>
#include <v4r/debug.hpp>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <random>
#include <thread>
#include <vector>

using namespace std;
using namespace v4r;

constexpr size_t max_load_frames = 10000;
constexpr size_t max_render_frames = 10000;
constexpr int num_threads = 1;
constexpr bool debug = false;

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
    if (argc < 2) {
        cerr << "SCENE VIEWS" << endl;
        exit(EXIT_FAILURE);
    }

    RenderDoc rdoc;

    using Pipeline = Unlit<RenderOutputs::Color | RenderOutputs::Depth,
                           DataSource::Texture>;

    BatchRenderer renderer({0, 1, num_threads, 1, 256, 256,
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0,
            0, 0, 0, 1
        )},
        RenderFeatures<Pipeline> { RenderOptions::CpuSynchronization }
    );

    vector<glm::mat4> init_views = readViews(argv[2]);
    size_t num_frames = min(init_views.size(), max_render_frames);

    pthread_barrier_t start_barrier;
    pthread_barrier_init(&start_barrier, nullptr, num_threads + 1);
    pthread_barrier_t end_barrier;
    pthread_barrier_init(&end_barrier, nullptr, num_threads + 1);

    vector<thread> threads;
    threads.reserve(num_threads);

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    atomic_bool go(false);

    for (int t_idx = 0; t_idx < num_threads; t_idx++) {
        threads.emplace_back(
            [num_frames, &go, &renderer, &start_barrier, &end_barrier, &scene]
            (vector<glm::mat4> cam_views)
            {
                auto cmd_stream = renderer.makeCommandStream();
                vector<Environment> envs;
                envs.emplace_back(cmd_stream.makeEnvironment(scene, 90, 0.01, 1000));

                random_device rd;
                mt19937 g(rd());
                shuffle(cam_views.begin(), cam_views.end(), g);

                pthread_barrier_wait(&start_barrier);
                while (!go.load()) {}

                for (size_t i = 0; i < num_frames; i++) {
                    auto mat = cam_views[i];
                    envs[0].setCameraView(mat);

                    cmd_stream.render(envs);
                    cmd_stream.waitForFrame();
                }

                pthread_barrier_wait(&end_barrier);
            }, 
            init_views);
    }

    pthread_barrier_wait(&start_barrier);
    if (debug) {
        rdoc.startFrame();
    }

    auto start = chrono::steady_clock::now();
    go.store(true);

    pthread_barrier_wait(&end_barrier);
    auto end = chrono::steady_clock::now();

    if (debug) {
        rdoc.endFrame();
    }

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "FPS: " << ((double)num_frames * num_threads / (double)diff.count()) * 1000.0 << endl;
    cout << "MS per 32 frames " << (double)diff.count() / ((double)num_frames * num_threads / 32.0) << endl;

    for (thread &t : threads) {
        t.join();
    }
}
