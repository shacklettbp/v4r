#include <v4r.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <random>
#include <vector>

using namespace std;
using namespace v4r;

constexpr int num_frames = 10000;
constexpr int num_threads = 1;
constexpr int frames_per_thread = num_frames / num_threads;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Specify scene" << endl;
        exit(EXIT_FAILURE);
    }

    RenderContext ctx({0, 128, 128});

    vector<RenderContext::SceneHandle> handles;
    vector<RenderContext::CommandStream> streams;
    vector<Camera> cameras;
    handles.reserve(num_threads);
    streams.reserve(num_threads);
    cameras.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        auto cmd_stream = ctx.makeCommandStream();
        handles.emplace_back(move(cmd_stream.loadScene(argv[1])));
        streams.emplace_back(move(cmd_stream));
        cameras.emplace_back(60, 1, 0.1, 1000,
                             glm::vec3(1.f, 1.f, 0.f),
                             glm::vec3(0.f, 1.f, 0.f),
                             glm::vec3(0.f, 1.f, 0.f));
    }

    vector<thread> threads;
    threads.reserve(num_threads);

    auto start = chrono::steady_clock::now();

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(
            [] (RenderContext::CommandStream &cmd_stream,
                RenderContext::SceneHandle &scene,
                Camera &camera) {
                random_device rd;
                mt19937 gen(rd());
                uniform_int_distribution<> dis(1, 90);

                for (int f = 0; f < frames_per_thread; f++) {
                    camera.rotate(dis(gen), glm::vec3(0.f, 1.f, 0.f));
                    auto frame = cmd_stream.render(scene, camera);
                    (void)frame;
                }
        }, ref(streams[i]), ref(handles[i]), ref(cameras[i]));
    }

    for (thread &t : threads) {
        t.join();
    }

    auto end = chrono::steady_clock::now();

    for (int i = 0; i < num_threads; i++) {
        streams[i].dropScene(move(handles[i]));
    }

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "FPS: " << ((double)num_frames / (double)diff.count()) * 1000.0 << endl;
}
