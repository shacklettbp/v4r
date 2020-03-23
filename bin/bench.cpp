#include <v4r.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <vector>

using namespace std;
using namespace v4r;

constexpr int num_frames = 10000;
constexpr int num_threads = 2;
constexpr int frames_per_thread = num_frames / num_threads;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Specify scene" << endl;
        exit(EXIT_FAILURE);
    }

    RenderContext ctx({0, 128, 128});

    vector<RenderContext::SceneHandle> handles;
    vector<RenderContext::CommandStream> streams;
    handles.reserve(num_threads);
    streams.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        auto cmd_stream = ctx.makeCommandStream();
        handles.emplace_back(move(cmd_stream.loadScene(argv[1])));
        streams.emplace_back(move(cmd_stream));
    }

    vector<thread> threads;
    threads.reserve(num_threads);

    auto start = chrono::steady_clock::now();

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(
            [] (RenderContext::CommandStream &cmd_stream,
                RenderContext::SceneHandle &scene) {
                for (int f = 0; f < frames_per_thread; f++) {
                    auto frame = cmd_stream.renderCamera(scene);
                    (void)frame;
                }
        }, ref(streams[i]), ref(handles[i]));
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
