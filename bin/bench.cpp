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
constexpr size_t max_render_frames = 3;
constexpr int num_threads = 4;
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
    if (argc < 3) {
        cerr << "SCENE VIEWS" << endl;
        exit(EXIT_FAILURE);
    }

    RenderDoc rdoc;

    RenderContext ctx({0, 256, 256,
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0,
            0, 0, 0, 1
        )
    });

    vector<glm::mat4> init_views = readViews(argv[2]);
    size_t num_frames = min(init_views.size(), max_render_frames);

    pthread_barrier_t start_barrier;
    pthread_barrier_init(&start_barrier, nullptr, num_threads + 1);
    pthread_barrier_t end_barrier;
    pthread_barrier_init(&end_barrier, nullptr, num_threads + 1);

    vector<thread> threads;
    threads.reserve(num_threads);

    atomic_bool go(false);

    for (int t_idx = 0; t_idx < num_threads; t_idx++) {
        threads.emplace_back(
            [num_frames, &go, &ctx, &start_barrier, &end_barrier]
            (const char *scene_path, vector<glm::mat4> views)
            {
                auto cmd_stream = ctx.makeCommandStream();
                auto scene = cmd_stream.loadScene(scene_path);

                random_device rd;
                mt19937 g(rd());
                shuffle(views.begin(), views.end(), g);

                auto cam = ctx.makeCamera(90, 0.01, 1000, glm::mat4(1.f));

                pthread_barrier_wait(&start_barrier);
                while (!go.load()) {}

                for (size_t i = 0; i < num_frames; i++) {
                    const glm::mat4 &mat = views[i];
                    cam.setView(mat);
                    auto frame = cmd_stream.render(scene, cam);
                    (void)frame;
                }

                pthread_barrier_wait(&end_barrier);

                cmd_stream.dropScene(move(scene));
            }, 
            argv[1], init_views);
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

    for (thread &t : threads) {
        t.join();
    }
}
