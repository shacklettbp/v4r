#include <v4r/display.hpp>
#include <v4r/debug.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <random>
#include <algorithm>

using namespace std;
using namespace v4r;

constexpr uint32_t num_frames = 30000;
constexpr uint32_t max_load_frames = num_frames;

static GLFWwindow * makeWindow(const glm::u32vec2 &dim)
{
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    return glfwCreateWindow(dim.x, dim.y,
                            "V4R", NULL, NULL);
}

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

BatchPresentRenderer makeRenderer(const RenderConfig &cfg,
                           RenderOutputs desired_outputs,
                           RenderOptions opts)
{
    if ((desired_outputs & RenderOutputs::Color) &&
        (desired_outputs & RenderOutputs::Depth)) {
        return BatchPresentRenderer(cfg,
            RenderFeatures<Unlit<RenderOutputs::Color | RenderOutputs::Depth,
                                 DataSource::Texture>> { opts }, true);
    } else if (desired_outputs & RenderOutputs::Color) {
        return BatchPresentRenderer(cfg,
            RenderFeatures<Unlit<RenderOutputs::Color,
                                 DataSource::Texture>> { opts }, true);
    } else {
        return BatchPresentRenderer(cfg,
            RenderFeatures<Unlit<RenderOutputs::Depth,
                                 DataSource::None>> { opts }, true);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << argv[0] << " scene views batch_size [output]" << endl;
        exit(EXIT_FAILURE);
    }

    if (!glfwInit()) {
        cerr << "GLFW failed to initialize" << endl;
        exit(EXIT_FAILURE);
    }

    RenderDoc rdoc;

    RenderOutputs outputs =
        RenderOutputs::Color | RenderOutputs::Depth;

    if (argc > 4) {
        if (!strcmp(argv[4], "color")) {
            outputs = RenderOutputs::Color;
        } else if (!strcmp(argv[4], "depth")) {
            outputs = RenderOutputs::Depth;
        }
    }

    vector<glm::mat4> views = readViews(argv[2]);
    auto rng = default_random_engine {};
    shuffle(begin(views), end(views), rng);

    uint32_t batch_size = stoul(argv[3]);

    BatchPresentRenderer renderer = makeRenderer(
        {0, 1, 1, batch_size, 256, 256,
            glm::mat4(
                1, 0, 0, 0,
                0, -1.19209e-07, -1, 0,
                0, 1, -1.19209e-07, 0,
                0, 0, 0, 1
            )},
        outputs, RenderOptions::DoubleBuffered |
            RenderOptions::CpuSynchronization
    );

    glm::u32vec2 frame_dim = renderer.getFrameDimensions();
    GLFWwindow *window = makeWindow(frame_dim);

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    auto cmd_stream = renderer.makeCommandStream(window);
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

    rdoc.startFrame();
    uint32_t prev_frame = cmd_stream.render(envs);
    rdoc.endFrame();

    uint32_t view_idx = 0;
    while (true) {
        for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            envs[batch_idx].setCameraView(views[view_idx]);
            view_idx = (view_idx + 1) % views.size();
        }

        rdoc.startFrame();
        uint32_t new_frame = cmd_stream.render(envs);
        rdoc.endFrame();
        cmd_stream.waitForFrame(prev_frame);
        prev_frame = new_frame;
    }

    auto end = chrono::steady_clock::now();

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "FPS: " << ((double)num_iters * (double)batch_size /
            (double)diff.count()) * 1000.0 << endl;
}
