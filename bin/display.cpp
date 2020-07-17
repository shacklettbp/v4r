#include <v4r/display.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace std;
using namespace v4r;

constexpr uint32_t num_frames = 10000;

static GLFWwindow * makeWindow(const glm::u32vec2 &dim)
{
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    return glfwCreateWindow(dim.x, dim.y,
                            "V4R", NULL, NULL);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << " scene batch_size [output]" << endl;
        exit(EXIT_FAILURE);
    }

    if (!glfwInit()) {
        cerr << "GLFW failed to initialize" << endl;
        exit(EXIT_FAILURE);
    }

    RenderFeatures::Outputs outputs =
        RenderFeatures::Outputs::Color | RenderFeatures::Outputs::Depth;
    RenderFeatures::MeshColor color_src = RenderFeatures::MeshColor::Texture;

    if (argc > 3) {
        if (!strcmp(argv[3], "color")) {
            outputs = RenderFeatures::Outputs::Color;
        } else if (!strcmp(argv[3], "depth")) {
            outputs = RenderFeatures::Outputs::Depth;
            color_src = RenderFeatures::MeshColor::None;
        }
    }

    uint32_t batch_size = stoul(argv[2]);

    BatchPresentRenderer renderer({0, 1, 1, batch_size, 256, 256,
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0,
            0, 0, 0, 1
        ),
        {
            color_src,
            RenderFeatures::Pipeline::Unlit,
            outputs,
            RenderFeatures::Options::DoubleBuffered
        }
    });

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

    RenderSync prevsync = cmd_stream.render(envs);

    for (uint32_t i = 1; i < num_iters; i++) {
        RenderSync newsync = cmd_stream.render(envs);
        prevsync.cpuWait();
        prevsync = move(newsync);
    }

    auto end = chrono::steady_clock::now();

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "FPS: " << ((double)num_iters * (double)batch_size /
            (double)diff.count()) * 1000.0 << endl;
}
