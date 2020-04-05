#include <v4r.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace v4r;

constexpr int num_frames = 10000;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Specify scene" << endl;
        exit(EXIT_FAILURE);
    }

    RenderContext ctx({0, 256, 256,
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0,
            0, 0, 0, 1
        )
    });

    Camera cam = ctx.makeCamera(90, 0.01, 1000,
        glm::inverse(glm::mat4(-1.19209e-07, 0, 1, 0,
                  0, 1, 0, 0,
                  -1, 0, -1.19209e-07, 0,
                  -3.38921, 1.62114, -3.34509, 1)));


    auto cmd_stream = ctx.makeCommandStream();


    auto scene_handle = cmd_stream.loadScene(argv[1]);

    auto start = chrono::steady_clock::now();

    for (int i = 0; i < 10000; i++) {
        cmd_stream.render(scene_handle, cam);
    }

    auto end = chrono::steady_clock::now();

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "FPS: " << ((double)num_frames / (double)diff.count()) * 1000.0 << endl;

    cmd_stream.dropScene(move(scene_handle));
}
