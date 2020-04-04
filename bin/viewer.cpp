#include <v4r.hpp>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace v4r;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Specify scene" << endl;
        exit(EXIT_FAILURE);
    }

    RenderContext ctx({0, 128, 128,
        glm::mat4(
            1, 0, 0, 0,
            0, -1.19209e-07, -1, 0,
            0, 1, -1.19209e-07, 0,
            0, 0, 0, 1
        )
    });

    auto cmd_stream = ctx.makeCommandStream();

    auto scene_handle = cmd_stream.loadScene(argv[1]);

    Camera cam = ctx.makeCamera(60, 0.01, 1000,
                                glm::vec3(1.f, 1.f, 0.f),
                                glm::vec3(0.f, 0.f, 0.f),
                                glm::vec3(0.f, 1.f, 0.f));

    auto frame = cmd_stream.render(scene_handle, cam);
    (void)frame;

    cmd_stream.dropScene(move(scene_handle));
}
