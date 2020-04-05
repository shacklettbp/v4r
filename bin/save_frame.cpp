#include <v4r.hpp>
#include <v4r/debug.hpp>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace v4r;

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

    RenderDoc rdoc {};

    auto scene_handle = cmd_stream.loadScene(argv[1]);

    rdoc.startFrame();
    auto frame = cmd_stream.render(scene_handle, cam);
    rdoc.endFrame();

    saveFrame("/tmp/out_color.bmp", frame.colorPtr, 256, 256, 4);
    saveFrame("/tmp/out_depth.bmp", frame.depthPtr, 256, 256, 1);

    cmd_stream.dropScene(move(scene_handle));
}
