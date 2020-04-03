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

    RenderContext ctx({0, 128, 128});
    auto cmd_stream = ctx.makeCommandStream();

    RenderDoc rdoc {};

    auto scene_handle = cmd_stream.loadScene(argv[1]);

    Camera cam(60, 1, 0.001, 1000,
               glm::vec3(1.f, 1.f, 0.f),
               glm::vec3(0.f, 1.f, 0.f),
               glm::vec3(0.f, 1.f, 0.f));

    rdoc.startFrame();
    auto frame = cmd_stream.render(scene_handle, cam);
    rdoc.endFrame();

    saveFrame("/tmp/out_color.bmp", frame.colorPtr, 128, 128, 4);
    saveFrame("/tmp/out_depth.bmp", frame.depthPtr, 128, 128, 1);

    cmd_stream.dropScene(move(scene_handle));
}
