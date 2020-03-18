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

    RenderContext ctx({0, 128, 128});
    auto cmd_stream = ctx.makeCommandStream();

    auto scene_handle = cmd_stream.loadScene(argv[1]);

    auto frame = cmd_stream.renderCamera(scene_handle);
    (void)frame;

    cmd_stream.dropScene(move(scene_handle));
}
