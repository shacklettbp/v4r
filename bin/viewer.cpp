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

    RenderContext r({0, 128, 128});

    auto scene_handle { r.loadScene(argv[1]) };

    auto cmd_stream = r.makeCommandStream();
    auto frame = cmd_stream.renderCamera();
    (void)frame;

    r.dropScene(move(scene_handle));
}
