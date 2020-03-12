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

    RenderContext r(0);

    auto scene_handle { r.loadScene(argv[1]) };

    r.dropScene(move(scene_handle));
}
