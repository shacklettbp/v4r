#include <iostream>
#include <cstdlib>

#include <v4r/preprocess.hpp>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << argv[0] << " src dst output[s]" << endl;
        exit(EXIT_FAILURE);
    }

    string_view src = argv[1];
    string_view dst = argv[2];
    for (int i = 3; i < argc; i++) {
    }

    v4r::ScenePreprocessor dumper(src);

    dumper.dump(dst);

    return 0;
}
