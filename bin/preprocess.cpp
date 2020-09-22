#include <iostream>
#include <cstdlib>

#include <v4r/preprocess.hpp>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << argv[0] << " src dst" << endl;
        exit(EXIT_FAILURE);
    }

    v4r::ScenePreprocessor dumper(argv[1]);

    dumper.dump(argv[2]);

    return 0;
}
