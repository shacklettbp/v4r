#include <iostream>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>

#include <v4r.hpp>

#include "dispatch.hpp"

using namespace std;

namespace v4r {

void entry() {
    vector<thread> threads;
    for (int i = 0; i < 16; i++) {
        threads.emplace_back([i]() {
            cout << "blah " << i << endl;
        });
    }

    for (thread &t : threads) {
        t.join();
    }
}

}
