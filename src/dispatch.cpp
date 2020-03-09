#include "dispatch.hpp"
#include <iostream>
#include <cstdlib>

namespace v4r {

static inline PFN_vkVoidFunction checkPtr(PFN_vkVoidFunction ptr,
                                   const std::string &name) {
    if (!ptr) {
        std::cerr << name << " failed to load" << std::endl;
        exit(EXIT_FAILURE);
    }

    return ptr;
}

InstanceDispatch::InstanceDispatch(VkInstance ctx)
#include "dispatch_instance_impl.cpp"
{}

DeviceDispatch::DeviceDispatch(VkDevice ctx)
#include "dispatch_device_impl.cpp"
{}

}
