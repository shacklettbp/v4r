#ifndef DISPATCH_HPP_INCLUDED
#define DISPATCH_HPP_INCLUDED

#include <vulkan/vulkan.h>

namespace v4r {

struct InstanceDispatch {
#include "dispatch_instance_impl.hpp"
    
    InstanceDispatch(VkInstance inst);
};

struct DeviceDispatch {
#include "dispatch_device_impl.hpp"

    DeviceDispatch(VkDevice dev);
};

}

#endif
