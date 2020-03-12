#ifndef DISPATCH_HPP_INCLUDED
#define DISPATCH_HPP_INCLUDED

#include <vulkan/vulkan.h>

namespace v4r {

struct InstanceDispatch {
#include "dispatch_instance_impl.hpp"
    
    InstanceDispatch(VkInstance inst);

    InstanceDispatch(const InstanceDispatch &) = delete;
};

struct DeviceDispatch {
#include "dispatch_device_impl.hpp"

    DeviceDispatch(VkDevice dev);

    DeviceDispatch(const DeviceDispatch &) = delete;
};

}

#endif
