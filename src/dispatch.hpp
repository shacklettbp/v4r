#ifndef DISPATCH_HPP_INCLUDED
#define DISPATCH_HPP_INCLUDED

#define VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan.h>
#undef VK_ENABLE_BETA_EXTENSIONS

namespace v4r {

struct InstanceDispatch {
#include "dispatch_instance_impl.hpp"

    InstanceDispatch(VkInstance inst, bool need_present);
};

struct DeviceDispatch {
#include "dispatch_device_impl.hpp"

    DeviceDispatch(VkDevice dev, bool need_present);
};

}

#endif
