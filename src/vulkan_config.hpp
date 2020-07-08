#ifndef VULKAN_CONFIG_HPP_INCLUDED
#define VULKAN_CONFIG_HPP_INCLUDED

#include <vulkan/vulkan.hpp>

namespace v4r {

namespace VulkanConfig {

constexpr float gfx_priority = 1.0;
constexpr float compute_priority = 1.0;
constexpr float transfer_priority = 1.0;

constexpr uint32_t descriptor_pool_size = 10;
constexpr uint32_t max_textures = 500;
constexpr uint32_t max_instances = 1000;
constexpr uint32_t max_lights = 500;

}

}

#endif
