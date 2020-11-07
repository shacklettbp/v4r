#ifndef VULKAN_CONFIG_HPP_INCLUDED
#define VULKAN_CONFIG_HPP_INCLUDED

#include <vulkan/vulkan.hpp>

namespace v4r {

namespace VulkanConfig {

constexpr float gfx_priority = 1.0;
constexpr float compute_priority = 1.0;
constexpr float transfer_priority = 1.0;
constexpr uint32_t descriptor_pool_size = 10;
constexpr uint32_t minibatch_divisor = 4;

constexpr uint64_t texture_backing_size = 1ul << 29;

static constexpr int num_meshlet_vertices = 64;
static constexpr int num_meshlet_triangles = 126;
static constexpr int num_meshlets_per_chunk = 32;

}

}

#endif
