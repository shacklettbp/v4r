#ifndef DESCRIPTORS_HPP_INCLUDED
#define DESCRIPTORS_HPP_INCLUDED

#include <atomic>
#include <list>

#include "vulkan_config.hpp"
#include "vulkan_handles.hpp"
#include "vk_utils.hpp"

namespace v4r {

template<uint32_t BindingNum, VkDescriptorType DescType,
         uint32_t NumDescriptors, VkShaderStageFlags DescStage,
         VkDescriptorBindingFlags BindingFlags = 0>
struct BindingConfig {
    using Num = std::integral_constant<uint32_t, BindingNum>;
    using Type = std::integral_constant<VkDescriptorType, DescType>;
    using Count = std::integral_constant<uint32_t, NumDescriptors>;
    using Stage = std::integral_constant<VkShaderStageFlags, DescStage>;
    using Flags =
        std::integral_constant<VkDescriptorBindingFlags, BindingFlags>;
};

template<typename... Binding>
struct DescriptorLayout {
    static constexpr size_t NumBindings = sizeof...(Binding);

    static VkDescriptorSetLayout makeSetLayout(
        const DeviceState &dev)
    {
        std::array<VkSampler *, NumBindings> layout_args;
        layout_args.fill(nullptr);
        return apply([&](auto ...args) {
            return makeSetLayout(dev, args...);
        }, layout_args);
    }

    template<typename... SamplerType>
    static VkDescriptorSetLayout makeSetLayout(
            const DeviceState &dev,
            const SamplerType... sampler)
    {
        static_assert(sizeof...(Binding) == sizeof...(SamplerType));

        std::array<VkDescriptorSetLayoutBinding, sizeof...(Binding)> bindings
        {{
            {
                Binding::Num::value,
                Binding::Type::value,
                Binding::Count::value,
                Binding::Stage::value,
                sampler
            } ...
        }};

        std::array<VkDescriptorBindingFlags, sizeof...(Binding)> binding_flags
        {{
            Binding::Flags::value
            ...
        }};

        VkDescriptorSetLayoutBindingFlagsCreateInfo flag_info;
        flag_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flag_info.pNext = nullptr;
        flag_info.bindingCount = static_cast<uint32_t>(binding_flags.size());
        flag_info.pBindingFlags = binding_flags.data();

        VkDescriptorSetLayoutCreateInfo info;
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.pNext = &flag_info;
        info.flags = 0;
        info.bindingCount = static_cast<uint32_t>(bindings.size());
        info.pBindings = bindings.data();

        VkDescriptorSetLayout layout;
        REQ_VK(dev.dt.createDescriptorSetLayout(dev.hdl, &info,
                                                nullptr, &layout));

        return layout;
    }

    static VkDescriptorPool makePool(const DeviceState &dev, uint32_t max_sets)
    {
        // Pool sizes describes the max number of descriptors of each type that
        // can be allocated from the pool. Therefore for max_sets descriptor
        // sets, the pool needs max_sets * descriptorCount for each type of
        // descriptor used in the set.
        std::array<VkDescriptorPoolSize, sizeof...(Binding)> pool_sizes {{
            {
                Binding::Type::value,
                max_sets * Binding::Count::value
            } ...
        }};

        VkDescriptorPoolCreateInfo pool_info;
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.pNext = nullptr;
        pool_info.flags = 0;
        pool_info.maxSets = max_sets;
        pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();

        VkDescriptorPool pool;
        REQ_VK(dev.dt.createDescriptorPool(dev.hdl, &pool_info,
                                           nullptr, &pool));

        return pool;
    }
};

struct PoolState {
    PoolState(VkDescriptorPool p)
        : pool(p), numUsed(0), numActive(0)
    {}

    VkDescriptorPool pool;
    uint32_t numUsed;
    std::atomic_uint32_t numActive;
};

struct DescriptorSet {
    DescriptorSet(VkDescriptorSet d, PoolState *p)
        : hdl(d), pool(p)
    {}

    DescriptorSet(const DescriptorSet &) = delete;

    DescriptorSet(DescriptorSet &&o)
        : hdl(o.hdl),
          pool(o.pool)
    {
        o.hdl = VK_NULL_HANDLE;
    }

    ~DescriptorSet()
    {
        if (hdl == VK_NULL_HANDLE) return;
        pool->numActive--;
    };

    VkDescriptorSet hdl;
    PoolState *pool;
};

template<typename... LayoutType>
VkDescriptorSet makeDescriptorSet(const DeviceState &dev,
                                  VkDescriptorPool pool,
                                  LayoutType... layout)
{
    std::array layouts {
        layout
        ...
    };

    VkDescriptorSetAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.descriptorPool = pool;
    alloc.descriptorSetCount = 
        static_cast<uint32_t>(layouts.size());
    alloc.pSetLayouts = layouts.data();

    VkDescriptorSet desc_set;
    REQ_VK(dev.dt.allocateDescriptorSets(dev.hdl, &alloc, &desc_set));

    return desc_set;
}

class DescriptorManager {
public:
    using MakePoolType = std::add_pointer_t<
        VkDescriptorPool(const DeviceState &, uint32_t)>;

    DescriptorManager(const DeviceState &dev,
                      const VkDescriptorSetLayout &layout,
                      MakePoolType make_pool);
    DescriptorManager(const DescriptorManager &) = delete;
    DescriptorManager(DescriptorManager &&) = default;

    ~DescriptorManager();

    DescriptorSet makeSet();

private:
    const DeviceState &dev;
    const VkDescriptorSetLayout &layout_;
    MakePoolType make_pool_;

    std::list<PoolState> free_pools_;
    std::list<PoolState> used_pools_;
};

}

#endif
