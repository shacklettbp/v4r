#include "descriptors.hpp"

#include <cassert>
#include <iostream>

using namespace std;

namespace v4r {

DescriptorManager::DescriptorManager(const DeviceState &d,
                                     const VkDescriptorSetLayout &layout)
    : dev(d), layout_(layout),
      free_pools_(), used_pools_()
{}

DescriptorManager::~DescriptorManager()
{
    for (PoolState &pool_state : free_pools_) {
        dev.dt.destroyDescriptorPool(dev.hdl, pool_state.pool, nullptr);
        assert(pool_state.numActive == 0);
    }

    for (PoolState &pool_state : used_pools_) {
        dev.dt.destroyDescriptorPool(dev.hdl, pool_state.pool, nullptr);
        assert(pool_state->numActive == 0);
    }
}

DescriptorSet DescriptorManager::makeSet()
{
    if (free_pools_.empty()) {
        auto iter = used_pools_.begin();
        while (iter != used_pools_.end()) {
            auto next_iter = next(iter);
            if (iter->numActive == 0) {
                REQ_VK(dev.dt.resetDescriptorPool(dev.hdl, iter->pool, 0));
                free_pools_.splice(free_pools_.end(), used_pools_, iter);
            }
            iter = next_iter;
        }
        if (free_pools_.empty()) {
            free_pools_.emplace_back(
                PerSceneDescriptorLayout::makePool(dev,
                    VulkanConfig::descriptor_pool_size));
        }
    }

    PoolState &cur_pool = free_pools_.front();

    VkDescriptorSet desc_set = makeDescriptorSet(dev, cur_pool.pool, layout_);

    cur_pool.numActive++;

    if (cur_pool.numActive.load() == VulkanConfig::descriptor_pool_size) {
        used_pools_.splice(used_pools_.end(), free_pools_,
                           free_pools_.begin());
    }

    return DescriptorSet(
        desc_set,
        cur_pool
    );
}

}
