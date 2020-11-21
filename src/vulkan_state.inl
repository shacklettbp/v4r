#ifndef VULKAN_STATE_INL_INCLUDED
#define VULKAN_STATE_INL_INCLUDED

#include "vulkan_state.hpp"

#include <iostream>

namespace v4r {

template <typename Fn>
uint32_t CommandStreamState::render(const std::vector<Environment> &envs,
                                    Fn &&submit_func)
{
    PerFrameState &frame_state = frame_states_[cur_frame_];

    VkCommandBuffer render_cmd = frame_state.commands[0];

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(render_cmd, &begin_info));

    dev.dt.cmdBindPipeline(render_cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                           pipeline.gfxPipeline);

    dev.dt.cmdBindDescriptorSets(render_cmd,
                                 VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                 pipeline.gfxLayout, 0,
                                 1, &frame_state.frameSet,
                                 0, nullptr);

    ViewInfo *view_ptr = frame_state.viewPtr;
    for (uint32_t batch_idx = 0; batch_idx < envs.size(); batch_idx++) {
        const Environment &env = envs[batch_idx];
        view_ptr->view = env.view_;
        view_ptr->projection = env.state_->projection;
        view_ptr++;
    }

    for (uint32_t batch_idx = 0; batch_idx < envs.size(); batch_idx++) {
        const Environment &env = envs[batch_idx];
        const Scene &scene = *(env.state_->scene);

        glm::u32vec2 batch_offset = frame_state.batchFBOffsets[batch_idx];

        RenderPushConstant render_const {
            batch_idx,
            batch_offset.x,
            batch_offset.y,
        };

        dev.dt.cmdPushConstants(render_cmd, pipeline.gfxLayout,
                                VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                                0,
                                sizeof(RenderPushConstant),
                                &render_const);

        dev.dt.cmdBindDescriptorSets(render_cmd,
                                     VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                     pipeline.gfxLayout, 1,
                                     1, &scene.sceneSet.hdl,
                                     0, nullptr);


        dev.dt.cmdTraceRaysKHR(
            render_cmd,
            &pipeline.raygenEntry,
            &pipeline.missEntry,
            &pipeline.hitEntry,
            &pipeline.callableEntry,
            per_elem_render_size_.x,
            per_elem_render_size_.y,
            1);
    }

    REQ_VK(dev.dt.endCommandBuffer(render_cmd));

    // FIXME 
    per_render_buffer_.flush(dev);

    uint32_t rendered_frame_idx = cur_frame_;

    submit_func(rendered_frame_idx,
                frame_state.commands.size(),
                frame_state.commands.data(),
                frame_state.fence);

    cur_frame_ = (cur_frame_ + 1) % frame_states_.size();

    return rendered_frame_idx;
}

}

#endif
