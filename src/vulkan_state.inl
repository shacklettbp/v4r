#ifndef VULKAN_STATE_INL_INCLUDED
#define VULKAN_STATE_INL_INCLUDED

#include "vulkan_state.hpp"
#include "profiler.hpp"

#include <cstring>
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
                                 1, &frame_state.rtSet,
                                 0, nullptr);

    ViewInfo *view_ptr = frame_state.viewPtr;
    uint32_t *material_ptr = frame_state.materialPtr;
    LightProperties *light_ptr = frame_state.lightPtr;
    for (uint32_t batch_idx = 0; batch_idx < envs.size(); batch_idx++) {
        const Environment &env = envs[batch_idx];
        const Scene &scene = *(envs[batch_idx].state_->scene);

        view_ptr->view = env.view_;
        view_ptr->projection = env.state_->projection;
        view_ptr++;

        for (uint32_t mesh_idx = 0; mesh_idx < scene.numMeshes; mesh_idx++) {
            uint32_t num_instances = env.transforms_[mesh_idx].size();

            if (material_ptr) {
                memcpy(material_ptr, env.materials_[mesh_idx].data(),
                       num_instances * sizeof(uint32_t));

                material_ptr += num_instances;
            }
        }

        if (light_ptr) {
            uint32_t num_lights = env.state_->lights.size();

            memcpy(light_ptr, env.state_->lights.data(),
                   num_lights * sizeof(LightProperties));

            *frame_state.numLightsPtr = num_lights;

            light_ptr += num_lights;
        }
    }

    for (uint32_t batch_idx = 0; batch_idx < envs.size(); batch_idx++) {
        const Environment &env = envs[batch_idx];
        const Scene &scene = *(env.state_->scene);

        glm::u32vec2 batch_offset = frame_state.batchFBOffsets[batch_idx];

        Shader::RTRenderPushConstant render_const {
            batch_idx,
            batch_offset.x,
            batch_offset.y,
        };

        dev.dt.cmdPushConstants(render_cmd, pipeline.gfxLayout,
                                VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                                0,
                                sizeof(Shader::RTRenderPushConstant),
                                &render_const);

        dev.dt.cmdBindDescriptorSets(render_cmd,
                                     VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                     pipeline.gfxLayout, 1,
                                     1, &env.state_->rtState->tlasSet.hdl,
                                     0, nullptr);

        dev.dt.cmdBindDescriptorSets(render_cmd,
                                     VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                     pipeline.gfxLayout, 2,
                                     1, &scene.rtSet->hdl,
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
