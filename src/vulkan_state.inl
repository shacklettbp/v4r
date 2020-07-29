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

    dev.dt.cmdBindDescriptorSets(render_cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline.gfxLayout, 0,
                                 1, &frame_state.frameSet,
                                 0, nullptr);

    // FIXME
    dev.dt.cmdBindPipeline(render_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline.gfxPipeline);

    VkRenderPassBeginInfo render_begin;
    render_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_begin.pNext = nullptr;
    render_begin.renderPass = render_pass_;
    render_begin.framebuffer = fb_.hdl;
    render_begin.renderArea.offset = {
        static_cast<int32_t>(frame_state.baseFBOffset.x), 
        static_cast<int32_t>(frame_state.baseFBOffset.y) 
    };
    render_begin.renderArea.extent = { render_extent_.x, 
                                       render_extent_.y };
    render_begin.clearValueCount =
        static_cast<uint32_t>(fb_cfg_.clearValues.size());
    render_begin.pClearValues = fb_cfg_.clearValues.data();

    dev.dt.cmdBeginRenderPass(render_cmd, &render_begin,
                              VK_SUBPASS_CONTENTS_INLINE);

    uint32_t cur_instance = 0;
    glm::mat4x3 *transform_ptr = frame_state.transformPtr;
    uint32_t *material_ptr = frame_state.materialPtr;
    LightProperties *light_ptr = frame_state.lightPtr;
    ViewInfo *view_ptr = frame_state.viewPtr;
    for (uint32_t batch_idx = 0; batch_idx < envs.size(); batch_idx++) {
        const Environment &env = envs[batch_idx];

        const Scene &scene = *(envs[batch_idx].state_->scene);
        if (scene.materialSet.hdl != VK_NULL_HANDLE) {
            dev.dt.cmdBindDescriptorSets(render_cmd,
                                         VK_PIPELINE_BIND_POINT_GRAPHICS,
                                         pipeline.gfxLayout, 1,
                                         1, &scene.materialSet.hdl,
                                         0, nullptr);
        }

        view_ptr->view = env.view_;
        view_ptr->projection = env.state_->projection;
        view_ptr++;

        RenderPushConstant push_const {
            batch_idx
        };

        dev.dt.cmdPushConstants(render_cmd, pipeline.gfxLayout,
                                VK_SHADER_STAGE_VERTEX_BIT |
                                    VK_SHADER_STAGE_FRAGMENT_BIT,
                                0,
                                sizeof(RenderPushConstant),
                                &push_const);

        glm::u32vec2 batch_offset = frame_state.batchFBOffsets[batch_idx];

        VkViewport viewport;
        viewport.x = batch_offset.x;
        viewport.y = batch_offset.y;
        viewport.width = render_size_.x;
        viewport.height = render_size_.y;
        viewport.minDepth = 0.f;
        viewport.maxDepth = 1.f;

        dev.dt.cmdSetViewport(render_cmd, 0, 1, &viewport);

        frame_state.vertexBuffers[0] = scene.data.buffer;
        dev.dt.cmdBindVertexBuffers(render_cmd, 0,
                                    frame_state.vertexBuffers.size(),
                                    frame_state.vertexBuffers.data(),
                                    frame_state.vertexOffsets.data());
        dev.dt.cmdBindIndexBuffer(render_cmd, scene.data.buffer,
                                  scene.indexOffset, VK_INDEX_TYPE_UINT32);

        for (uint32_t mesh_idx = 0; mesh_idx < scene.meshes.size();
                mesh_idx++) {
            uint32_t num_instances = env.transforms_[mesh_idx].size();
            if (num_instances == 0) continue;

            auto &mesh = scene.meshes[mesh_idx];

            dev.dt.cmdDrawIndexed(render_cmd, mesh.numIndices, num_instances,
                                  mesh.startIndex, mesh.vertexOffset,
                                  cur_instance);

            memcpy(transform_ptr, env.transforms_[mesh_idx].data(),
                   sizeof(glm::mat4x3) * num_instances);

            cur_instance += num_instances;
            transform_ptr += num_instances;

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

    dev.dt.cmdEndRenderPass(render_cmd);

    REQ_VK(dev.dt.endCommandBuffer(render_cmd));

    assert(cur_instance < VulkanConfig::max_instances);

    // FIXME 
    per_render_buffer_.flush(dev);

    submit_func(frame_state.commands.size(),
                frame_state.commands.data(),
                frame_state.semaphore,
                frame_state.fence);

    uint32_t rendered_frame_idx = cur_frame_;
    cur_frame_ = (cur_frame_ + 1) % frame_states_.size();

    return rendered_frame_idx;
}

}

#endif
