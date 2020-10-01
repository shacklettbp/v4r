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

    dev.dt.cmdBindPipeline(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline.meshCullPipeline);

    dev.dt.cmdBindDescriptorSets(render_cmd,
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline.meshCullLayout, 0,
                                 1, &frame_state.cullSet,
                                 0, nullptr);

    dev.dt.cmdBindPipeline(render_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline.gfxPipeline);

    dev.dt.cmdBindDescriptorSets(render_cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline.gfxLayout, 0,
                                 1, &frame_state.frameSet,
                                 0, nullptr);

    // Reset count buffer
    dev.dt.cmdFillBuffer(render_cmd, indirect_draw_buffer_.buffer,
                         frame_state.indirectCountBaseOffset,
                         frame_state.indirectCountTotalBytes, 0);

    VkBufferMemoryBarrier init_barrier;
    init_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    init_barrier.pNext = nullptr;
    init_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    init_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    init_barrier.srcQueueFamilyIndex = VK_NULL_HANDLE;
    init_barrier.dstQueueFamilyIndex = VK_NULL_HANDLE;
    init_barrier.buffer = indirect_draw_buffer_.buffer;
    init_barrier.offset = 0;
    init_barrier.size = VK_WHOLE_SIZE;

    dev.dt.cmdPipelineBarrier(render_cmd,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0,
                              0, nullptr,
                              1, &init_barrier,
                              0, nullptr);

    uint32_t draw_id = 0;
    uint32_t inst_offset = 0;
    glm::mat4x3 *transform_ptr = frame_state.transformPtr;
    uint32_t *material_ptr = frame_state.materialPtr;
    LightProperties *light_ptr = frame_state.lightPtr;
    ViewInfo *view_ptr = frame_state.viewPtr;
    for (uint32_t batch_idx = 0; batch_idx < envs.size(); batch_idx++) {
        const Environment &env = envs[batch_idx];
        const Scene &scene = *(envs[batch_idx].state_->scene);

        view_ptr->view = env.view_;
        view_ptr->projection = env.state_->projection;
        view_ptr++;

        frame_state.drawOffsets[batch_idx] = draw_id;

        for (uint32_t mesh_idx = 0; mesh_idx < scene.numMeshes; mesh_idx++) {
            const MeshInfo &mesh_metadata = scene.meshMetadata[mesh_idx];
            uint32_t num_instances = env.transforms_[mesh_idx].size();

            for (uint32_t inst_idx = 0; inst_idx < num_instances; inst_idx++) {
                for (uint32_t chunk_id = 0;
                     chunk_id < mesh_metadata.numChunks;
                     chunk_id++) {
                    frame_state.drawPtr[draw_id] = DrawInput {
                        inst_idx + inst_offset,
                        chunk_id + mesh_metadata.chunkOffset,
                    };
                    draw_id++;
                }
            }
            inst_offset += num_instances;

            memcpy(transform_ptr, env.transforms_[mesh_idx].data(),
                   sizeof(glm::mat4x3) * num_instances);

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

        frame_state.maxNumDraws[batch_idx] =
            draw_id - frame_state.drawOffsets[batch_idx];
    }

    uint32_t total_draws = draw_id;

    assert(total_draws < VulkanConfig::max_instances);

    VkRenderPassBeginInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.renderPass = render_pass_;
    render_pass_info.framebuffer = fb_.hdl;
    render_pass_info.clearValueCount =
        static_cast<uint32_t>(fb_cfg_.clearValues.size());
    render_pass_info.pClearValues = fb_cfg_.clearValues.data();

    // 1 indirect draw per batch elem
    uint32_t global_batch_offset = 0;
    for (uint32_t mini_batch_idx = 0; mini_batch_idx < num_mini_batches_;
         mini_batch_idx++) {

        // Record culling for this mini batch
        for (uint32_t local_batch_idx = 0; local_batch_idx < mini_batch_size_;
             local_batch_idx++) {
            uint32_t batch_idx = global_batch_offset + local_batch_idx;
            const Environment &env = envs[batch_idx];
            const Scene &scene = *(env.state_->scene);

            dev.dt.cmdBindDescriptorSets(render_cmd,
                                         VK_PIPELINE_BIND_POINT_COMPUTE,
                                         pipeline.meshCullLayout, 1,
                                         1, &scene.cullSet.hdl,
                                         0, nullptr);

            if (scene.materialSet.hdl != VK_NULL_HANDLE) {
                dev.dt.cmdBindDescriptorSets(render_cmd,
                                             VK_PIPELINE_BIND_POINT_GRAPHICS,
                                             pipeline.gfxLayout, 1,
                                             1, &scene.materialSet.hdl,
                                             0, nullptr);
            }

            CullPushConstant cull_const {
                env.state_->frustumBounds,
                batch_idx,
                frame_state.drawOffsets[batch_idx],
                frame_state.maxNumDraws[batch_idx]
            };

            dev.dt.cmdPushConstants(render_cmd, pipeline.meshCullLayout,
                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                    0,
                                    sizeof(CullPushConstant),
                                    &cull_const);
                
            dev.dt.cmdDispatch(render_cmd,
                getWorkgroupSize(frame_state.maxNumDraws[batch_idx]), 1, 1);
        }

        // Cull / render barrier
        VkBufferMemoryBarrier buffer_barrier;
        buffer_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        buffer_barrier.pNext = nullptr;
        buffer_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        buffer_barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        buffer_barrier.srcQueueFamilyIndex = VK_NULL_HANDLE;
        buffer_barrier.dstQueueFamilyIndex = VK_NULL_HANDLE;
        buffer_barrier.buffer = indirect_draw_buffer_.buffer;
        buffer_barrier.offset = 0;
        buffer_barrier.size = VK_WHOLE_SIZE;

        dev.dt.cmdPipelineBarrier(render_cmd,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                  0,
                                  0, nullptr,
                                  1, &buffer_barrier,
                                  0, nullptr);

        // Record rendering for this mini batch
        glm::u32vec2 minibatch_offset =
            frame_state.batchFBOffsets[global_batch_offset];
        render_pass_info.renderArea.offset = {
            static_cast<int32_t>(minibatch_offset.x),
            static_cast<int32_t>(minibatch_offset.y),
        };
        render_pass_info.renderArea.extent = { per_minibatch_render_size_.x,
                                               per_minibatch_render_size_.y };

        dev.dt.cmdBeginRenderPass(render_cmd, &render_pass_info,
                                  VK_SUBPASS_CONTENTS_INLINE);

        for (uint32_t local_batch_idx = 0; local_batch_idx < mini_batch_size_;
             local_batch_idx++) {
            uint32_t batch_idx = global_batch_offset + local_batch_idx;
            const Scene &scene = *(envs[batch_idx].state_->scene);

            glm::u32vec2 batch_offset = frame_state.batchFBOffsets[batch_idx];

            RenderPushConstant render_const {
                batch_idx,
            };

            dev.dt.cmdPushConstants(render_cmd, pipeline.gfxLayout,
                                    VK_SHADER_STAGE_VERTEX_BIT |
                                        VK_SHADER_STAGE_FRAGMENT_BIT,
                                    0,
                                    sizeof(RenderPushConstant),
                                    &render_const);

            VkViewport viewport;
            viewport.x = batch_offset.x;
            viewport.y = batch_offset.y;
            viewport.width = per_elem_render_size_.x;
            viewport.height = per_elem_render_size_.y;
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

            VkDeviceSize indirect_offset = frame_state.indirectBaseOffset +
                frame_state.drawOffsets[batch_idx] *
                sizeof(VkDrawIndexedIndirectCommand);

            VkDeviceSize count_offset = frame_state.indirectCountBaseOffset +
                batch_idx * sizeof(uint32_t);

            dev.dt.cmdDrawIndexedIndirectCountKHR(render_cmd,
                indirect_draw_buffer_.buffer,
                indirect_offset,
                indirect_draw_buffer_.buffer,
                count_offset,
                frame_state.maxNumDraws[batch_idx],
                sizeof(VkDrawIndexedIndirectCommand));
        }
        dev.dt.cmdEndRenderPass(render_cmd);

        global_batch_offset += mini_batch_size_;
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
