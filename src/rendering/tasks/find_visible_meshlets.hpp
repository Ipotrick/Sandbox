#pragma once

#include "../gpu_context.hpp"
#include "../../../shaders/util.inl"
#include "find_visible_meshlets.inl"

inline static constexpr std::string_view FIND_VISIBLE_MESHLETS_PIPELINE_NAME = "find visible meshlets";

inline static const daxa::ComputePipelineCompileInfo FIND_VISIBLE_MESHLETS_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"./src/rendering/tasks/find_visible_meshlets.glsl"},
        .compile_options = {
            .defines = {{"d"}},
        },
    },
    .push_constant_size = sizeof(FindVisibleMeshletsPush),
    .name = std::string{FIND_VISIBLE_MESHLETS_PIPELINE_NAME},
};

inline void t_find_visible_meshlets(
    GPUContext *context,
    daxa::TaskList &task_list,
    daxa::TaskBufferHandle prefix_sum_mehslet_counts,
    daxa::TaskBufferHandle entity_meta_data,
    daxa::TaskBufferHandle entity_meshlists,
    daxa::TaskBufferHandle meshes,
    daxa::TaskBufferHandle instanciated_meshlets,
    std::function<u32()> input_callback)
{
    using enum daxa::TaskBufferAccess;
    task_list.add_task({
        .uses = {
            daxa::BufferShaderRead{prefix_sum_mehslet_counts},
            daxa::BufferShaderRead{entity_meta_data},
            daxa::BufferShaderRead{entity_meshlists},
            daxa::BufferShaderRead{meshes},
            daxa::BufferShaderWrite{instanciated_meshlets},
        },
        .task = [=](daxa::TaskInterface ti)
        {
            auto value_count = input_callback();
            daxa::CommandList cmd = ti.get_command_list();
            cmd.set_pipeline(*context->compute_pipelines.at(FIND_VISIBLE_MESHLETS_PIPELINE_NAME));
            cmd.push_constant(FindVisibleMeshletsPush{
                .prefix_sum_mehslet_counts = context->device.get_device_address(ti.uses[prefix_sum_mehslet_counts].buffer()),
                .entity_meta_data = context->device.get_device_address(ti.uses[entity_meta_data].buffer()),
                .entity_meshlists = context->device.get_device_address(ti.uses[entity_meshlists].buffer()),
                .meshes = context->device.get_device_address(ti.uses[meshes].buffer()),
                .instanciated_meshlets = context->device.get_device_address(ti.uses[instanciated_meshlets].buffer()),
                .meshlet_count = value_count,
            });
            cmd.dispatch(round_up_div(value_count, FIND_VISIBLE_MESHLETS_WORKGROUP_X), 1, 1);
        },
        .name = std::string{FIND_VISIBLE_MESHLETS_PIPELINE_NAME},
    });
}