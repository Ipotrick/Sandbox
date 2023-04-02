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
    .debug_name = std::string{FIND_VISIBLE_MESHLETS_PIPELINE_NAME},
};

inline void t_find_visible_meshlets(
    GPUContext *context,
    daxa::TaskList &task_list,
    daxa::TaskBufferId prefix_sum_mehslet_counts,
    daxa::TaskBufferId entity_meta_data,
    daxa::TaskBufferId entity_meshlists,
    daxa::TaskBufferId meshes,
    daxa::TaskBufferId instanciated_meshlets,
    std::function<u32()> input_callback)
{
    using enum daxa::TaskBufferAccess;
    task_list.add_task({
        .used_buffers = {
            daxa::TaskBufferUse{prefix_sum_mehslet_counts, COMPUTE_SHADER_READ_ONLY},
            daxa::TaskBufferUse{entity_meta_data, COMPUTE_SHADER_READ_ONLY},
            daxa::TaskBufferUse{entity_meshlists, COMPUTE_SHADER_READ_ONLY},
            daxa::TaskBufferUse{meshes, COMPUTE_SHADER_READ_ONLY},
            daxa::TaskBufferUse{instanciated_meshlets, COMPUTE_SHADER_WRITE_ONLY},
        },
        .task = [=](daxa::TaskRuntimeInterface const &runtime)
        {
            auto value_count = input_callback();
            daxa::CommandList cmd = runtime.get_command_list();
            cmd.set_pipeline(*context->compute_pipelines.at(FIND_VISIBLE_MESHLETS_PIPELINE_NAME));
            cmd.push_constant(FindVisibleMeshletsPush{
                .prefix_sum_mehslet_counts = context->device.get_device_address(runtime.get_buffers(prefix_sum_mehslet_counts)[0]),
                .entity_meta_data = context->device.get_device_address(runtime.get_buffers(entity_meta_data)[0]),
                .entity_meshlists = context->device.get_device_address(runtime.get_buffers(entity_meshlists)[0]),
                .meshes = context->device.get_device_address(runtime.get_buffers(meshes)[0]),
                .instanciated_meshlets = context->device.get_device_address(runtime.get_buffers(instanciated_meshlets)[0]),
                .meshlet_count = value_count,
            });
            cmd.dispatch(round_up_div(value_count, FIND_VISIBLE_MESHLETS_WORKGROUP_X), 1, 1);
        },
        .debug_name = std::string{FIND_VISIBLE_MESHLETS_PIPELINE_NAME},
    });
}