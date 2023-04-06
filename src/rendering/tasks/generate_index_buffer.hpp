#pragma once

#include "../gpu_context.hpp"
#include "../../../shaders/util.inl"
#include "generate_index_buffer.inl"

inline static constexpr std::string_view GENERATE_INDEX_BUFFER_PIPELINE_NAME = "generate_index_buffer";

inline static const daxa::ComputePipelineCompileInfo GENERATE_INDEX_BUFFER_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"./src/rendering/tasks/generate_index_buffer.inl"},
        .compile_options = {
            .defines = {{"d"}},
        },
    },
    .push_constant_size = sizeof(GenerateIndexBufferPush),
    .name = std::string{GENERATE_INDEX_BUFFER_PIPELINE_NAME},
};

inline void t_generate_index_buffer(
    GPUContext *context,
    daxa::TaskList &task_list,
    daxa::TaskBufferId meshes,
    daxa::TaskBufferId instanciated_meshlets,
    daxa::TaskBufferId index_buffer,
    std::function<u32()> input_callback)
{
    task_list.add_task({
        .used_buffers = {
            daxa::TaskBufferUse{meshes, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            daxa::TaskBufferUse{instanciated_meshlets, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            daxa::TaskBufferUse{index_buffer, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [=](daxa::TaskRuntimeInterface const &runtime)
        {
            auto value_count = input_callback();
            daxa::CommandList cmd = runtime.get_command_list();
            cmd.set_pipeline(*context->compute_pipelines.at(GENERATE_INDEX_BUFFER_PIPELINE_NAME));
            cmd.push_constant(GenerateIndexBufferPush{
                .meshes = context->device.get_device_address(runtime.get_buffers(meshes)[0]),
                .instanciated_meshlets = context->device.get_device_address(runtime.get_buffers(instanciated_meshlets)[0]),
                .global_triangle_count = context->device.get_device_address(runtime.get_buffers(index_buffer)[0]) + 0,
                .index_buffer = context->device.get_device_address(runtime.get_buffers(index_buffer)[0]) + 16,
            });
            cmd.dispatch(round_up_div(value_count * MAX_TRIANGLES_PER_MESHLET, GENERATE_INDEX_BUFFER_WORKGROUP_X), 1, 1);
        },
        .name = std::string{GENERATE_INDEX_BUFFER_PIPELINE_NAME},
    });
}