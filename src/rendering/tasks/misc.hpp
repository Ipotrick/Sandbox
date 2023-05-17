#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_list.hpp>
#include "../gpu_context.hpp"

template <typename T_USES_BASE, char const *T_FILE_PATH>
struct WriteIndirectDispatchArgsBaseTask : T_USES_BASE
{
    GPUContext * context = {};
    daxa::ComputePipelineCompileInfo pipe_info = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{T_FILE_PATH},
            .compile_options = {
                .defines = {{T_USES_BASE::NAME}},
            },
        },
        .name = std::string{T_USES_BASE::NAME},
    };
    std::shared_ptr<daxa::ComputePipeline> pipeline = context->pipeline_manager.add_compute_pipeline(pipe_info).value();
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_constant_buffer(context->shader_globals_set_info);
        cmd.set_constant_buffer(ti.uses.constant_buffer_set_info());
        cmd.dispatch(1, 1, 1);
    }
};