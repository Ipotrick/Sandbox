#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_list.hpp>
#include "../gpu_context.hpp"

template <typename T_USES_BASE, char const *T_FILE_PATH>
struct WriteIndirectDispatchArgsBaseTask : T_USES_BASE
{
    static inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{T_FILE_PATH},
            .compile_options = {
                .defines = {{std::string(T_USES_BASE::NAME), "1"}},
            },
        },
        .name = std::string{T_USES_BASE::NAME},
    };
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_constant_buffer(context->shader_globals_set_info);
        cmd.set_constant_buffer(ti.uses.constant_buffer_set_info());
        cmd.set_pipeline(*context->compute_pipelines.at(T_USES_BASE::NAME));
        cmd.dispatch(1, 1, 1);
    }
};

template <typename T_USES_BASE, char const *T_FILE_PATH, typename T_PUSH>
struct WriteIndirectDispatchArgsPushBaseTask : T_USES_BASE
{
    static inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{T_FILE_PATH},
            .compile_options = {
                .defines = {{std::string(T_USES_BASE::NAME), "1"}},
            },
        },
        .name = std::string{T_USES_BASE::NAME},
    };
    GPUContext * context = {};
    T_PUSH push = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_constant_buffer(context->shader_globals_set_info);
        cmd.set_constant_buffer(ti.uses.constant_buffer_set_info());
        cmd.set_pipeline(*context->compute_pipelines.at(T_USES_BASE::NAME));
        cmd.push_constant(push);
        cmd.dispatch(1, 1, 1);
    }
};