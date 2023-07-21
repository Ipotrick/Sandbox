#pragma once

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include "../gpu_context.hpp"

template <typename T_USES_BASE, char const *T_FILE_PATH>
struct WriteIndirectDispatchArgsBaseTask
{
    DAXA_USE_TASK_HEADER(T_USES_BASE)
    static inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{T_FILE_PATH},
            .compile_options = {
                .defines = {{std::string(T_USES_BASE::NAME) + std::string("_COMMAND"), "1"}},
            },
        },
        .name = std::string{T_USES_BASE::NAME},
    };
    GPUContext * context = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(T_USES_BASE::NAME));
        cmd.dispatch(1, 1, 1);
    }
};

template <typename T_USES_BASE, char const *T_FILE_PATH, typename T_PUSH>
struct WriteIndirectDispatchArgsPushBaseTask
{
    DAXA_USE_TASK_HEADER(T_USES_BASE)
    static inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{T_FILE_PATH},
            .compile_options = {
                .defines = {{std::string(T_USES_BASE::NAME) + std::string("_COMMAND"), "1"}},
            },
        },
        .push_constant_size = sizeof(T_PUSH),
        .name = std::string{T_USES_BASE::NAME},
    };
    GPUContext * context = {};
    T_PUSH push = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_uniform_buffer(context->shader_globals_set_info);
        cmd.set_uniform_buffer(ti.uses.get_uniform_buffer_info());
        cmd.set_pipeline(*context->compute_pipelines.at(T_USES_BASE::NAME));
        cmd.push_constant(push);
        cmd.dispatch(1, 1, 1);
    }
};

void task_clear_buffer(daxa::TaskGraph & tg, daxa::TaskBufferView buffer, u32 value, i32 range = -1)
{
    tg.add_task({
        .uses = {daxa::task_resource_uses::BufferTransferWrite{buffer}},
        .task = [=](daxa::TaskInterface ti){
            auto cmd = ti.get_command_list();
            cmd.clear_buffer({
                .buffer = ti.uses[buffer].buffer(),
                .offset = 0,
                .size = (range == -1) ? ti.get_device().info_buffer(ti.uses[buffer].buffer()).size : static_cast<u32>(range),
                .clear_value = value,
            });
        },
        .name = std::string("clear task buffer"),
    });
}