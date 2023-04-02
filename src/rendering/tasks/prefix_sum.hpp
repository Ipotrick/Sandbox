#pragma once

#include "../gpu_context.hpp"
#include "../../../shaders/util.inl"
#include "prefix_sum.inl"

inline static constexpr std::string_view PREFIX_SUM_PIPELINE_NAME = "prefix sum";

inline static const daxa::ComputePipelineCompileInfo PREFIX_SUM_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"util.glsl"},
        .compile_options = {
            .defines = {{"ENTRY_PREFIX_SUM"}},
        },
    },
    .push_constant_size = sizeof(PrefixSumPush),
    .debug_name = std::string{PREFIX_SUM_PIPELINE_NAME},
};

inline void t_prefix_sum(
    GPUContext *context,
    daxa::TaskList &task_list,
    daxa::TaskBufferId src,
    daxa::TaskBufferId dst,
    std::function<std::tuple<u32, u32, u32>()> input_callback)
{
    task_list.add_task({
        .used_buffers = {
            daxa::TaskBufferUse{src, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            daxa::TaskBufferUse{dst, daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE_ONLY},
        },
        .task = [=](daxa::TaskRuntimeInterface const &runtime)
        {
            auto [src_stride, src_offset, value_count] = input_callback();

            daxa::CommandList cmd = runtime.get_command_list();
            cmd.set_pipeline(*context->compute_pipelines.at(PREFIX_SUM_PIPELINE_NAME));
            cmd.push_constant(PrefixSumPush{
                .src = context->device.get_device_address(runtime.get_buffers(src)[0]),
                .dst = context->device.get_device_address(runtime.get_buffers(dst)[0]),
                .src_stride = src_stride,
                .src_offset = src_offset,
                .value_count = value_count,
            });
            cmd.dispatch((value_count + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE, 1, 1);
        },
        .debug_name = std::string{PREFIX_SUM_PIPELINE_NAME},
    });
}

inline static constexpr std::string_view PREFIX_SUM_MESHLETS_PIPELINE_NAME = "prefix sum meshlets";

inline static const daxa::ComputePipelineCompileInfo PREFIX_SUM_MESHLETS_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"./src/rendering/tasks/prefix_sum.glsl"},
    },
    .push_constant_size = sizeof(PrefixSumMeshletCountPush),
    .debug_name = std::string{PREFIX_SUM_MESHLETS_PIPELINE_NAME},
};

inline static constexpr std::string_view PREFIX_SUM_TWO_PASS_FINALIZE_PIPELINE_NAME = "prefix sum two pass finalize";

inline static const daxa::ComputePipelineCompileInfo PREFIX_SUM_TWO_PASS_FINALIZE_PIPELINE_INFO{
    .shader_info = daxa::ShaderCompileInfo{
        .source = daxa::ShaderFile{"util.glsl"},
        .compile_options = {
            .defines = {{"ENTRY_PREFIX_SUM_TWO_PASS_FINALIZE"}},
        },
    },
    .push_constant_size = sizeof(PrefixSumTwoPassFinalizePush),
    .debug_name = std::string{PREFIX_SUM_TWO_PASS_FINALIZE_PIPELINE_NAME},
};

inline void t_prefix_sum_two_pass_finalize(
    GPUContext *context,
    daxa::TaskList &task_list,
    daxa::TaskBufferId partial_sums,
    daxa::TaskBufferId values,
    std::function<u32()> input_callback)
{
    task_list.add_task({
        .used_buffers = {
            daxa::TaskBufferUse{partial_sums, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_ONLY},
            daxa::TaskBufferUse{values, daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE},
        },
        .task = [=](daxa::TaskRuntimeInterface const &runtime)
        {
            const u32 value_count = input_callback();
            daxa::CommandList cmd = runtime.get_command_list();
            cmd.set_pipeline(*context->compute_pipelines.at(PREFIX_SUM_TWO_PASS_FINALIZE_PIPELINE_NAME));
            cmd.push_constant(PrefixSumTwoPassFinalizePush{
                .partial_sums = context->device.get_device_address(runtime.get_buffers(partial_sums)[0]),
                .values = context->device.get_device_address(runtime.get_buffers(values)[0]),
            });
            const u32 workgroups = static_cast<u32>(std::max(0, static_cast<i32>(round_up_div(value_count, PREFIX_SUM_WORKGROUP_SIZE)) - 1));
            const u32 dispatch_x = workgroups * (PREFIX_SUM_WORKGROUP_SIZE / PREFIX_SUM_TWO_PASS_FINALIZE_WORKGROUP_SIZE);
            cmd.dispatch(dispatch_x, 1, 1);
        },
        .debug_name = std::string{PREFIX_SUM_TWO_PASS_FINALIZE_PIPELINE_NAME},
    });
}