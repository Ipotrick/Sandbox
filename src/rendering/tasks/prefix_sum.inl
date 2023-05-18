#pragma once

#include <daxa/daxa.inl>

#include "../../mesh/mesh.inl"
#include "../../scene/scene.inl"
#include "misc.hpp"

#define PREFIX_SUM_BLOCK_SIZE 1024
#define PREFIX_SUM_WORKGROUP_SIZE PREFIX_SUM_BLOCK_SIZE

struct DispatchIndirectValueCount
{
    DispatchIndirectStruct command;
    daxa_u32 value_count;
};

#if __cplusplus || defined(PrefixSumWriteCommandBase)
DAXA_INL_TASK_USE_BEGIN(PrefixSumWriteCommandBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(u_value_count, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_upsweep_command0, daxa_RWBufferPtr(DispatchIndirectValueCount), COMPUTE_SHADER_WRITE)
DAXA_INL_TASK_USE_BUFFER(u_upsweep_command1, daxa_RWBufferPtr(DispatchIndirectValueCount), COMPUTE_SHADER_WRITE)
DAXA_INL_TASK_USE_BUFFER(u_downsweep_command, daxa_RWBufferPtr(DispatchIndirectValueCount), COMPUTE_SHADER_WRITE)
DAXA_INL_TASK_USE_END()
#endif
#if __cplusplus || defined(UPSWEEP)
DAXA_INL_TASK_USE_BEGIN(PrefixSumBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(u_command, daxa_BufferPtr(DispatchIndirectValueCount), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_src, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_dst, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_END()
#endif
#if __cplusplus || defined(DOWNSWEEP)
DAXA_INL_TASK_USE_BEGIN(PrefixSumDownsweepBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(u_command, daxa_BufferPtr(DispatchIndirectValueCount), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_block_sums, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_values, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_WRITE)
DAXA_INL_TASK_USE_END()
#endif

struct PrefixSumWriteCommandPush
{
    daxa_u32 uint_offset;
};

struct PrefixSumPush
{
    daxa_u32 uint_offset;
    daxa_u32 uint_stride;
};

#if __cplusplus

#include "../gpu_context.hpp"

static constexpr inline char const PREFIX_SUM_SHADER_PATH[] = "./src/rendering/tasks/prefix_sum.glsl";

using PrefixSumCommandWrite = WriteIndirectDispatchArgsPushBaseTask<
    PrefixSumWriteCommandBase,
    PREFIX_SUM_SHADER_PATH,
    PrefixSumWriteCommandPush
>;

struct PrefixSumUpsweep : PrefixSumBase
{
    static const inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{PREFIX_SUM_SHADER_PATH},
            .compile_options = {
                .defines = {{"UPSWEEP", "1"}},
            },
        },
        .push_constant_size = sizeof(PrefixSumPush),
        .name = std::string{PrefixSumBase::NAME},
    };
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext * context = {};
    PrefixSumPush push = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_constant_buffer(context->shader_globals_set_info);
        cmd.set_constant_buffer(ti.uses.constant_buffer_set_info());
        cmd.set_pipeline(*context->compute_pipelines.at(PrefixSumBase::NAME));
        cmd.push_constant(push);
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_command.buffer(),
        });
    }
};
struct PrefixSumDownsweep : PrefixSumDownsweepBase
{
    static const inline daxa::ComputePipelineCompileInfo PIPELINE_COMPILE_INFO = {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{PREFIX_SUM_SHADER_PATH},
            .compile_options = {
                .defines = {{"DOWNSWEEP", "1"}},
            },
        },
        .push_constant_size = sizeof(PrefixSumPush),
        .name = std::string{PrefixSumDownsweepBase::NAME},
    };
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext * context = {};
    PrefixSumPush push = {};
    void callback(daxa::TaskInterface ti)
    {
        auto cmd = ti.get_command_list();
        cmd.set_constant_buffer(context->shader_globals_set_info);
        cmd.set_constant_buffer(ti.uses.constant_buffer_set_info());
        cmd.set_pipeline(*context->compute_pipelines.at(PrefixSumDownsweepBase::NAME));
        cmd.push_constant(push);
        cmd.dispatch_indirect({
            .indirect_buffer = uses.u_command.buffer(),
        });
    }
};

struct PrefixSumTaskGroupInfo
{
    GPUContext * context;
    daxa::TaskList& task_list;
    daxa::TaskBufferHandle value_count;
    u32 value_count_uint_offset;
    daxa::TaskBufferHandle values;
    u32 src_uint_offset;
    u32 src_uint_stride;
};
void task_prefix_sum(PrefixSumTaskGroupInfo info)
{
    auto upsweep0_command_buffer = info.task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct),
        .name = "prefix sum upsweep0_command_buffer",
    });
    auto upsweep1_command_buffer = info.task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct),
        .name = "prefix sum upsweep1_command_buffer",
    });
    auto downsweep_command_buffer = info.task_list.create_transient_buffer({
        .size = sizeof(DispatchIndirectStruct),
        .name = "prefix sum downsweep_command_buffer",
    });
    info.task_list.add_task(PrefixSumCommandWrite{
        {.uses={
            .u_value_count = info.value_count,
            .u_upsweep_command0 = upsweep0_command_buffer,
            .u_upsweep_command1 = upsweep1_command_buffer,
            .u_downsweep_command = downsweep_command_buffer,
        }},
        .context = info.context,
        .push = {info.value_count_uint_offset}
    });
    auto intermediate_buffer = info.task_list.create_transient_buffer({
        .size = sizeof(u32) * MAX_DRAWN_MESHES,
        .name = "prefix sum intermediate_buffer",
    });
    info.task_list.add_task(PrefixSumUpsweep{
        {.uses={
            .u_command = upsweep0_command_buffer,
            .u_src = info.values,
            .u_dst = intermediate_buffer,
        }},
        .context = info.context,
        .push = {
            .uint_offset = info.src_uint_offset,
            .uint_stride = info.src_uint_stride,
        },
    });
    static constexpr u32 BLOCK_COUNT = (MAX_DRAWN_MESHES + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE;
    auto intermediate_block_sum_buffer = info.task_list.create_transient_buffer({
        .size = sizeof(u32) * BLOCK_COUNT,
        .name = "prefix sum intermediate_block_sum_buffer",
    });
    info.task_list.add_task(PrefixSumUpsweep{
        {.uses={
            .u_command = upsweep1_command_buffer,
            .u_src = intermediate_buffer,
            .u_dst = intermediate_block_sum_buffer,
        }},
        .context = info.context,
        .push = {
            .uint_offset = 0,
            .uint_stride = PREFIX_SUM_BLOCK_SIZE,
        },
    });
    info.task_list.add_task(PrefixSumDownsweep{
        {.uses={
            .u_command = downsweep_command_buffer,
            .u_block_sums = intermediate_block_sum_buffer,
            .u_values = info.values,
        }},
        .context = info.context,
        .push = {
            .uint_offset = info.src_uint_offset,
            .uint_stride = info.src_uint_stride,
        },
    });
}

#endif