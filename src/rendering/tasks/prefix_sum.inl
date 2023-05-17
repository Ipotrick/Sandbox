#pragma once

#include <daxa/daxa.inl>

#include "../../../shaders/util.inl"
#include "../../mesh/mesh.inl"
#include "../../scene/scene.inl"
#include "misc.hpp"

#if __cplusplus || defined(PREFIX_SUM_BASE)
DAXA_INL_TASK_USE_BEGIN(PrefixSumBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(u_count_buffer, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_WRITE)
DAXA_INL_TASK_USE_BUFFER(u_src, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_dst, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_END()
#endif
#if __cplusplus || defined(PREFIX_SUM_COLLECT)
DAXA_INL_TASK_USE_BEGIN(PrefixSumCollectBase, DAXA_CBUFFER_SLOT1)
DAXA_INL_TASK_USE_BUFFER(u_count_buffer, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_WRITE)
DAXA_INL_TASK_USE_BUFFER(u_block_sums, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_sums, daxa_BufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_BUFFER(u_dst, daxa_RWBufferPtr(daxa_u32), COMPUTE_SHADER_READ)
DAXA_INL_TASK_USE_END()
#endif

struct PrefixSumPush
{
    daxa_u32 stride;
    daxa_u32 start_offset;
};

#if __cplusplus

#include "../gpu_context.hpp"

static constexpr inline char const PREFIX_SUM_SHADER_PATH[] = "./src/rendering/tasks/prefix_sum.glsl";

struct PrefixSum : PrefixSumBase
{
    std::shared_ptr<daxa::ComputePipeline> pipeline = {};
    GPUContext * context = {};
    u32 dispatch_x = {};
    void callback(daxa::TaskInterface ti)
    {

    }
};

void task_prefix_sum(
    daxa::TaskList& task_list,
    GPUContext * context,
    u32 count)
{
    u32 const block_count = (count + PREFIX_SUM_WORKGROUP_SIZE - 1) / PREFIX_SUM_WORKGROUP_SIZE;
    auto const prefix_sum_intermediate_blocks = task_list.create_transient_buffer({
        .size = block_count * static_cast<u32>(sizeof(u32)),
        .name = "prefix_sum_intermediate_blocks",
    });
    auto const prefix_sum_level_two_block = task_list.create_transient_buffer({
        .size = PREFIX_SUM_WORKGROUP_SIZE * sizeof(u32),
        .name = "prefix_sum_intermediate_blocks",
    });
    task_list.add_task(PrefixSum{
        {.uses={
            .u_command = prefix_sum_command,
            .u_count_buffer = 
        }},
        context,
    });
    task_list.add_task(PrefixSum{
        {.uses={
            .u_command = prefix_sum_command,
            .u_count_buffer = 
        }},
        .context = context,
        .dispatch_x = 
    });


}