#include <daxa/daxa.inl>

#include "prefix_sum.inl"

DAXA_DECL_PUSH_CONSTANT(PrefixSumPush, push)

#if defined(PrefixSumWriteCommandBase_COMMAND)
layout(local_size_x = 1) in;
void main()
{
    const uint value_count = deref(u_value_count[push.uint_offset]);
    const uint block_count = (value_count + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;
    DispatchIndirectValueCount command_and_count;
    
    command_and_count.command.x = block_count;
    command_and_count.command.y = 1;
    command_and_count.command.z = 1;
    command_and_count.value_count = value_count;
    deref(u_upsweep_command0) = command_and_count;
     
    command_and_count.command.x = max((block_count + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE, 1) - 1;
    command_and_count.command.y = 1;
    command_and_count.command.z = 1;
    command_and_count.value_count = block_count;
    deref(u_upsweep_command1) = command_and_count;

    command_and_count.command.x = block_count;
    command_and_count.command.y = 1;
    command_and_count.command.z = 1;
    command_and_count.value_count = value_count;
    deref(u_downsweep_command) = command_and_count;
}
#endif

#define SHARED_PREFIX_SUM_VALUE_COUNT ( PREFIX_SUM_BLOCK_SIZE / WARP_SIZE )
shared uint shared_prefix_sum_values[SHARED_PREFIX_SUM_VALUE_COUNT];
void prefix_sum(
    uint warp_index,
    uint warp_id,
    inout uint value)
{
    value = subgroupInclusiveAdd(value);
    if (warp_index == (WARP_SIZE - 1))
    {
        shared_prefix_sum_values[warp_id] = value;
    }
    memoryBarrierShared();
    barrier();
    // Only one warp of size 32 is required to prefix sum the shared values, as 32 * 32 = 1024
    if (warp_id == 0)
    {
        uint shared_value = shared_prefix_sum_values[warp_index];
        uint shared_sum = subgroupInclusiveAdd(shared_value);
        shared_prefix_sum_values[warp_index] = shared_sum;
    }
    memoryBarrierShared();
    barrier();
    // Now all threads can use the finished shared values results to add on to their local result.
    if (warp_id == 0)
    {
        // The first warp is already done in the prefix sum.
        return;
    }
    value += shared_prefix_sum_values[warp_id - 1];
}

#if defined(UPSWEEP)
layout(local_size_x = PREFIX_SUM_BLOCK_SIZE) in;
void main()
{
    const uint global_index = gl_GlobalInvocationID.x;
    const uint warp_id = gl_SubgroupID;
    const uint warp_index = gl_SubgroupInvocationID;
    const uint src_index = global_index * push.uint_stride + push.uint_offset;
    const uint value_count = deref(u_command).value_count;

    uint value = 0;
    if (global_index < value_count)
    {
        value = deref(u_src[src_index]);
    }
    prefix_sum(
        warp_index,
        warp_id,
        value);
    if (global_index < value_count)
    {
        deref(u_dst[global_index]) = value;
    }
}
#endif // #if defined(UPSWEEP)

void prefix_sum_twoppass_finalize(
    uint value_index,
    uint workgroup_index,
    daxa_BufferPtr(daxa_u32) block_sums,
    daxa_RWBufferPtr(daxa_u32) values,
    uint offset,
    uint stride)
{
    const uint block_sum = deref(block_sums[workgroup_index]);
    const uint real_index = offset + stride * value_index;
    deref(values[real_index]) = deref(values[real_index]) + block_sum;
}

#if defined(DOWNSWEEP)
layout(local_size_x = PREFIX_SUM_BLOCK_SIZE) in;
void main()
{
    const uint value_index = gl_GlobalInvocationID.x + PREFIX_SUM_BLOCK_SIZE; // Skip the first block.
    const uint workgroup_index = gl_WorkGroupID.x;
    const uint value_count = deref(u_command).value_count;

    if (value_index >= value_count)
    {
        return;
    }

    prefix_sum_twoppass_finalize(
        value_index,
        workgroup_index,
        u_block_sums,
        u_values,
        push.uint_offset,
        push.uint_stride);
}
#endif // #if defined(DOWNSWEEP)