#include <daxa/daxa.inl>

#include "shared.inl"
#include "util.inl"

#define SHARED_PREFIX_SUM_VALUE_COUNT ( PREFIX_SUM_WORKGROUP_SIZE / WARP_SIZE )
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

#if defined(ENTRY_PREFIX_SUM)
DEFINE_PUSHCONSTANT(PrefixSumPush, push)
layout(local_size_x = PREFIX_SUM_WORKGROUP_SIZE) in;
void main()
{
    const uint global_index = gl_GlobalInvocationID.x;
    const uint warp_id = gl_SubgroupID;
    const uint warp_index = gl_SubgroupInvocationID;
    const uint src_index = global_index * push.src_stride + push.src_offset;

    uint value = 0;
    if (global_index < push.value_count)
    {
        value = deref(push.src[src_index]);
    }
    prefix_sum(
        warp_index,
        warp_id,
        value);
    if (global_index < push.value_count)
    {
        deref(push.dst[global_index]) = value;
    }
}
#endif // #if defined(ENTRY_PREFIX_SUM)

// As the first PREFIX_SUM_WORKGROUP_SIZE values are already correct, 
// this must be dispatched with an offset of PREFIX_SUM_WORKGROUP_SIZE and size of SIZE - PREFIX_SUM_WORKGROUP_SIZE.
void prefix_sum_twoppass_finalize(
    uint global_index,
    uint workgroup_index,
    daxa_BufferPtr(daxa_u32) partial_sums,
    daxa_RWBufferPtr(daxa_u32) values)
{
    uint partial_sum = deref(partial_sums[workgroup_index]);
    const uint index = global_index + PREFIX_SUM_WORKGROUP_SIZE;
    uint value = deref(values[index]);
    deref(values[index]) = value + partial_sum;
}


#if defined(ENTRY_PREFIX_SUM_TWO_PASS_FINALIZE)
DEFINE_PUSHCONSTANT(PrefixSumTwoPassFinalizePush, push)
layout(local_size_x = PREFIX_SUM_WORKGROUP_SIZE) in;
void main()
{
    const uint global_index = gl_GlobalInvocationID.x;
    const uint workgroup_index = gl_WorkGroupID.x;

    prefix_sum_twoppass_finalize(
        global_index,
        workgroup_index,
        push.partial_sums,
        push.values);
}
#endif // #if defined(ENTRY_PREFIX_SUM_TWO_PASS_FINALIZE)