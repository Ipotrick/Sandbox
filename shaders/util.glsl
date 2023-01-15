#include <daxa/daxa.glsl>

#include "shared.inl"
#include "util.inl"

#if !defined(WARP_SIZE)
#define WARP_SIZE 32
#endif // #if !defined(WARP_SIZE)

#define SHARED_PREFIX_SUM_VALUE_COUNT ( PREFIX_SUM_WORKGROUP_SIZE / WARP_SIZE )
shared uint shared_prefix_sum_values[SHARED_PREFIX_SUM_VALUE_COUNT];
void prefix_sum(
    uint warp_index,
    uint warp_id,
    inout uint value)
{
    uint sum = subgroupInclusiveAdd(value);
    if (warp_index == (WARP_SIZE - 1))
    {
        shared_prefix_sum_values[warp_id] = sum;
    }
    memoryBarrierShared();
    // Only one warp of size 32 is required to prefix sum the shared values, as 32 * 32 = 1024
    if (warp_id == 0)
    {
        uint shared_value = shared_prefix_sum_values[warp_index];
        uint shared_sum = subgroupInclusiveAdd(shared_value);
        shared_prefix_sum_values[warp_index] = shared_sum;
    }
    memoryBarrierShared();
    // Now all threads can use the finished shared values results to add on to their local result.
    if (warp_id == 0)
    {
        // The first warp obviously is already done in the prefix sum.
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

    uint value = 0;
    if (global_index < push.value_count)
    {
        const uint index = global_index * push.src_stride;
        value = deref(push.src[index]);
    }
    prefix_sum(
        warp_index,
        warp_id,
        value);
    if (global_index < push.value_count)
    {
        deref(push.dst) = value;
    }
}
#endif // #if defined(ENTRY_PREFIX_SUM)

// As the first PREFIX_SUM_WORKGROUP_SIZE values are already correct, 
// this must be dispatched with an offset of PREFIX_SUM_WORKGROUP_SIZE and size of SIZE - PREFIX_SUM_WORKGROUP_SIZE.
void prefix_sum_twoppass_finalize(
    uint value_count,
    uint global_index,
    daxa_BufferPtr(daxa_u32) partial_sums,
    daxa_RWBufferPtr(daxa_u32) values)
{
    if (global_index >= value_count)
    {
        return;
    }
    uint partial_sum_index = (global_index / PREFIX_SUM_WORKGROUP_SIZE) - 1;
    uint partial_sum = deref(partial_sums[partial_sum_index]);
    uint value = deref(values[global_index]);
    value += partial_sum;
    deref(values[global_index]) = value;
}


#if defined(ENTRY_PREFIX_SUM_TWO_PASS_FINALIZE)
DEFINE_PUSHCONSTANT(PrefixSumTwoPassFinalizePush, push)
layout(local_size_x = WARP_SIZE) in;
void main()
{
    const uint global_index = gl_GlobalInvocationID.x;

    prefix_sum_twoppass_finalize(
        push.value_count,
        global_index,
        push.partial_sums,
        push.values);
}
#endif // #if defined(ENTRY_PREFIX_SUM_TWO_PASS_FINALIZE)