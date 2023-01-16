#pragma once

#include <daxa/daxa.inl>

#define PREFIX_SUM_WORKGROUP_SIZE 1024

#if !defined(WARP_SIZE)
#define WARP_SIZE 32
#endif // #if !defined(WARP_SIZE)

#define PREFIX_SUM_TWO_PASS_FINALIZE_WORKGROUP_SIZE WARP_SIZE

struct PrefixSumPush
{
    daxa_BufferPtr(daxa_u32) src;
    daxa_RWBufferPtr(daxa_u32) dst;
    daxa_u32 src_stride;
    daxa_u32 src_offset;
    daxa_u32 value_count;
};

struct PrefixSumTwoPassFinalizePush
{
    daxa_BufferPtr(daxa_u32) partial_sums;
    daxa_RWBufferPtr(daxa_u32) values;
};