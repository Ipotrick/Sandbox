#include <daxa/daxa.inl>

#define PREFIX_SUM_WORKGROUP_SIZE 1024
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