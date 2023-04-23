#pragma once
#include <daxa/daxa.inl>

struct DrawOpaqueIdsPush
{
    daxa_u32 dummy;
};

struct DrawOpaqueDrawInfo
{
    daxa_u32 indexCount;
    daxa_u32 instanceCount;
    daxa_u32 firstIndex;
    daxa_i32 vertexOffset;
    daxa_u32 firstInstance;
};

#if DAXA_SHADER

#endif