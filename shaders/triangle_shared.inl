#pragma once

#include <daxa/daxa.inl>
#include "../src/shared.inl"

struct TriangleTaskPushConstant
{
    daxa_BufferPtr(ShaderGlobals) globals;
};