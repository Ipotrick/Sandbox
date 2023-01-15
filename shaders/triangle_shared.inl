#pragma once

#include <daxa/daxa.inl>
#include "shared.inl"

struct TriangleTaskPushConstant
{
    daxa_BufferPtr(ShaderGlobals) globals;
};