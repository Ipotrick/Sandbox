#pragma once

#include <daxa/daxa.inl>

DAXA_DECL_BUFFER_STRUCT(
    ShaderGlobals,
    {
        
    }
);

DAXA_DECL_BUFFER(
    ShaderIndexBuffer,
    {
        daxa_u32 indices[];
    }
);