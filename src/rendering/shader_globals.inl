#pragma once

#include <daxa/daxa.inl>

#include "meshlet.inl"

DAXA_DECL_BUFFER(
    ShaderGlobals,
    {
        daxa_BufferRef(VisibleMeshlets) visible_meshlets;
    }
);