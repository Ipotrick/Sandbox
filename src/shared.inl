#pragma once

#include <daxa/daxa.inl >

DAXA_DECL_BUFFER_STRUCT(
    ShaderGlobals,
    {
        daxa_f32mat4x4 camera_view;
        daxa_f32mat4x4 camera_projection;
        daxa_f32mat4x4 camera_view_projection;
    }
)