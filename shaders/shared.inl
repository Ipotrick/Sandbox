#pragma once

#include <daxa/daxa.inl>

struct ShaderGlobals
{
    daxa_f32mat4x4 camera_view;
    daxa_f32mat4x4 camera_projection;
    daxa_f32mat4x4 camera_view_projection;
};
DAXA_ENABLE_BUFFER_PTR(ShaderGlobals)

#define MAX_DRAWN_MESHLETS 1000000
#define MAX_DRAWN_TRIANGLES 10000000
#define TRIANGLE_SIZE 12

#define DEFINE_PUSHCONSTANT(STRUCT, NAME) layout(push_constant, scalar) uniform Push { STRUCT NAME; };