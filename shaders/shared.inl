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
#define MAX_DRAWN_TRIANGLES 1000000000u
#define TRIANGLE_SIZE 12

#define DEFINE_PUSHCONSTANT(STRUCT, NAME) layout(push_constant, scalar) uniform Push { STRUCT NAME; };

#if defined(__cplusplus)
#define SHARED_FUNCTION inline
#else
#define SHARED_FUNCTION
#endif

SHARED_FUNCTION daxa_u32 round_up_to_multiple(daxa_u32 value, daxa_u32 multiple_of)
{
    return ((value + multiple_of - 1) / multiple_of) * multiple_of;
}

SHARED_FUNCTION daxa_u32 round_up_div(daxa_u32 value, daxa_u32 div)
{
    return (value + div - 1) / div;
}

#define ENABLE_TASK_USES(STRUCT, NAME) 

struct DrawIndexedIndirectInfo
{
    daxa_u32 index_count;
    daxa_u32 instance_count;
    daxa_u32 first_index;
    daxa_u32 vertex_offset;
    daxa_u32 first_instance;
};
DAXA_ENABLE_BUFFER_PTR(DrawIndexedIndirectInfo)