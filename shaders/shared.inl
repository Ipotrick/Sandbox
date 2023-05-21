#pragma once

#include <daxa/daxa.inl>

#define SHADER_GLOBALS_SLOT 0

#define MAX_INSTANTIATED_MESHES 100000u
#define MAX_INSTANTIATED_MESHLETS 1000000u
#define VISIBLE_ENTITY_MESHLETS_BITFIELD_SCRATCH 1000000
#define MAX_DRAWN_TRIANGLES 1000000000u
#define MAX_DRAWN_MESHES 100000u
#define TRIANGLE_SIZE 12
#define WARP_SIZE 32

#define ENABLE_MESHLET_CULLING 1
#define ENABLE_TRIANGLE_CULLING 1
#define ENABLE_SHADER_PRINT_DEBUG 1

#if __cplusplus
#define DECL_BUFFER_REF struct
#else
#define DECL_BUFFER_REFDAXA_BUFFER_REFERENCE_LAYOUT buffer
#endif

#if __cplusplus
#define SHADER_ONLY(x) 
#else
#define SHADER_ONLY(x) x
#endif

struct Settings
{
    daxa_u32 enable_mesh_shader;
    daxa_u32 update_culling_information;
#if __cplusplus
    auto operator <=>(Settings const & other) const = default;
#endif
};

struct ShaderGlobals
{
    daxa_f32mat4x4 camera_view;
    daxa_f32mat4x4 camera_projection;
    daxa_f32mat4x4 camera_view_projection;
    daxa_f32mat4x4 cull_camera_view_projection;
    daxa_u32 frame_index;
    daxa_f32 delta_time;
    Settings settings;
};
DAXA_ENABLE_BUFFER_PTR(ShaderGlobals)

DAXA_CONSTANT_BUFFER(SHADER_GLOBALS_SLOT) ShaderGlobalsBlock
{
    ShaderGlobals globals;
};

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

struct DrawIndexedIndirectStruct
{
    daxa_u32 index_count;
    daxa_u32 instance_count;
    daxa_u32 first_index;
    daxa_u32 vertex_offset;
    daxa_u32 first_instance;
};
DAXA_ENABLE_BUFFER_PTR(DrawIndexedIndirectStruct)

struct DrawIndirectStruct
{
    daxa_u32 vertex_count;
    daxa_u32 instance_count;
    daxa_u32 first_vertex;
    daxa_u32 first_instance;
};
DAXA_ENABLE_BUFFER_PTR(DrawIndirectStruct)

struct DispatchIndirectStruct
{
    daxa_u32 x;
    daxa_u32 y;
    daxa_u32 z;
};
DAXA_ENABLE_BUFFER_PTR(DispatchIndirectStruct)

#define BUFFER_COMPUTE_READ(NAME, TYPE) DAXA_INL_TASK_USE_BUFFER(NAME, daxa_BufferPtr(TYPE), COMPUTE_SHADER_READ)
#define BUFFER_COMPUTE_WRITE(NAME, TYPE) DAXA_INL_TASK_USE_BUFFER(NAME, daxa_RWBufferPtr(TYPE), COMPUTE_SHADER_WRITE)