#pragma once

#include <daxa/daxa.inl>

#define INVALID_MESHLET_INDEX (~(0u))

// Can never be greater then 128!
#define MAX_TRIANGLES_PER_MESHLET 124

// Can never be greater then 384!
#define MAX_VERTICES_PER_MESHLET (128)

#define ENTITY_MESHLET_VISIBILITY_ARENA_SIZE (1<<20)
#define ENTITY_MESHLET_VISIBILITY_ARENA_BIT_SIZE (ENTITY_MESHLET_VISIBILITY_ARENA_SIZE * 8)
#define ENTITY_MESHLET_VISIBILITY_ARENA_UINT_SIZE (ENTITY_MESHLET_VISIBILITY_ARENA_SIZE / 4)

// Used to tell threads in the meshlet cull dispatch what to work on.
struct MeshletCullIndirectArg
{
    daxa_u32 entity_id;
    daxa_u32 mesh_id;
    daxa_u32 entity_meshlist_index;
    daxa_u32 meshlet_index_start_offset;
};
DAXA_DECL_BUFFER_PTR(MeshletCullIndirectArg)

// Table is set up in write command of cull_meshes.glsl.
struct MeshletCullIndirectArgTable
{
    daxa_RWBufferPtr(MeshletCullIndirectArg) indirect_arg_ptrs[32];
    daxa_u32 indirect_arg_counts[32];
};
DAXA_DECL_BUFFER_PTR(MeshletCullIndirectArgTable)

// !!NEEDS TO BE ABI COMPATIBLE WITH meshopt_Meshlet!!
struct Meshlet
{
    // Offset into the meshs vertex index array.
    daxa_u32 indirect_vertex_offset;
    // Equivalent to meshoptimizers triangle_offset.
    // Renamed the field for more clarity.
    daxa_u32 micro_indices_offset;
    daxa_u32 vertex_count;
    daxa_u32 triangle_count;
};
DAXA_DECL_BUFFER_PTR(Meshlet)

struct MeshletInstance
{
    daxa_u32 entity_index;
    daxa_u32 mesh_id;
    daxa_u32 mesh_index;
    daxa_u32 entity_meshlist_index;
};
DAXA_DECL_BUFFER_PTR(MeshletInstance)

struct BoundingSphere{
    daxa_f32vec3 center;
    daxa_f32 radius;
};
DAXA_DECL_BUFFER_PTR(BoundingSphere)

#if 0
#define DEBUG_VERTEX_ID 1
void encode_vertex_id(daxa_u32 instantiated_meshlet_index, daxa_u32 triangle_index, daxa_u32 triangle_corner, out daxa_u32 vertex_id)
{
    #if DEBUG_VERTEX_ID
    vertex_id = instantiated_meshlet_index * 10000 + triangle_index * 10 + triangle_corner;
    #else
    vertex_id = (instantiated_meshlet_index << 9) | (triangle_index << 2) | triangle_corner;
    #endif
}
void decode_vertex_id(daxa_u32 vertex_id, out daxa_u32 instantiated_meshlet_index, out daxa_u32 triangle_index, out daxa_u32 triangle_corner)
{
    #if DEBUG_VERTEX_ID
    instantiated_meshlet_index = vertex_id / 10000;
    triangle_index = (vertex_id / 10) % 1000;
    triangle_corner = vertex_id % 10;
    #else
    instantiated_meshlet_index = vertex_id >> 9;
    triangle_index = (vertex_id >> 2) & 0x3F;
    triangle_corner = vertex_id & 0x3f;
    #endif
}
#endif // #if defined(DAXA_SHADER)

struct Mesh
{
    daxa_BufferId mesh_buffer;
    daxa_u32 meshlet_count;
    daxa_u32 vertex_count;
    daxa_u64 end_ptr;
    daxa_BufferPtr(Meshlet) meshlets;
    daxa_BufferPtr(BoundingSphere) meshlet_bounds;
    daxa_BufferPtr(daxa_u32) micro_indices;
    daxa_BufferPtr(daxa_u32) indirect_vertices;
    daxa_BufferPtr(daxa_f32vec3) vertex_positions;
};
DAXA_DECL_BUFFER_PTR(Mesh)

#if defined(DAXA_SHADER)
uint get_micro_index(daxa_BufferPtr(daxa_u32) micro_indices, daxa_u32 index_offset)
{
    uint pack_index = index_offset / 4;
    uint index_pack = deref(micro_indices[pack_index]);
    uint in_pack_offset = index_offset % 4;
    uint in_pack_shift = in_pack_offset * 8;
    return (index_pack >> in_pack_shift) & 0xFF;
}
#endif // #if defined(DAXA_SHADER)

struct MeshList
{
    daxa_u32 mesh_ids[7];
    daxa_u32 count;
};
DAXA_DECL_BUFFER_PTR(MeshList)

struct InstantiatedMeshlets
{
    daxa_u32 first_count;
    daxa_u32 second_count;
    MeshletInstance meshlets[MAX_INSTANTIATED_MESHLETS];
};
DAXA_DECL_BUFFER_PTR(InstantiatedMeshlets)

struct VisibleMeshletList
{
    daxa_u32 count;
    daxa_u32 meshlet_ids[MAX_INSTANTIATED_MESHLETS];
};
DAXA_DECL_BUFFER_PTR(VisibleMeshletList)

struct EntityMeshletVisibilityBitfieldOffsets
{
    daxa_u32 mesh_bitfield_offset[7];
    daxa_u32 padd;
};
DAXA_DECL_BUFFER_PTR(EntityMeshletVisibilityBitfieldOffsets)

#if !defined(__cplusplus)
DAXA_BUFFER_REFERENCE_LAYOUT buffer EntityMeshletVisibilityBitfieldOffsetsView
{
    daxa_u32 back_offset;
    EntityMeshletVisibilityBitfieldOffsets entity_offsets[];
};
#endif