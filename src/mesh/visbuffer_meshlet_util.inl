#pragma once

#include <daxa/daxa.inl>
#include "mesh.inl"
#include "../../shaders/shared.inl"

#include "../../shaders/visbuffer.inl"

// Use the y dimension of each dispatch struct as the counter here.
struct InstantiatedMeshlets
{
    DispatchIndirectStruct first_count;
    DispatchIndirectStruct second_count;
    DispatchIndirectStruct total_count;
    InstantiatedMeshlet meshlets[MAX_INSTANTIATED_MESHLETS];
};
DAXA_ENABLE_BUFFER_PTR(InstantiatedMeshlets)

struct TriangleDrawList
{
    DrawIndirectStruct count;
    daxa_u32 triangle_ids[MAX_DRAWN_TRIANGLES];
};
DAXA_ENABLE_BUFFER_PTR(TriangleDrawList)

struct MeshDrawInfo
{
    daxa_u32 entity_id;
    daxa_u32 mesh_index;
};

struct MeshDrawList
{
    DrawIndirectStruct count;
    DrawIndirectStruct draw_tasks_count[MAX_INSTANTIATED_MESHES];
    MeshDrawInfo mesh_draw_infos[MAX_INSTANTIATED_MESHES];
};
DAXA_ENABLE_BUFFER_PTR(MeshDrawList)