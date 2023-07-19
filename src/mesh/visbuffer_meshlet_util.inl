#pragma once

#include <daxa/daxa.inl>
#include "mesh.inl"
#include "../../shaders/shared.inl"

#include "../../shaders/visbuffer.inl"

// Use the y dimension of each dispatch struct as the counter here.
struct InstantiatedMeshlets
{
    daxa_u32 count;
    InstantiatedMeshlet meshlets[MAX_INSTANTIATED_MESHLETS];
};
DAXA_DECL_BUFFER_PTR(InstantiatedMeshlets)

struct TriangleList
{
    daxa_u32 count;
    daxa_u32 triangle_ids[MAX_DRAWN_TRIANGLES];
};
DAXA_DECL_BUFFER_PTR(TriangleList)

struct MeshDrawInfo
{
    daxa_u32 entity_id;
    daxa_u32 mesh_index;
    daxa_u32 mesh_id;
    daxa_u32 padd0[1];
};

// Intended to be read by vkCmdDrawMeshTasksIndirectCountEXT.
struct MeshDrawList
{
    daxa_u32 count;
    DispatchIndirectStruct mesh_dispatch_indirects[MAX_INSTANTIATED_MESHES];
    MeshDrawInfo mesh_infos[MAX_INSTANTIATED_MESHES];
};
DAXA_DECL_BUFFER_PTR(MeshDrawList)