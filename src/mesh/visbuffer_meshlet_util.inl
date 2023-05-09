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

struct TriangleDrawInfo
{
    daxa_u32 meshlet_index;
    daxa_u32 triangle_index;
};
DAXA_ENABLE_BUFFER_PTR(TriangleDrawInfo)

struct TriangleDrawList
{
    DrawIndirectStruct count;
    TriangleDrawInfo triangles[MAX_DRAWN_TRIANGLES];
};
DAXA_ENABLE_BUFFER_PTR(TriangleDrawList)