#pragma once

#include <daxa/daxa.inl>

#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

#define FIND_VISIBLE_MESHLETS_WORKGROUP_X 96
struct FindVisibleMeshletsPush
{
    daxa_BufferPtr(daxa_u32) prefix_sum_mehslet_counts;
    daxa_BufferPtr(EntityMetaData) entity_meta_data;
    daxa_BufferPtr(MeshList) entity_meshlists;
    daxa_BufferPtr(Mesh) meshes;
    daxa_RWBufferPtr(InstanciatedMeshlet) instanciated_meshlets;
    daxa_u32 meshlet_count;
};
DAXA_ENABLE_BUFFER_PTR(FindVisibleMeshletsPush)