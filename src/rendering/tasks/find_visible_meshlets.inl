#pragma once

#include <daxa/daxa.inl>

#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

struct FindVisibleMeshletsPush
{
    daxa_BufferPtr(daxa_u32) prefix_sum_mehslet_counts;
    daxa_BufferPtr(EntityData) entities;
    daxa_BufferPtr(Mesh) meshes;
    daxa_RWBufferPtr(MeshletDrawInfo) instanciated_meshlets;
};
DAXA_ENABLE_BUFFER_PTR(FindVisibleMeshletsPush)