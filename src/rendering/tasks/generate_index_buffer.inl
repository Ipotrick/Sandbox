#pragma once

#include <daxa/daxa.inl>

#include "../../../shaders/util.inl"
#include "../../../shaders/shared.inl"
#include "../../scene/scene.inl"
#include "../../mesh/mesh.inl"

#define GENERATE_INDEX_BUFFER_WORKGROUP_X PREFIX_SUM_WORKGROUP_SIZE
struct GenerateIndexBufferPush
{
    daxa_BufferPtr(Mesh) meshes;
    daxa_BufferPtr(InstanciatedMeshlet) instanciated_meshlets;
    daxa_RWBufferPtr(daxa_u32) global_triangle_count;
    daxa_RWBufferPtr(daxa_u32) index_buffer;
    daxa_u32 instanciated_meshlet_count;
};
DAXA_ENABLE_BUFFER_PTR(GenerateIndexBufferPush)