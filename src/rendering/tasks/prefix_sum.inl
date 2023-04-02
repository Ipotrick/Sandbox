#pragma once

#include <daxa/daxa.inl>

#include "../../mesh/mesh.inl"
#include "../../scene/scene.inl"

struct PrefixSumMeshletCountPush
{
    daxa_BufferPtr(EntityMetaData) entity_meta_data;
    daxa_BufferPtr(MeshList) entity_meshlists;
    daxa_BufferPtr(Mesh) meshes;
    daxa_RWBufferPtr(daxa_u32) dst;
};