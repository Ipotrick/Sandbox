#pragma once

#include <daxa/daxa.inl>

#include "../../scene/scene.inl"

struct CullEntitiesPush
{
    daxa_BufferPtr(EntityData) entities;
    daxa_BufferPtr(Mesh) meshes;
    daxa_RWBufferPtr(daxa_u32) culled_entitiy_count;
    daxa_RWBufferPtr(EntityId) culled_entity_ids;
    daxa_RWBufferPtr(daxa_u32) culled_entitiy_meshlet_counts;
};