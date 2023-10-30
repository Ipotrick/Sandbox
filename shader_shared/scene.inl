#pragma once

#include <daxa/daxa.inl>

#define INVALID_ENTITY_INDEX (~(0u))

struct GPUEntityId
{
#if (DAXA_SHADER)
    daxa_u32 value;
#else
    daxa::types::u32 value = INVALID_ENTITY_INDEX;
#endif
};
DAXA_DECL_BUFFER_PTR(GPUEntityId)
SHARED_FUNCTION daxa_u32 version_of_entity_id(GPUEntityId id)
{
    return id.value & 0xFF;
}
SHARED_FUNCTION daxa_u32 index_of_entity_id(GPUEntityId id)
{
    return id.value >> 8;
}
SHARED_FUNCTION bool entity_id_valid(GPUEntityId id)
{
    return version_of_entity_id(id) != 0 && index_of_entity_id(id) != 0;
}

struct EntityMetaData
{
    daxa_u32 entity_count;
};
DAXA_DECL_BUFFER_PTR(EntityMetaData)

#if __cplusplus
struct EntityRef
{
    daxa_f32mat4x4* transform;
    GPUEntityId* first_child;
    GPUEntityId* next_silbing;
    GPUEntityId* parent;
    GPUMeshGroup* meshes;
};
// 0x000001d5edf87b20

#endif
