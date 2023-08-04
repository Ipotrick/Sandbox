#pragma once

#include <daxa/daxa.inl>

#define INVALID_ENTITY_INDEX (~(0))

struct EntityId
{
#if (DAXA_SHADER)
    daxa_u32 index;
#else
    daxa::types::u32 index = INVALID_ENTITY_INDEX;
#endif
};
DAXA_DECL_BUFFER_PTR(EntityId)


#if !(DAXA_SHADER)
inline
#endif
bool entity_id_valid(EntityId id)
{
    return id.index != INVALID_ENTITY_INDEX;
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
    EntityId* first_child;
    EntityId* next_silbing;
    EntityId* parent;
    MeshList* meshes;
};
// 0x000001d5edf87b20

#endif
