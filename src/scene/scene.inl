#pragma once

#include <daxa/daxa.inl>

#define MAX_ENTITY_COUNT 100000
#define INVALID_ENTITY_INDEX (~(0))

struct EntityId
{
#if defined(DAXA_SHADER)
    daxa_u32 index;
#else
    daxa::types::u32 index = INVALID_ENTITY_INDEX;
#endif
};

#if !defined(DAXA_SHADER)
inline
#endif
bool entity_id_valid(EntityId id)
{
    return id.index != INVALID_ENTITY_INDEX;
}

struct EntityData
{
    daxa_u32 entity_count;
    daxa_f32mat4x4 transform[MAX_ENTITY_COUNT];
    daxa_f32mat4x4 combined_transform[MAX_ENTITY_COUNT];
    EntityId first_child[MAX_ENTITY_COUNT];
    EntityId next_silbing[MAX_ENTITY_COUNT];
    EntityId parent[MAX_ENTITY_COUNT];
    daxa_u32 meshes[8][MAX_ENTITY_COUNT];
    daxa_u32 meshes_count[MAX_ENTITY_COUNT];
};
DAXA_ENABLE_BUFFER_PTR(EntityData)

#if !defined(DAXA_SHADER)
struct EntityRef
{
    daxa_f32mat4x4* transform;
    EntityId* first_child;
    EntityId* next_silbing;
    EntityId* parent;
    daxa_u32* meshes;
    daxa_u32* meshes_count;
};
#endif
