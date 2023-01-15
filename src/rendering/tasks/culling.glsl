#include <daxa/daxa.glsl>

#include "../../util.glsl"

#include "culling.inl"

bool cull_entity()
{
    return true;
}

#if defined(ENTRY_CULL_ENTITIES)
DEFINE_PUSHCONSTANT(CullEntitiesPush, push)
layout(local_size_x = 32) in;
void main()
{
    uint entity_index = gl_GlobalInvocationID.x;
    if (entity_index >= deref(push.entities).entity_count)
    {
        return;
    }
    uint out_index = atomicAdd(deref(push.culled_entitiy_count), 1);
    deref(push.culled_entities[out_index]).index = entity_index;
}
#endif
