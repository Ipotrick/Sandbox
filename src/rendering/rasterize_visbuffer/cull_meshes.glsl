#include <daxa/daxa.inl>

#include "cull_meshes.inl"

#if defined(WRITE_COMMAND)
layout(local_size_x = 1)
void main()
{
    const uint entity_count = deref(u_entity_meta).entity_count;
    const uint dispatch_x = (entity_count + CULL_MESHES_WORKGROUP_X - 1) / CULL_MESHES_WORKGROUP_X;
    deref(u_command).x = dispatch_x;
    deref(u_command).y = 1;
    deref(u_command).z = 1;
}
#else
layout(local_size_x = CULL_MESHES_WORKGROUP_X)
void main()
{
    const uint entity_index = gl_GlobalInvocationID.x;
    if (entity_index >= deref(u_entity_meta.entity_count))
    {
        return;
    }
    
}
#endif
