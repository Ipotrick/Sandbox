#include <daxa/daxa.inl>

#include "cull_meshes.inl"

#if defined(CullMeshesCommandBase)
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
    // TODO: Cull meshes.
    MeshList culled_meshes = deref(u_entity_meshlists[entity_index]);
    if (culled_meshes.count == 0)
    {
        return;
    }
    const uint offset = atomicAdd(deref(u_mesh_draw_list).count.x, culled_meshes.count);
    for (uint mesh_index = 0; mesh_index < culled_meshes.count; ++mesh_index)
    {
        MeshDrawInfo draw_info;
        draw_info.entity_id = entity_id;
        draw_info.mesh_index = mesh_index;
        draw_info.mesh_id = culled_meshes.mesh_ids[mesh_index];
        DispatchIndirectStruct mesh_dispatch_info;
        mesh_dispatch_info.x = deref(u_meshes[draw_info.mesh_id]).meshlet_count;
        mesh_dispatch_info.y = 1;
        mesh_dispatch_info.z = 1;
        const uint out_index = offset + mesh_index;
        deref(u_mesh_draw_list).mesh_dispatch_indirects[out_index] = mesh_dispatch_info;
        deref(u_mesh_draw_list).mesh_infos[out_index] = draw_info;
    }
}
#endif
