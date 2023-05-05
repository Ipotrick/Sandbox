#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "allocate_meshlet_visibility_bitfields.inl"

layout(local_size_x = 128) in;
void main()
{
    uint entity_index = gl_GlobalInvocationID.x;
    if (entity_index >= deref(u_entity_meta).entity_count)
    {
        return;
    }
    MeshList mesh_list = deref(u_meshlists[entity_index]);
    EntityVisibilityBitfieldOffsets visibility_lists;
    for (uint mesh_index = 0; mesh_index < mesh_list.count; ++mesh_index)
    {
        const uint mesh_meshlet_count = deref(u_meshes[mesh_list.mesh_indices[mesh_index]]).meshlet_count;
        const uint needed_uints = (mesh_meshlet_count + 32 - 1) / 32;
        visibility_lists.mesh_bitfield_offset[mesh_index] = atomicAdd(deref(u_visibility_bitfield_sratch), needed_uints) + 1 /*first uint is reserved for counter*/;
    }
    for (uint mesh_index = mesh_list.count; mesh_index < 7; ++mesh_index)
    {
        visibility_lists.mesh_bitfield_offset[mesh_index] = ~0;
    }
    deref(u_meshlet_visibilities[entity_index]) = visibility_lists;
}