#include "../../../shaders/util.glsl"

#include "prefix_sum.inl"

DEFINE_PUSHCONSTANT(PrefixSumMeshletCountPush, push)
layout(local_size_x = PREFIX_SUM_WORKGROUP_SIZE) in;
void main()
{
    const uint entity_index = gl_GlobalInvocationID.x;
    const uint warp_id = gl_SubgroupID;
    const uint warp_index = gl_SubgroupInvocationID;

    daxa_BufferPtr(EntityData) entities = daxa_BufferPtr(EntityData)(push.entities);
    daxa_BufferPtr(Mesh) meshes = daxa_BufferPtr(Mesh)(push.meshes);
    daxa_RWBufferPtr(daxa_u32) dst = daxa_RWBufferPtr(daxa_u32)(push.dst);

    uint meshlets = 0;
    if (entity_index < deref(entities).entity_count)
    {
        const MeshList meshlist = deref(entities).meshes[entity_index];
        for (uint mesh_i = 0; mesh_i < meshlist.count; ++mesh_i)
        {
            const uint mesh_index = meshlist.mesh_indices[mesh_i];
            meshlets += deref(meshes[mesh_index]).meshlet_count;
        }
    }
    prefix_sum(
        warp_index,
        warp_id,
        meshlets);
    deref(dst[entity_index]) = meshlets;
}