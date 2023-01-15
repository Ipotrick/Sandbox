#include "../../../shaders/util.glsl"

#include "prefix_sum.inl"

DEFINE_PUSHCONSTANT(PrefixSumMeshletCountPush, push)
layout(local_size_x = PREFIX_SUM_WORKGROUP_SIZE) in;
void main()
{
    const uint entity_index = gl_GlobalInvocationID.x;
    const uint warp_id = gl_SubgroupID;
    const uint warp_index = gl_SubgroupInvocationID;
    uint meshlets = 0;
    if (entity_index < deref(push.entities).entity_count)
    {
        const uint mesh_count = deref(push.entities).meshes_count[entity_index];
        for (uint mesh_i = 0; mesh_i < mesh_count; ++mesh_i)
        {
            const uint mesh_index = deref(push.entities).meshes[entity_index][mesh_i];
            meshlets += deref(push.meshes).meshlet_count;
        }
    }
    prefix_sum(
        warp_index,
        warp_id,
        meshlets);
    if (entity_index < deref(push.entities).entity_count)
    {
        deref(push.dst) = meshlets;
    }
}