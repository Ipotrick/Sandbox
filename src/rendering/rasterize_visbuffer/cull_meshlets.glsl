#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "cull_meshlets.inl"

#include "cull_util.glsl"

DAXA_DECL_PUSH_CONSTANT(CullMeshletsPush,push)
layout(local_size_x = CULL_MESHLETS_WORKGROUP_X) in;
void main()
{
    const int tid = int(gl_GlobalInvocationID.x);
    const uint indirect_arg_index = tid >> push.indirect_args_table_id;
    const uint valid_arg_count = deref(u_meshlet_cull_indirect_args).indirect_arg_counts[push.indirect_args_table_id];
    if (tid == 0)
    {
        deref(u_draw_command).vertex_count = 3 * MAX_TRIANGLES_PER_MESHLET;
    }
    if (indirect_arg_index >= valid_arg_count)
    {
        return;
    }
    const uint arg_work_offset = tid - (indirect_arg_index << push.indirect_args_table_id);
    const MeshletCullIndirectArg arg = deref(deref(u_meshlet_cull_indirect_args).indirect_arg_ptrs[push.indirect_args_table_id][indirect_arg_index]);
    MeshletInstance instanced_meshlet;
    instanced_meshlet.entity_index = arg.entity_id;
    instanced_meshlet.mesh_id = arg.mesh_id;
    instanced_meshlet.mesh_index = arg.entity_meshlist_index;
    instanced_meshlet.entity_meshlist_index = arg.meshlet_index_start_offset + arg_work_offset;
#if ENABLE_MESHLET_CULLING
    if (is_meshlet_occluded(
        instanced_meshlet,
        u_entity_meshlet_visibility_bitfield_offsets,
        u_entity_meshlet_visibility_bitfield_arena,
        u_entity_combined_transforms,
        u_meshes,
        u_hiz))
    {
        return;
    }
#endif
    const uint out_index = atomicAdd(deref(u_instantiated_meshlets).second_count, 1);
    const uint offset = deref(u_instantiated_meshlets).first_count;
    deref(u_instantiated_meshlets).meshlets[out_index + offset] = instanced_meshlet;
    atomicAdd(deref(u_draw_command).instance_count, 1);
}