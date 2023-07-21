#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "cull_meshlets.inl"

#include "cull_util.glsl"

#if defined(CullMeshletsCommandWrite_COMMAND)
layout(local_size_x = 32) in;
void main()
{
    const uint index = gl_LocalInvocationID.x;
    const uint indirect_arg_count = deref(u_meshlet_cull_indirect_args).indirect_arg_counts[index];
    const uint threads_per_indirect_arg = 1 << index;
    const uint needed_threads = threads_per_indirect_arg * indirect_arg_count;
    const uint needed_workgroups = (needed_threads + CULL_MESHLETS_WORKGROUP_X - 1) / CULL_MESHLETS_WORKGROUP_X;
    DispatchIndirectStruct command;
    command.x = needed_workgroups;
    command.y = 1;
    command.z = 1;
    deref(u_commands[index]) = command;
    if (index == 0)
    {
        deref(u_instantiated_meshlets).second_count = 0;
    }
}
#else
DAXA_DECL_PUSH_CONSTANT(CullMeshletsPush,push)
layout(local_size_x = CULL_MESHLETS_WORKGROUP_X) in;
void main()
{
    const int tid = int(gl_GlobalInvocationID.x);
    const uint indirect_arg_index = tid >> push.indirect_args_table_id;
    const uint arg_work_offset = tid - (indirect_arg_index << push.indirect_args_table_id);
    const MeshletCullIndirectArg arg = deref(deref(u_meshlet_cull_indirect_args).indirect_arg_ptrs[push.indirect_args_table_id][indirect_arg_index]);
    InstantiatedMeshlet instanced_meshlet;
    instanced_meshlet.entity_index = arg.entity_id;
    instanced_meshlet.mesh_id = arg.mesh_id;
    instanced_meshlet.mesh_index = arg.entity_meshlist_index;
    instanced_meshlet.meshlet_index = arg.meshlet_index_start_offset + arg_work_offset;
    Mesh mesh_data = deref(u_meshes[instanced_meshlet.mesh_id]);
    if (instanced_meshlet.meshlet_index >= mesh_data.meshlet_count)
    {
        return;
    }
#if ENABLE_MESHLET_CULLING
    BoundingSphere bounds = deref(mesh_data.meshlet_bounds[instanced_meshlet.meshlet_index]);

    NdcBounds ndc_bounds;
    init_ndc_bounds(ndc_bounds);
    // construct bounding box from bounding sphere,
    // project each vertex of the box to ndc, min and max the coordinates.
    for (int z = -1; z <= 1; z += 2)
    {
        for (int y = -1; y <= 1; y += 2)
        {
            for (int x = -1; x <= 1; x += 2)
            {
                const vec3 bounding_box_corner_ws = bounds.center + bounds.radius * vec3(x,y,z);
                const vec4 projected_pos = globals.cull_camera_view_projection * vec4(bounding_box_corner_ws, 1);
                const vec3 ndc_pos = projected_pos.xyz / projected_pos.w;
                add_vertex_to_ndc_bounds(ndc_bounds, ndc_pos);
            }
        }
    }
    bool culled = !is_in_frustum(ndc_bounds);

    const uint bitfield_uint_offset = instanced_meshlet.meshlet_index / 32;
    const uint bitfield_uint_bit = 1u << (instanced_meshlet.meshlet_index % 32);
    const uint entity_arena_offset = u_entity_meshlet_visibility_bitfield_offsets.entity_offsets[instanced_meshlet.entity_index].mesh_bitfield_offset[instanced_meshlet.mesh_index];
    if (entity_arena_offset != ~0)
    {
        const uint mask = deref(u_entity_meshlet_visibility_bitfield_arena[entity_arena_offset + bitfield_uint_offset]);
        const bool visible_last_frame = (mask & bitfield_uint_bit) != 0;
        culled = culled || visible_last_frame;
    }

    if (!culled)
#endif
    {
        const uint out_index = atomicAdd(deref(u_instantiated_meshlets).second_count, 1);
        const uint offset = deref(u_instantiated_meshlets).first_count;
        deref(u_instantiated_meshlets).meshlets[out_index + offset] = instanced_meshlet;
    }
}
#endif