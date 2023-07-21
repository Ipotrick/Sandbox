#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "prepopulate_inst_meshlets.inl"
#include "cull_util.glsl"

#define WORKGROUP_SIZE 1024

#if defined(PrepopulateInstantiatedMeshletsCommandWrite_COMMAND)
layout(local_size_x = 1) in;
void main()
{
    const uint needed_threads = deref(u_visible_meshlets_prev).count;
    const uint needed_workgroups = (needed_threads + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    DispatchIndirectStruct command;
    command.x = needed_workgroups;
    command.y = 1;
    command.z = 1;
    deref(u_command) = command;
}
#elif defined(SetEntityMeshletVisibilityBitMasks_SHADER)
layout(local_size_x = WORKGROUP_SIZE) in;
void main()
{
    const uint count = deref(u_instantiated_meshlets).first_count;
    const uint inst_meshlet_index = gl_GlobalInvocationID.x;
    const bool thread_active = inst_meshlet_index < count;
    // TODO: check if entity, its mesh id and meshlets are valid
    if (thread_active)
    {
        InstantiatedMeshlet inst_meshlet = deref(u_instantiated_meshlets).meshlets[inst_meshlet_index];
        const uint mask_bit = 1u << (inst_meshlet.meshlet_index % 32);
        const uint local_mask_offset = inst_meshlet.meshlet_index / 32;
        const uint global_mask_offset = u_entity_meshlet_visibility_bitfield_offsets
                                        .entity_offsets[inst_meshlet.entity_index]
                                        .mesh_bitfield_offset[inst_meshlet.mesh_index];
        const uint offset = global_mask_offset + local_mask_offset;

        atomicOr(deref(u_entity_meshlet_visibility_bitfield_arena[offset]), mask_bit);
    }
}
#else
shared uint s_out_count;
shared uint s_out_offset;
layout(local_size_x = WORKGROUP_SIZE) in;
void main()
{
    const uint count =  deref(u_visible_meshlets_prev).count;
    const bool thread_active = gl_GlobalInvocationID.x < count;
    // TODO: check if entity, its mesh id and meshlets are valid
    uint inst_meshlet_index_prev = 0;
    InstantiatedMeshlet inst_meshlet;
    if (thread_active)
    {
        inst_meshlet_index_prev = deref(u_visible_meshlets_prev).meshlet_ids[gl_GlobalInvocationID.x];
        inst_meshlet = deref(u_instantiated_meshlets_prev).meshlets[inst_meshlet_index_prev];
        const uint counters_index = inst_meshlet.entity_index * 8 + inst_meshlet.mesh_index;
        const uint prev_value = atomicAdd(deref(u_entity_visibility_counters[counters_index]), 1);
        // Saw entity (and entity mesh index) the first time -> allocate bitfield offset
        if (prev_value == 0)
        {
            const uint meshlets_in_mesh = deref(u_meshes[inst_meshlet.mesh_id]).meshlet_count;
            if (meshlets_in_mesh > 0)
            {
                const uint needed_uints_in_bitfield = (meshlets_in_mesh + 32 - 1) / 32;
                const uint bitfield_arena_offset = atomicAdd(u_entity_meshlet_visibility_bitfield_offsets.back_offset, needed_uints_in_bitfield);
                u_entity_meshlet_visibility_bitfield_offsets.entity_offsets[inst_meshlet.entity_index].mesh_bitfield_offset[inst_meshlet.mesh_index] = bitfield_arena_offset;
            }
        }
    }
    
    if (gl_LocalInvocationID.x == 0)
    {
        s_out_count = 0;
    }
    memoryBarrierShared();
    barrier();
    uint local_offset = 0;
    if (thread_active)
    {
        local_offset = atomicAdd(s_out_count, 1);
    }
    memoryBarrierShared();
    barrier();
    if (gl_LocalInvocationID.x == 0)
    {
        s_out_offset = atomicAdd(deref(u_instantiated_meshlets).first_count, s_out_count);
    }
    memoryBarrierShared();
    barrier();
    if (thread_active)
    {
        deref(u_instantiated_meshlets).meshlets[s_out_offset+local_offset] = inst_meshlet;
    }
}
#endif