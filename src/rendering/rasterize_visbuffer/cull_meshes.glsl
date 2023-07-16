#include <daxa/daxa.inl>
#extension GL_EXT_debug_printf : enable
#include "cull_meshes.inl"

#define my_sizeof(T) uint64_t(daxa_BufferPtr(T)(daxa_u64(0)) + 1)



#if defined(CullMeshesCommandBase_COMMAND)
layout(local_size_x = 32) in;
void main()
{
    if (gl_GlobalInvocationID.x == 0)
    {
        const uint entity_count = deref(u_entity_meta).entity_count;
        const uint dispatch_x = (entity_count + CULL_MESHES_WORKGROUP_X - 1) / CULL_MESHES_WORKGROUP_X;
        deref(u_command).x = dispatch_x;
        deref(u_command).y = 1;
        deref(u_command).z = 1;
    }
    const uint index = gl_GlobalInvocationID.x;
    uint64_t sizeof_arg_table = my_sizeof(MeshletCullIndirectArgTable);
    uint64_t sizeof_arg = my_sizeof(MeshletCullIndirectArg);
    // Each table is half the size of the previous one.
    // Table power 0 is the size of MAX_INSTANTIATED_MESHLETS.
    // index 0 starts at sizeof_arg_table + sizeof_arg * MAX_INSTANTIATED_MESHLETS,
    // indes 1 starts at sizeof_arg_table + sizeof_arg * MAX_INSTANTIATED_MESHLETS * 1/2,
    // index 2 ...
    daxa_u64 addr = daxa_u64(u_meshlet_cull_indirect_args) + sizeof_arg_table + (sizeof_arg * MAX_INSTANTIATED_MESHLETS >> index);
    deref(u_meshlet_cull_indirect_args).indirect_arg_ptrs[index] = daxa_RWBufferPtr(MeshletCullIndirectArg)(addr);
    deref(u_meshlet_cull_indirect_args).indirect_arg_counts[index] = 0;
}
#else
layout(local_size_x = CULL_MESHES_WORKGROUP_X, local_size_y = CULL_MESHES_WORKGROUP_Y) in;
void main()
{
    const uint entity_index = gl_GlobalInvocationID.x;
    const uint mesh_index = gl_LocalInvocationID.y;
    if (entity_index >= deref(u_entity_meta).entity_count)
    {
        return;
    }

    const uint mesh_id = deref(u_entity_meshlists[entity_index]).mesh_ids[mesh_index];
    const uint meshlet_count = deref(u_meshes[mesh_id]).meshlet_count;
    // TODO: Cull mesh.
    
    // How does this work?
    // - this is an asymertric work distribution problem
    // - each mesh cull thread needs x followup threads where x is the number of meshlets for the mesh
    // - writing x times to some argument buffer to dispatch over later is extreamly divergent and inefficient
    //   - solution is to combine writeouts in powers of two:
    //   - instead of x writeouts, only do log2(x), one writeout per set bit in the meshletcount.
    //   - when you want to write out 17 meshlet work units, instead of writing 7 args into a buffer, 
    //     you write one 1x arg, no 2x arg, no 4x arg, no 8x arg and one 16x arg. the 1x and the 16x args together contain 17 work units.
    // - still not good enough, in large cases like 2^20 - 1 it would need 19 writeouts
    //   - solution is to limit the writeouts to some smaller number (i chose 5, as it has a max thread waste of < 5%)
    //   - A strong compromise is to round up invocation count from meshletcount in such a way that the round up value only has 4 bits set at most.
    //   - as we do one writeout per bit set in meshlet count, this limits the writeout to 5.
    // - in worst case this can go down from thousands of divergent writeouts down to 5 while only wasting < 5% of invocations. 
    const uint MAX_BITS = 5;
    uint meshlet_count_msb = findMSB(meshlet_count);
    const uint shift = uint(max(0,int(meshlet_count_msb) - int(MAX_BITS)));
    // clip off all bits below the 5 most significant ones.
    uint clipped_bits_meshlet_count = (meshlet_count >> shift) << shift;
    // Need to round up if there were bits clipped.
    if (clipped_bits_meshlet_count < meshlet_count)
    {
        clipped_bits_meshlet_count += (1 << shift);
    }
    // Now bit by bit, do one writeout of an indirect command:
    uint writeout_bit_mask = clipped_bits_meshlet_count;
    // Each time we write out a command we add on the number of meshlets processed by that arg.
    uint meshlet_offset = 0;
    while (writeout_bit_mask != 0)
    {
        const uint writeout_power = findMSB(writeout_bit_mask);
        // Mask out bit.
        writeout_bit_mask &= writeout_power;
        const uint indirect_arg_meshlet_count = 1 << writeout_power;
        const uint arg_array_offset = atomicAdd(deref(u_meshlet_cull_indirect_args).indirect_arg_counts[writeout_power], 1);
    
        MeshletCullIndirectArg arg;
        arg.entity_id = entity_index;
        arg.mesh_id = mesh_id;
        arg.meshlet_index = meshlet_offset;
        arg.dummy = 0;
        deref(deref(u_meshlet_cull_indirect_args).indirect_arg_ptrs[writeout_power][arg_array_offset]) = arg;
        meshlet_offset += indirect_arg_meshlet_count;
    }
}
#endif
