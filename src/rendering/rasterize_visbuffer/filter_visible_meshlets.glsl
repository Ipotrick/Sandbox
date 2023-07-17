#pragma once

#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "filter_visible_meshlets.inl"

#if defined(FilterVisibleMeshletsCommandWriteBase_COMMAND)
layout(local_size_x = FILTER_VISIBLE_MESHLETS_DISPATCH_X) in;
void main()
{
    DispatchIndirectStruct command;
    command.x = deref(u_instantiated_meshlets_prev).first_pass_count + deref(u_instantiated_meshlets_prev).second_pass_count;
    command.y = 1;
    command.z = 1;
    deref(u_command) = command;
    deref(u_instantiated_meshlets).first_pass_count = 0;
}
#else

shared uint gs_triangle_count;
shared uint gs_filtered_meshlet_index;
shared uint gs_triangle_buffer_offset;
layout(local_size_x = FILTER_VISIBLE_MESHLETS_DISPATCH_X) in;
void main()
{
    // TODO
    return;
    const uint src_triangle_index = gl_LocalInvocationID.x;
    const uint src_meshlet_index = gl_WorkGroupID.x;
    const InstantiatedMeshlet src_inst_meshlet = deref(u_instantiated_meshlets_prev).meshlets[src_meshlet_index];
    const uint base_offset = deref(u_entity_visibility_bitfield_offsets_prev[src_inst_meshlet.entity_index]).mesh_bitfield_offset[src_inst_meshlet.mesh_index];
    if (base_offset == (~0))
    {
        return;
    }
    const uint meshlet_offset = base_offset + src_meshlet_index;
    const uvec4 visibility_bitmask = deref(u_entity_visibility_bitfield_prev[meshlet_offset]);
    const bool meshlet_visible = visibility_bitmask.x != 0 || 
                                visibility_bitmask.y != 0  || 
                                visibility_bitmask.z != 0  || 
                                visibility_bitmask.w != 0;
    if (!meshlet_visible)
    {
        return;
    }

    const uint bitmask_partition = src_triangle_index / 32;
    const uint bitmask_partition_bit = src_triangle_index % 32;
    const bool triangle_visible = (visibility_bitmask[bitmask_partition] & (1u << bitmask_partition_bit)) != 0;

    uint local_offset = 0;
    if (triangle_visible)
    {
        local_offset = atomicAdd(gs_triangle_count, 1);
    }

    barrier();
    memoryBarrierShared();

    if (gl_GlobalInvocationID.x == 0)
    {
        gs_filtered_meshlet_index = atomicAdd(deref(u_instantiated_meshlets).first_pass_count, 1);
        gs_triangle_buffer_offset = atomicAdd(deref(u_triangle_draw_list).count.vertex_count, gs_triangle_count * 3) / 3;
    }

    barrier();
    memoryBarrierShared();

    if (triangle_visible)
    {
        const uint out_index = gs_triangle_buffer_offset + local_offset;
        #if ENABLE_SHADER_PRINT_DEBUG
            if (out_index >= MAX_DRAWN_TRIANGLES)
            {
                debugPrintfEXT("max triangle count exceeded! idx: %u\n", out_index);
                return;
            }
        #endif // if ENABLE_SHADER_PRINT_DEBUG
        uint triangle_id = 0;
        encode_triangle_id(gs_filtered_meshlet_index, src_triangle_index, triangle_id);
        deref(u_triangle_draw_list).triangle_ids[out_index] = triangle_id;
    }
}
#endif