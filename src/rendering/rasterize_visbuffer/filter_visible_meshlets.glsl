#pragma once

#include <daxa/daxa.inl>
#include "filter_visible_meshlets.inl"

shared uint gs_triangle_count;
shared uint gs_filtered_meshlet_index;
shared uint gs_triangle_buffer_offset;
layout(local_size_x = FILTER_VISIBLE_MESHLETS_DISPATCH_X)
void main()
{
    const uint src_triangle_index = gl_GlobalInvocationID.x;
    const uint src_meshlet_index = gl_GlobalInvocationID.y;
    const uvec4 visibility_bitmask = deref(u_meshlet_visibility_bitmasks[src_meshlet_index]);
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
    memoryBarrierShaded();

    const InstantiatedMeshlet src_inst_meshlet = deref(u_src_instantiated_meshlets).meshlets[src_meshlet_index];
    if (gl_GlobalInvocationID.x == 0)
    {
        gs_filtered_meshlet_index = atomidAdd(deref(u_filtered_meshlets).first_count.y, 1);
        gs_triangle_buffer_offset = atomicAdd(deref(u_filtered_triangles).count.vertex_count, gs_triangle_count * 3) / 3;
    }

    barrier();
    memoryBarrierShaded();

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
        TriangleDrawInfo out_triangle_info;
        out_triangle_info.meshlet_index = gs_filtered_meshlet_index;
        out_triangle_info.triangle_index = src_triangle_index;
        deref(u_filtered_triangles).triangles[out_index] = out_triangle_info;
    }
}