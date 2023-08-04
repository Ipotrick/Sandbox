#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "analyze_visbuffer.inl"
#include "visbuffer.glsl"

DAXA_DECL_PUSH_CONSTANT(AnalyzeVisbufferPush2, push)
layout(local_size_x = ANALYZE_VIS_BUFFER_WORKGROUP_X, local_size_y = ANALYZE_VIS_BUFFER_WORKGROUP_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    uint inst_meshlet_indices[16];
    uint inst_meshlet_index_count = 0;
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            const ivec2 sub_i = ivec2(x, y);
            const ivec2 src_i = min(index * 4 + sub_i, ivec2(push.size) - 1);
            const uint vis_id = texelFetch(daxa_utexture2D(u_visbuffer), src_i, 0).x;
            uint instantiated_meshlet_index;
            uint triangle_index;
            decode_triangle_id(vis_id, instantiated_meshlet_index, triangle_index);
            bool new_id = vis_id != INVALID_TRIANGLE_ID;
            for (uint other_i = 0; other_i < inst_meshlet_index_count; ++other_i)
            {
                new_id = new_id && (inst_meshlet_indices[other_i] != instantiated_meshlet_index);
            }
            if (new_id)
            {
                inst_meshlet_indices[inst_meshlet_index_count] = instantiated_meshlet_index;
                inst_meshlet_index_count += 1;
            }
        }
    }
    // subgroup merged 
    for (uint iteration = 0; iteration < 8 && inst_meshlet_index_count > 0; ++iteration)
    {
        const uint uniform_meshlet_id = subgroupBroadcastFirst(inst_meshlet_indices[0]);
        {
            // If we have the elected id in our list, remove it.
            int found_index = -1;
            for (int i = 0; i < inst_meshlet_index_count; ++i)
            {
                if (uniform_meshlet_id == inst_meshlet_indices[i])
                {
                    found_index = i;
                }
            }
            if (found_index != -1)
            {
                inst_meshlet_indices[found_index] = inst_meshlet_indices[inst_meshlet_index_count - 1];
                inst_meshlet_index_count -= 1;
            }
        }
        if (subgroupElect())
        {
            const uint prev_value = atomicOr(deref(u_meshlet_visibility_bitfield[uniform_meshlet_id]), 1);
            if (prev_value == 0)
            {
                const uint offset = atomicAdd(deref(u_visible_meshlets).count, 1);
                deref(u_visible_meshlets).meshlet_ids[offset] = uniform_meshlet_id;
            }
        }
    }
    for (uint i = 0; i < inst_meshlet_index_count; ++i)
    {
        const uint meshlet_index = inst_meshlet_indices[i];
        const uint prev_value = atomicOr(deref(u_meshlet_visibility_bitfield[meshlet_index]), 1);
        if (prev_value == 0)
        {
            const uint offset = atomicAdd(deref(u_visible_meshlets).count, 1);
            deref(u_visible_meshlets).meshlet_ids[offset] = meshlet_index;
        }
    }
}
