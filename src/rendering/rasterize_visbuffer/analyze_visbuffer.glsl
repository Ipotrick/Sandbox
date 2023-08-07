#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "analyze_visbuffer.inl"
#include "visbuffer.glsl"
#include "shared.inl"

// MUST BE SMALLER EQUAL TO WARP_SIZE!
#define UNIQUE_VALUE_LIST 32

vec2 make_gather_uv(vec2 inv_size, uvec2 top_left_index)
{
    return (vec2(top_left_index) + 1.0f) * inv_size;
}

DAXA_DECL_PUSH_CONSTANT(AnalyzeVisbufferPush2, push)
layout(local_size_x = ANALYZE_VIS_BUFFER_WORKGROUP_X, local_size_y = ANALYZE_VIS_BUFFER_WORKGROUP_Y) in;
void main()
{
    // How does this Work?
    // Problem:
    //   We need to atomic or the visibility mask of each meshlet instance to mark it as visible.
    //   Then, we test if it not already visible before that atomic or.
    //   If not, we write its id to the visible meshlet list.
    // Naive Solution:
    //   Launch an invocation per pixel, do the atomicOr and conditional append per thread.
    //   VERY slow, as it has brutal atomic contention.
    // Better Solution:
    //   Launch one invoc per pixel, then find the minimal list of unique values from the values of each thread in the warp. 
    //     Find unique value list:
    //       In a loop, elect a value from the threads, 
    //       each thread checks if they have the elected value, if so mark to be done (when marked as done, thread does not participate in vote anymore),
    //       write out value to list.
    //       early out when all threads are marked done.
    //     Writeout:
    //       Each thread in the warp takes 1 value from the list and does the atomic ops coherently.
    // Even Better Solution:
    //   Take the better solution with the twist, that each thread takes 4 pixels instead of one, tracking them with a bitmask per thread in the unique list generation.
    //   Still do only up to WARP_SIZE iterations in the loop with a list size of WARP_SIZE.
    //   In the end threads can still have values left in a worst case (because we read WARPSIZE*4 pixels but the list can only hold N values), so write the rest out divergently but coherent.
    //   In 90% of warps the list covers all WARPSIZE*4 pixels, so the writeout is very coherent and atomic op count greatly reduced as a result.
    //   Around 26x faster even in scenes with lots of small meshlets.
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 sampleIndex = min(index << 1, ivec2(push.size) - 1);
    uvec4 vis_ids = textureGather(daxa_usampler2D(u_visbuffer, globals.samplers.linear_clamp), make_gather_uv(1.0f / push.size, sampleIndex), 0);
    uint id_mask = (vis_ids[0] != ~0 ? 1 : 0) | (vis_ids[1] != ~0 ? 2 : 0) | (vis_ids[2] != ~0 ? 4 : 0) | (vis_ids[3] != ~0 ? 8 : 0);
    [[unroll]] for (uint i = 0; i < 4; ++i)
    {
        vis_ids[i] = vis_ids[i] >> 7;
    }
    uint assigned_id = ~0;
    uint assigned_id_count = 0;
    for (; assigned_id_count < UNIQUE_VALUE_LIST && subgroupAny(id_mask != 0); ++assigned_id_count)
    {
        const bool lane_on = id_mask != 0;
        const uint voted_for_id = lane_on ? vis_ids[findLSB(id_mask)] : ~0;
        const uint elected_id = subgroupBroadcast(voted_for_id, subgroupBallotFindLSB(subgroupBallot(lane_on)));
        // If we have the elected id in our list, remove it.
        [[unroll]] for (uint i = 0; i < 4; ++i)
        {
            if (vis_ids[i] == elected_id)
            {
                id_mask &= ~(1u << i);
            }
        }
        if (assigned_id_count == gl_SubgroupInvocationID.x)
        {
            assigned_id = elected_id;
        }
    }
    // Write out 
    if (gl_SubgroupInvocationID.x < assigned_id_count)
    {
        const uint prev_value = atomicOr(deref(u_meshlet_visibility_bitfield[assigned_id]), 1);
        if (prev_value == 0)
        {
            const uint offset = atomicAdd(deref(u_visible_meshlets).count, 1);
            deref(u_visible_meshlets).meshlet_ids[offset] = assigned_id;
        }
    }
    // Write out rest of local meshlet list:
    [[loop]] while (id_mask != 0)
    {
        const uint lsb = findLSB(id_mask);
        const uint meshlet_index = vis_ids[lsb];
        id_mask &= ~(1 << lsb);
        const uint prev_value = atomicOr(deref(u_meshlet_visibility_bitfield[meshlet_index]), 1);
        if (prev_value == 0)
        {
            const uint offset = atomicAdd(deref(u_visible_meshlets).count, 1);
            deref(u_visible_meshlets).meshlet_ids[offset] = meshlet_index;
        }
    }
}