#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "analyze_visbuffer.inl"
#include "visbuffer.glsl"
#include "shared.inl"

#define SCALARIZED_ITERATIONS 4

vec2 make_gather_uv(vec2 inv_size, uvec2 top_left_index)
{
    return (vec2(top_left_index) + 1.0f) * inv_size;
}

#define QUAD_LOOP                                                 \
    [[unrol]] for (int y = 0; y < 2; ++y) \
        [[unrol]] for (int x = 0; x < 2; ++x)

DAXA_DECL_PUSH_CONSTANT(AnalyzeVisbufferPush2, push)
layout(local_size_x = ANALYZE_VIS_BUFFER_WORKGROUP_X, local_size_y = ANALYZE_VIS_BUFFER_WORKGROUP_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    uint unique_meshlet_indices[2 * 2] = { ~0, ~0, ~0, ~0 };
    uint unique_meshlet_index_count = 0;
    QUAD_LOOP
    {
        const ivec2 sampleIndex = min(index * ivec2(2, 2) + ivec2(x, y), ivec2(push.size) - 1);
        const uint vis_id = texelFetch(daxa_utexture2D(u_visbuffer), sampleIndex, 0).x;
        uint instantiated_meshlet_index;
        uint triangle_index;
        decode_triangle_id(vis_id, instantiated_meshlet_index, triangle_index);
        bool new_id = vis_id != INVALID_TRIANGLE_ID && 
            unique_meshlet_indices[0] != instantiated_meshlet_index &&
            unique_meshlet_indices[1] != instantiated_meshlet_index &&
            unique_meshlet_indices[2] != instantiated_meshlet_index &&
            unique_meshlet_indices[3] != instantiated_meshlet_index;
        if (new_id)
        {
            unique_meshlet_indices[unique_meshlet_index_count] = instantiated_meshlet_index;
            unique_meshlet_index_count += 1;
        }
    }
    [[loop]] for (uint iteration = 0; iteration < SCALARIZED_ITERATIONS && unique_meshlet_index_count > 0; ++iteration)
    {
        const uint uniform_meshlet_id = subgroupBroadcastFirst(unique_meshlet_indices[0]);
        {
            // If we have the elected id in our list, remove it.
            int found_index = -1;
            [[unrol]] for (int i = 0; i < (2 * 2); ++i)
            {
                if (uniform_meshlet_id == unique_meshlet_indices[i])
                {
                    found_index = i;
                }
            }
            if (found_index != -1)
            {
                unique_meshlet_indices[found_index] = unique_meshlet_indices[unique_meshlet_index_count - 1];
                unique_meshlet_indices[unique_meshlet_index_count-1] = ~0;
                unique_meshlet_index_count -= 1;
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
    [[loop]] for (uint i = 0; i < unique_meshlet_index_count; ++i)
    {
        const uint meshlet_index = unique_meshlet_indices[i];
        const uint prev_value = atomicOr(deref(u_meshlet_visibility_bitfield[meshlet_index]), 1);
        if (prev_value == 0)
        {
            const uint offset = atomicAdd(deref(u_visible_meshlets).count, 1);
            deref(u_visible_meshlets).meshlet_ids[offset] = meshlet_index;
        }
    }
}
