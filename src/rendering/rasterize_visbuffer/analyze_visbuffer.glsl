#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "analyze_visbuffer.inl"
#include "visbuffer.glsl"
#include "visbuffer.glsl"

void get_quad_unique_meshlets(uint meshlet_ids[4], out uint unique_meshlet_ids[4], out uint count)
{
    count = 0;
    [[unroll]]
    for (uint i = 0; i < 4; ++i)
    {
        if (meshlet_ids[i] != INVALID_MESHLET_INDEX)
        {
            bool unique = true;
            for (uint o = 0; o < count; ++o)
            {
                if (meshlet_ids[i] == unique_meshlet_ids[o])
                {
                    unique = false;
                }
            }
            if (unique)
            {
                unique_meshlet_ids[count] = meshlet_ids[i];
                ++count;
            }
        }
    }
}

DAXA_DECL_PUSH_CONSTANT(AnalyzeVisbufferPush2, push)
layout(local_size_x = ANALYZE_VIS_BUFFER_WORKGROUP_X, local_size_y = ANALYZE_VIS_BUFFER_WORKGROUP_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy) * 2;

    const ivec2 quad_indices[4] = { index + ivec2(0,1), index + ivec2(0,0), index + ivec2(1,1), index + ivec2(1,0) };
    uint inst_meshlet_indices[4] = { INVALID_MESHLET_INDEX, INVALID_MESHLET_INDEX, INVALID_MESHLET_INDEX, INVALID_MESHLET_INDEX };
    uvec4 triangle_indices = uvec4(0, 0, 0, 0);
    [[unroll]]
    for (uint quad_i = 0; quad_i < 4; ++quad_i)
    {
        if (quad_indices[quad_i].x < push.width && quad_indices[quad_i].y < push.height)
        {
            // TODO: make this a texture gather.
            const uint vis_id = texelFetch(daxa_utexture2D(u_visbuffer), quad_indices[quad_i], 0).x;
            if (vis_id != INVALID_TRIANGLE_ID)
            {
                uint instantiated_meshlet_index;
                uint triangle_index;
                decode_triangle_id(vis_id, instantiated_meshlet_index, triangle_index);
                inst_meshlet_indices[quad_i] = instantiated_meshlet_index;
                triangle_indices[quad_i] = triangle_index;
            }
        }
    }
    uint quad_unique_meshlet_id_count;
    uint quad_unique_meshlet_ids[4];
    get_quad_unique_meshlets(inst_meshlet_indices, quad_unique_meshlet_ids, quad_unique_meshlet_id_count);

    // In each subgroup, vote on uniform meshlet id, then do the masking. Loop as long as unique ids remain.
    uint left_meshlet_ids = quad_unique_meshlet_id_count;
    const uint debug_buffer_index = globals.frame_index & 1;
    while (left_meshlet_ids > 0)
    {
        const uint uniform_meshlet_id = subgroupBroadcastFirst(quad_unique_meshlet_ids[left_meshlet_ids-1]);
        uint quad_list_index = (~0u);
        for (uint i = 0; i < left_meshlet_ids; ++i)
        {
            if (uniform_meshlet_id == quad_unique_meshlet_ids[i])
            {
                quad_list_index = i;
                break;
            }
        }
        if (quad_list_index != (~0u))
        {
            if (subgroupElect())
            {
                const uint prev_value = atomicOr(deref(u_meshlet_visibility_bitfield[uniform_meshlet_id]), 1);
                if (prev_value == 0)
                {
                    const uint offset = atomicAdd(deref(u_visible_meshlets).count, 1);
                    deref(u_visible_meshlets).meshlet_ids[offset] = uniform_meshlet_id;
                }
            }
            // Remove element from quad list, by swapping it with the last element and then decrementing the list size.
            if (/*if not last element in list*/quad_list_index != (left_meshlet_ids-1))
            {
                // Swap element with last element, this way we remove the element from the list and preserve the last element,
                // when we decrement the list size.
                quad_unique_meshlet_ids[quad_list_index] = quad_unique_meshlet_ids[left_meshlet_ids-1];
            }
            --left_meshlet_ids;
        }
    }
}