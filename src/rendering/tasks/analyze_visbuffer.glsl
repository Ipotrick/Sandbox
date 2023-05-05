#extension GL_EXT_debug_printf : enable

#define DAXA_ENABLE_IMAGE_OVERLOADS_BASIC 1
#include <daxa/daxa.inl>
#include "analyze_visbuffer.inl"
#include "../../mesh/mesh.inl"
#include "visbuffer.glsl"

void deduplicated_increment_meshlet_visibilities(uint meshlet_indices[4], daxa_RWBufferPtr(daxa_u32) instantiated_meshlet_counters)
{
    // deduplicate meshlet indices:
    uint unique_meshlet_indices[4] = {
        INVALID_MESHLET_INDEX,
        INVALID_MESHLET_INDEX,
        INVALID_MESHLET_INDEX,
        INVALID_MESHLET_INDEX
    };
    uint unique_indices_count = 0;
    [[unroll]]
    for (uint quad_i = 0; quad_i < 4; ++quad_i)
    {
        bool unique = true;
        for (uint unique_i = 0; unique_i < 4; ++unique_i)
        {
            unique = unique && (meshlet_indices[quad_i] != unique_meshlet_indices[unique_i]);
        }
        if (unique)
        {
            unique_meshlet_indices[unique_indices_count] = meshlet_indices[quad_i];
            unique_indices_count += 1;
        }
    }

    // Within each warp, write out each unique index ONCE.
    // This is achieved by selecting a unique index within the warp and then removing that index from all threads within that warp.
    // This way each warp will only atomicAdd ONCE per unique index it sees.
    // In the case of nvidia this means that the 16x8 window the warp sees, 
    // it will only perform n atomicAdds instead of 128, where n is the number of unique ids in the 16x8 screentile.
    bool need_write_out = unique_indices_count > 0;
    while (subgroupAny(need_write_out))
    {
        if (need_write_out)
        {
            // select a meshlet index
            const uint selected_index = subgroupBroadcastFirst(unique_meshlet_indices[0]);
            bool is_selected_in_samples = false;
            // All threads test if they have the selected id in their list of unique ids.
            // Those who find it in their list of unique ids, remove it from their list.
            [[unroll]]
            for (uint quad_i = 0; quad_i < unique_indices_count; ++quad_i)
            {
                const bool sample_is_selected = unique_meshlet_indices[quad_i] == selected_index;
                if (sample_is_selected)
                {
                    is_selected_in_samples = true;
                    // Move last sample into removed sample.
                    // This ensures that the list of unique indices stays packed.
                    if (unique_indices_count > 1)
                    {
                        unique_meshlet_indices[quad_i] = unique_meshlet_indices[unique_indices_count-1];
                    }
                    unique_indices_count -= 1;
                    need_write_out = unique_indices_count > 0;
                    break;
                }
            }
            // All threads that detected that they have the unique id,
            // elect one thread that perfoms the atomicAdd.
            if (is_selected_in_samples)
            {
                if (subgroupElect())
                {
                    atomicAdd(deref(instantiated_meshlet_counters[selected_index]), 1);
                }
            }
        }
    }
}

DAXA_USE_PUSH_CONSTANT(AnalyzeVisbufferPush, push)
layout(local_size_x = ANALYZE_VIS_BUFFER_WORKGROUP_X, local_size_y = ANALYZE_VIS_BUFFER_WORKGROUP_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 quad_offset = index << 1;

    const ivec2 quad_indices[4] = {
        quad_offset + ivec2(0,1),
        quad_offset + ivec2(0,0),
        quad_offset + ivec2(1,1),
        quad_offset + ivec2(1,0)
    };
    uint inst_meshlet_indices[4] = {
        INVALID_MESHLET_INDEX,
        INVALID_MESHLET_INDEX,
        INVALID_MESHLET_INDEX,
        INVALID_MESHLET_INDEX
    };
    uint triangle_indices[4] = {
        0,
        0,
        0,
        0
    };
    [[unroll]]
    for (uint quad_i = 0; quad_i < 4; ++quad_i)
    {
        if (quad_indices[quad_i].x < push.width && quad_indices[quad_i].y < push.height)
        {
            const uint vis_id = texelFetch(u_visbuffer, quad_indices[quad_i], 0).x;
            if (vis_id != INVALID_PIXEL_ID)
            {
                uint instantiated_meshlet_index;
                uint triangle_index;
                decode_pixel_id(vis_id, instantiated_meshlet_index, triangle_index);
                inst_meshlet_indices[quad_i] = instantiated_meshlet_index;
                triangle_indices[quad_i] = triangle_index;
            }
        }
    }

    deduplicated_increment_meshlet_visibilities(inst_meshlet_indices, u_instantiated_meshlet_counters);
}