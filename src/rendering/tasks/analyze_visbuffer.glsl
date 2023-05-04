#extension GL_EXT_debug_printf : enable

#define DAXA_ENABLE_IMAGE_OVERLOADS_BASIC 1
#include <daxa/daxa.inl>
#include "visbuffer.glsl"
#include "analyze_visbuffer.inl"

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
    uint inst_meshlet_indices[4];
    uint triangle_indices[4];
    uint samples = 0;
    [[unroll]]
    for (uint quad_i = 0; quad_i < 4; ++quad_i)
    {
        if (quad_indices[quad_i].x < push.width && quad_indices[quad_i].y < push.height)
        {
            const uint vis_id = texelFetch(u_visbuffer, quad_indices[quad_i], 0).x;
            uint instantiated_meshlet_index;
            uint triangle_index;
            decode_pixel_id(vis_id, instantiated_meshlet_index, triangle_index);

            bool is_new_id = true;
            for (uint stored_i = 0; stored_i < samples; ++stored_i)
            {
                is_new_id = is_new_id && 
                            (inst_meshlet_indices[stored_i] != instantiated_meshlet_index &&
                            triangle_indices[stored_i] != triangle_index);
            }
            if (is_new_id)
            {
                inst_meshlet_indices[samples] = instantiated_meshlet_index;
                triangle_indices[samples] = triangle_index;
                samples += 1;
            }
        }
    }

    bool need_write_out = samples > 0;
    while (subgroupAny(need_write_out))
    {
        if (need_write_out)
        {
            const uint inst_meshlet_index = inst_meshlet_indices[samples - 1];
            uint selected_index = subgroupBroadcastFirst(inst_meshlet_index);
            if (selected_index == inst_meshlet_index)
            {
                if (subgroupElect())
                {
                    atomicAdd(deref(u_instantiated_meshlet_counters[inst_meshlet_index]), 1);
                }
                samples -= 1;
                need_write_out = samples > 0;
            }
        }
    }
}