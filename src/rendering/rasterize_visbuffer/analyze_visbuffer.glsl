#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "analyze_visbuffer.inl"
#include "visbuffer.glsl"
#include "visbuffer.glsl"

DAXA_DECL_PUSH_CONSTANT(AnalyzeVisbufferPush2, push)
layout(local_size_x = ANALYZE_VIS_BUFFER_WORKGROUP_X, local_size_y = ANALYZE_VIS_BUFFER_WORKGROUP_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    uint vis_id = INVALID_TRIANGLE_ID;
    if (index.x < push.width && index.y < push.height)
    {
        vis_id = texelFetch(daxa_utexture2D(u_visbuffer), index, 0).x;
    }
    if (vis_id == INVALID_TRIANGLE_ID)
    {
        return;
    }
    uint instantiated_meshlet_index;
    uint triangle_index;
    decode_triangle_id(vis_id, instantiated_meshlet_index, triangle_index);
    while (instantiated_meshlet_index != (~0))
    {
        const uint uniform_meshlet_id = subgroupBroadcastFirst(instantiated_meshlet_index);
        if (uniform_meshlet_id == instantiated_meshlet_index)
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
            instantiated_meshlet_index = (~0);
        }
    }
}
