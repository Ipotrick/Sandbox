#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>
#include "gen_hiz.inl"
#include "cull_util.glsl"

vec2 make_gather_uv(vec2 inv_size, uvec2 top_left_index)
{
    return (vec2(top_left_index) + 1.0f) * inv_size;
}

DAXA_DECL_PUSH_CONSTANT(GenHizPush, push)
// For some reason, glsl multi dim array order is inverted compared to c... 
shared float s_mins[2][GEN_HIZ_Y][GEN_HIZ_X];
layout(local_size_x = GEN_HIZ_X, local_size_y = GEN_HIZ_Y) in;
void main()
{
    // DAXA MAGIC
    daxa_ImageViewId out_images[GEN_HIZ_LEVELS_PER_DISPATCH] = push.dst;
    const uvec2 glob_index         = gl_GlobalInvocationID.xy;
    const uvec2 local_i            = gl_LocalInvocationID.xy;
    const uvec2 src_index          = gl_GlobalInvocationID.xy * 2;
    const uvec2 size               = uvec2(push.width,push.height);
    vec4 fetch;
    if (push.sample_depth == 1)
    {
        fetch = textureGather(daxa_sampler2DShadow(push.src, globals.samplers.linear_clamp), make_gather_uv(1.0f / vec2(size), src_index), 0);
    }
    else
    {
        fetch = textureGather(daxa_sampler2D(push.src, globals.samplers.linear_clamp), make_gather_uv(1.0f / vec2(size), src_index), 0);
    }
    const float min_v           = min(min(fetch.x, fetch.y), min(fetch.z, fetch.w));
    const uvec2 glob_wg_offset0 = uvec2(GEN_HIZ_X,GEN_HIZ_Y) * gl_WorkGroupID.xy;
    imageStore(daxa_image2D(out_images[0]), ivec2(glob_index), vec4(min_v,0,0,0));
    s_mins[0][local_i.y][local_i.x] = min_v;

    [[unroll]]
    for (uint i = 1; i < push.mips; ++i)
    {
        const uint ping_pong_src_index = ((i+1) & 1u);
        const uint ping_pong_dst_index = (i & 1u);
        memoryBarrierShared();
        barrier();
        const bool invoc_active = local_i.x < (GEN_HIZ_X>>i) && local_i.y < (GEN_HIZ_Y>>i);
        if (invoc_active)
        {
            const uvec2 glob_wg_offset = glob_wg_offset0 >> i;
            const uvec2 src_i          = local_i * 2;
            const vec4  fetch          = vec4(
                s_mins[ping_pong_src_index][src_i.y+0][src_i.x+0],
                s_mins[ping_pong_src_index][src_i.y+0][src_i.x+1],
                s_mins[ping_pong_src_index][src_i.y+1][src_i.x+0],
                s_mins[ping_pong_src_index][src_i.y+1][src_i.x+1]
            );
            const float min_v = min(min(fetch.x,fetch.y), min(fetch.z,fetch.w));
            imageStore(daxa_image2D(out_images[i]), ivec2(glob_wg_offset + local_i), vec4(min_v,0,0,0));
            s_mins[ping_pong_dst_index][local_i.y][local_i.x] = min_v;
        }
    }
}