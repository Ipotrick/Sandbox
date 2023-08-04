#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "write_swapchain.inl"
#include "visbuffer.glsl"

DAXA_DECL_PUSH_CONSTANT(WriteSwapchainPush, push)
layout(local_size_x = WRITE_SWAPCHAIN_WG_X, local_size_y = WRITE_SWAPCHAIN_WG_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    float checker = ((((index.x / (4)) & 1) == 0) ^^ (((index.y / (4)) & 1) == 0)) ? 1.0f : 0.9f;
    const float checker2 = ((((index.x / (4*8)) & 1) == 0) ^^ (((index.y / (4*4)) & 1) == 0)) ? 1.0f : 0.9f;
    checker *= checker2;
    const uint triangle_id = imageLoad(daxa_uimage2D(vis_image), index).x;
    if (triangle_id != INVALID_TRIANGLE_ID)
    {
        uint instantiated_meshlet_index;
        uint triangle_index;
        decode_triangle_id(triangle_id, instantiated_meshlet_index, triangle_index);
        MeshletInstance inst_meshlet = deref(u_instantiated_meshlets).meshlets[instantiated_meshlet_index];
        
        float f = float(inst_meshlet.entity_index * 10 + inst_meshlet.entity_meshlist_index) * 0.93213213232;
        vec3 debug_color = vec3(cos(f), cos(f+2), cos(f+4));
        debug_color = debug_color * 0.5 + 0.5;
        //debug_color *= fract(inst_meshlet.entity_index * 0.0111);

        vec4 result = vec4(0,0,0,1);

        imageStore(daxa_image2D(swapchain), index, vec4(debug_color * checker,1));
    }
    else
    {
        imageStore(daxa_image2D(swapchain), index, vec4(vec3(1,1,1) * checker,1));
    }
}